"""Import de factures fournisseurs PDF dans Dolibarr, réglées par prélèvement.

Le workflow prend en entrée une bibliothèque Mistral contenant des PDF de factures
fournisseurs et un nom de fournisseur. Pour chaque PDF :
  1. extraction des données de la facture (OCR bibliothèque + modèle structuré) ;
  2. création de la facture fournisseur dans Dolibarr (avec anti-doublon) ;
  3. validation de la facture ;
  4. règlement par prélèvement à la date de facturation, sur le compte bancaire choisi ;
  5. liaison du PDF source à la facture, puis suppression du PDF de la bibliothèque.
"""

import base64
import json
import re
from datetime import datetime, timezone
from typing import Any

import mistralai.workflows as workflows
import pydantic
from mistralai.workflows.client import get_mistral_client
from mistralai.workflows.core.auth.run_as import RunAs

with workflows.workflow.unsafe.imports_passed_through():
    import dateparser

    from worker.dolibarr import DolibarrClient, DolibarrError

EXTRACTION_MODEL = "mistral-medium-latest"
DEFAULT_VAT_RATE = 20.0
DEFAULT_PAYMENT_MODE_CODE = "PRE"


class InvoiceLine(pydantic.BaseModel):
    description: str
    quantity: float = 1.0
    unit_price_ht: float | None = None
    unit_price_ttc: float | None = None
    line_total_ht: float | None = None
    vat_rate: float | None = None


class ExtractedInvoice(pydantic.BaseModel):
    is_invoice: bool = True
    invoice_date: str | None = pydantic.Field(
        default=None,
        description=(
            "Date de facturation recopiée exactement comme écrite sur la facture "
            "(ex. « 28 Décembre 2016 », « 28/12/2016 », « 2016-12-28 »)"
        ),
    )
    supplier_ref: str | None = pydantic.Field(
        default=None, description="Numéro de la facture chez le fournisseur"
    )
    lines: list[InvoiceLine] = pydantic.Field(default_factory=list)
    total_ht: float | None = None
    total_ttc: float | None = None
    currency: str | None = None


class DocumentResult(pydantic.BaseModel):
    document: str
    status: str
    invoice_id: int | None = None
    invoice_ref: str | None = None
    amount_paid: float | None = None
    detail: str | None = None


class ImportResult(pydantic.BaseModel):
    supplier: str
    supplier_id: int | None = None
    supplier_created: bool = False
    bank_account_id: int
    payment_mode: str
    documents: list[DocumentResult] = pydantic.Field(default_factory=list)


class SupplierInvoiceImportInput(pydantic.BaseModel):
    library_name_or_id: str = pydantic.Field(
        description="Nom ou ID de la bibliothèque Mistral contenant les PDF de factures"
    )
    supplier_name: str = pydantic.Field(description="Nom du fournisseur dans Dolibarr")
    bank_account_id: int = pydantic.Field(
        default=1, description="ID du compte bancaire Dolibarr à débiter"
    )
    payment_mode_code: str = pydantic.Field(
        default=DEFAULT_PAYMENT_MODE_CODE,
        description="Code du mode de paiement Dolibarr (PRE = prélèvement)",
    )
    create_missing_supplier: bool = pydantic.Field(
        default=True,
        description="Créer le tiers fournisseur s'il n'existe pas encore",
    )
    document_ids: list[str] = pydantic.Field(
        default_factory=list,
        description="Restreindre l'import à ces IDs de documents (tous les PDF si vide)",
    )
    max_documents: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Limiter le nombre de PDF traités, dans l'ordre de la bibliothèque (pratique pour les tests)",
    )
    payment_mode_id: int | None = pydantic.Field(
        default=None,
        description=(
            "ID Dolibarr du mode de paiement (table llx_c_paiement). S'il est fourni, "
            "la lecture des dictionnaires est évitée (utile si l'API renvoie 403)"
        ),
    )
    replace_mismatched_duplicates: bool = pydantic.Field(
        default=True,
        description=(
            "Si un doublon (même ref fournisseur) existe avec un total différent, "
            "importer quand même la version corrigée (l'ancienne facture reste à corriger)"
        ),
    )
    dry_run: bool = pydantic.Field(
        default=False,
        description="Si vrai : extraction seule, aucune écriture dans Dolibarr",
    )


def _dolibarr_client() -> DolibarrClient:
    """Construit le client REST Dolibarr à partir de l'environnement du worker."""
    return DolibarrClient()


def _created_id(created: Any) -> int | None:
    """Extrait l'ID d'une création Dolibarr (l'API renvoie un entier ou un objet)."""
    if isinstance(created, bool):
        return None
    if isinstance(created, int):
        return created
    if isinstance(created, str) and created.strip().isdigit():
        return int(created.strip())
    if isinstance(created, dict) and created.get("id") is not None:
        return int(created["id"])
    return None


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_date(raw: Any) -> str | None:
    """Normalise une date telle qu'écrite sur une facture en YYYY-MM-DD.

    Utilise dateparser (français et anglais, jour en premier) : gère « 28/12/2016 »,
    « 28.12.2016 », « 28 Décembre 2016 », « 1er Mars 2025 », « 28 December 2016 ».
    """
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    parsed = dateparser.parse(
        value,
        languages=["fr", "en"],
        settings={"DATE_ORDER": "DMY", "PREFER_DAY_OF_MONTH": "first"},
    )
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%d")


_TEXT_DATE_PATTERN = (
    r"\d{1,2}(?:er)?\s+[A-Za-z\u00c0-\u00ff]+\s+\d{4}"
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{4}"
    r"|\d{4}-\d{1,2}-\d{1,2}"
)
_TEXT_DATE_KEYWORD_RE = re.compile(
    r"(?:date\s+d[\u2019']\u00e9mission|date\s+de\s+facturation|date\s+de\s+facture"
    r"|facture[^:\n]{0,40}\bdu|date\s*:)\s*:?\s*(" + _TEXT_DATE_PATTERN + ")",
    re.IGNORECASE,
)
_TEXT_DATE_ANY_RE = re.compile(_TEXT_DATE_PATTERN)


def _search_dates_in_text(text: str) -> str | None:
    """Dernier secours : dateparser.search repère les dates dans le texte libre
    (formats non couverts par les regex, ex. années à deux chiffres)."""
    from dateparser.search import search_dates

    try:
        found = search_dates(
            text[:15000],
            languages=["fr", "en"],
            settings={"DATE_ORDER": "DMY", "PREFER_DAY_OF_MONTH": "first"},
        )
    except Exception:
        return None
    for _, parsed in found or []:
        if 2000 <= parsed.year <= 2100:
            return parsed.strftime("%Y-%m-%d")
    return None


def _find_date_in_text(text: str) -> str | None:
    """Recherche déterministe d'une date de facturation dans le texte OCR (secours)."""
    match = _TEXT_DATE_KEYWORD_RE.search(text)
    if match:
        normalized = _normalize_date(match.group(1))
        if normalized:
            return normalized
    for match in _TEXT_DATE_ANY_RE.finditer(text):
        normalized = _normalize_date(match.group(0))
        if normalized:
            return normalized
    return _search_dates_in_text(text)


def _loads_json_object(content: str | None) -> dict[str, Any]:
    if not content or not content.strip():
        raise ValueError("réponse vide du modèle")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match is None:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"JSON inattendu : {type(data).__name__}")
    return data


def _date_to_timestamp(iso_date: str | None) -> int | None:
    if not iso_date:
        return None
    try:
        return int(datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def _error_text(error: BaseException) -> str:
    """Détaille une exception (chaîne de causes incluse) pour le rapport par document."""
    parts: list[str] = []
    current: BaseException | None = error
    while current is not None:
        text = str(current)
        if text and text not in parts:
            parts.append(text)
        current = current.__cause__
    return " | ".join(parts) or type(error).__name__


def _line_ht_amount(line: InvoiceLine) -> float | None:
    """Total HT d'une ligne extraite, par la source la plus fiable disponible."""
    if line.line_total_ht is not None:
        return line.line_total_ht
    vat_rate = line.vat_rate if line.vat_rate is not None else DEFAULT_VAT_RATE
    if line.unit_price_ht is not None:
        return line.unit_price_ht * (line.quantity or 1.0)
    if line.unit_price_ttc is not None:
        return line.unit_price_ttc / (1.0 + vat_rate / 100.0) * (line.quantity or 1.0)
    return None


def _amount_warnings(invoice: ExtractedInvoice) -> list[str]:
    """Signale les incoherences entre lignes et totaux (denonciation d'erreur OCR/modelle)."""
    warnings: list[str] = []
    if invoice.lines and invoice.total_ht is not None:
        amounts = [amount for line in invoice.lines if (amount := _line_ht_amount(line)) is not None]
        if amounts:
            sum_lines = round(sum(amounts), 2)
            if abs(sum_lines - invoice.total_ht) > 0.05:
                warnings.append(
                    f"Somme des lignes ({sum_lines}) differente du total HT ({invoice.total_ht})"
                )
    if (
        invoice.total_ht is not None
        and invoice.total_ttc is not None
        and invoice.total_ttc < invoice.total_ht
    ):
        warnings.append(
            f"Total TTC ({invoice.total_ttc}) inferieur au total HT ({invoice.total_ht})"
        )
    return warnings


def _compute_lines(invoice: ExtractedInvoice) -> list[dict[str, Any]]:
    if invoice.lines:
        lines = []
        for line in invoice.lines:
            vat_rate = line.vat_rate if line.vat_rate is not None else DEFAULT_VAT_RATE
            pu_ht = line.unit_price_ht
            if pu_ht is None and line.unit_price_ttc is not None:
                pu_ht = line.unit_price_ttc / (1.0 + vat_rate / 100.0)
            if pu_ht is None and line.line_total_ht is not None and line.quantity:
                pu_ht = line.line_total_ht / line.quantity
            if pu_ht is None:
                pu_ht = 0.0
            lines.append(
                {
                    "description": line.description,
                    "qty": line.quantity,
                    "pu_ht": round(pu_ht, 4),
                    "tva_tx": vat_rate,
                }
            )
        return lines

    if invoice.total_ttc is None and invoice.total_ht is None:
        raise ValueError("Aucun montant exploitable trouvé dans la facture")

    if invoice.total_ht is not None and invoice.total_ttc is not None and invoice.total_ht > 0:
        vat_rate = round((invoice.total_ttc - invoice.total_ht) / invoice.total_ht * 100.0, 2)
        pu_ht = invoice.total_ht
    else:
        vat_rate = DEFAULT_VAT_RATE
        base = invoice.total_ttc if invoice.total_ttc is not None else invoice.total_ht
        pu_ht = base / (1.0 + vat_rate / 100.0)
    return [
        {
            "description": f"Facture {invoice.supplier_ref or 'fournisseur'}",
            "qty": 1.0,
            "pu_ht": round(pu_ht, 4),
            "tva_tx": vat_rate,
        }
    ]


@workflows.activity(name="dolibarr-resolve-supplier")
async def resolve_supplier(
    supplier_name: str,
    create_if_missing: bool,
    allow_missing: bool = False,
) -> dict[str, Any]:
    dolibarr = _dolibarr_client()
    candidates = await dolibarr.list_thirdparties(
        sqlfilters=f"((t.nom:like:'{supplier_name}%') or (t.name_alias:like:'{supplier_name}%'))",
        limit=100,
    )
    for thirdparty in candidates:
        if _norm(thirdparty.get("name") or thirdparty.get("nom")) == _norm(supplier_name):
            return {"id": int(thirdparty["id"]), "name": thirdparty.get("name"), "created": False}

    if not create_if_missing:
        if allow_missing:
            return {"id": None, "name": supplier_name, "created": False, "found": False}
        raise ValueError(
            f"Fournisseur '{supplier_name}' introuvable dans Dolibarr et création non autorisée"
        )

    created = await dolibarr.create_thirdparty({"name": supplier_name, "fournisseur": 1})
    created_id = _created_id(created)
    if created_id is None:
        raise ValueError(f"Création du fournisseur '{supplier_name}' impossible : {created!r}")
    return {"id": created_id, "name": supplier_name, "created": True}


@workflows.activity(name="dolibarr-resolve-payment-mode")
async def resolve_payment_mode(
    payment_mode_code: str,
    payment_mode_id: int | None = None,
) -> dict[str, Any]:
    if payment_mode_id is not None:
        return {"id": payment_mode_id, "label": payment_mode_code}
    dolibarr = _dolibarr_client()
    try:
        payment_types = await dolibarr.list_payment_types()
    except DolibarrError as error:
        if error.status_code == 403:
            raise ValueError(
                "L'utilisateur API Dolibarr n'a pas le droit de lire les dictionnaires "
                "(setup) : accordez-lui la lecture des factures, propales ou commandes dans "
                "Dolibarr (Utilisateurs & Groupes → permissions), ou passez l'ID du mode de "
                "paiement via payment_mode_id (table llx_c_paiement)"
            ) from error
        raise
    for payment_type in payment_types:
        if _norm(payment_type.get("code")) == _norm(payment_mode_code):
            return {"id": int(payment_type["id"]), "label": payment_type.get("label", payment_mode_code)}
    for payment_type in payment_types:
        if "prélèvement" in _norm(payment_type.get("label")) or "prelevement" in _norm(
            payment_type.get("label")
        ):
            return {"id": int(payment_type["id"]), "label": payment_type.get("label")}
    available = ", ".join(
        f"{pt.get('code', '?')}={pt.get('label', '?')} (id {pt.get('id', '?')})"
        for pt in payment_types
    )
    raise ValueError(
        f"Mode de paiement '{payment_mode_code}' introuvable. Disponibles : {available or 'aucun'}. "
        "Sinon, passez l'ID exact via payment_mode_id (--payment-mode-id)"
    )


@workflows.activity(name="dolibarr-resolve-bank-account")
async def resolve_bank_account(
    bank_account_id: int,
) -> dict[str, Any]:
    dolibarr = _dolibarr_client()
    accounts = await dolibarr.list_bank_accounts()
    for account in accounts:
        if int(account.get("id", -1)) == bank_account_id:
            return {"id": bank_account_id, "label": account.get("label", str(bank_account_id))}
    available = ", ".join(f"{a.get('label', '?')} (id {a.get('id', '?')})" for a in accounts)
    raise ValueError(f"Compte bancaire id {bank_account_id} introuvable. Comptes : {available or 'aucun'}")


@workflows.activity(name="library-list-invoice-pdfs")
async def list_library_pdfs(
    library_name_or_id: str,
    document_ids: list[str],
) -> dict[str, Any]:
    client = get_mistral_client(run_as=RunAs.AUTO)
    libraries = (await client.beta.libraries.list_async()).data
    library = None
    for candidate in libraries:
        if candidate.id == library_name_or_id or candidate.name == library_name_or_id:
            library = candidate
            break
    if library is None:
        available = ", ".join(lib.name for lib in libraries)
        raise ValueError(f"Bibliothèque '{library_name_or_id}' introuvable. Bibliothèques : {available or 'aucune'}")

    wanted_ids = {value.strip().lower() for value in document_ids if value.strip()}
    documents = (await client.beta.libraries.documents.list_async(library_id=library.id)).data
    pdfs = []
    for document in documents:
        if wanted_ids and not _document_matches(document, wanted_ids):
            continue
        if (document.extension or "").lower() != "pdf":
            continue
        pdfs.append({"id": document.id, "name": document.name})
    if wanted_ids and not pdfs:
        available = "\n".join(f"  - {doc.id} : {doc.name}" for doc in documents) or "  (aucun)"
        raise ValueError(
            f"Aucun document trouvé parmi {sorted(wanted_ids)} dans '{library.name}'. "
            f"Documents disponibles (id : nom) :\n{available}"
        )
    return {"library_id": library.id, "library_name": library.name, "pdfs": pdfs}


def _document_matches(document: Any, wanted: set[str]) -> bool:
    """Compare un document aux valeurs demandees : id exact, nom exact, ou nom sans .pdf."""
    targets = {str(document.id).strip().lower()}
    name = str(document.name or "").strip().lower()
    targets.add(name)
    targets.add(name.removesuffix(".pdf"))
    return bool(wanted & targets)


@workflows.activity(name="library-extract-invoice-data")
async def extract_invoice_data(
    library_id: str,
    document_id: str,
    document_name: str,
) -> dict[str, Any]:
    client = get_mistral_client(run_as=RunAs.AUTO)
    raw_text = await client.beta.libraries.documents.text_content_async(
        library_id=library_id, document_id=document_id
    )
    if isinstance(raw_text, str):
        text = raw_text
    elif isinstance(raw_text, dict):
        text = raw_text.get("text") or raw_text.get("content") or json.dumps(raw_text)
    elif hasattr(raw_text, "text"):
        text = raw_text.text
    else:
        text = str(raw_text)
    if not text or not str(text).strip():
        return {"status": "failed", "detail": "Document illisible ou vide", "invoice": None}

    response = await client.chat.complete_async(
        model=EXTRACTION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu extrais les données d'une facture fournisseur à partir de son texte "
                    "(sortie OCR). Réponds uniquement avec un objet JSON contenant exactement "
                    "les clés : is_invoice (booléen, false si le document n'est pas une facture), "
                    "invoice_date (la date de la facture, généralement sur la ligne « Date "
                    "d'émission » ou « Facture n°… du … » ; recopie-la exactement comme elle est "
                    "écrite, par exemple « 01/06/2023 » ou « 01 juin 2023 » ; jamais null si une "
                    "date apparaît dans le document), supplier_ref (numéro de la facture chez le "
                    "fournisseur), lines (liste d'objets {description, quantity, unit_price_ht, "
                    "unit_price_ttc, line_total_ht, vat_rate}), total_ht, total_ttc, currency. "
                    "Réponds uniquement à partir du document. Règles sur les montants : les "
                    "factures françaises écrivent les montants avec une virgule décimale « 10,33 » "
                    ": convertis-les en nombres à point (10.33), sans séparateur de milliers ; "
                    "unit_price_ht est le prix unitaire hors taxes de la ligne, line_total_ht est "
                    "le total HT de la ligne quand il figure sur la facture, total_ht est le "
                    "total HT de toute la facture : ne confonds jamais ces colonnes et ne recopie "
                    "jamais un sous-total ou un total de facture dans une ligne ; les taux de TVA "
                    "sont en pourcentage (20 pour 20 %)."
                ),
            },
            {"role": "user", "content": f"Document : {document_name}\n\n{str(text)[:60000]}"},
        ],
        response_format={"type": "json_object"},
    )
    message = response.choices[0].message
    if isinstance(message.content, str):
        content = message.content
    else:
        content = "".join(
            chunk.text for chunk in (message.content or []) if hasattr(chunk, "text")
        )
    parse_error = None
    invoice = None
    try:
        invoice = ExtractedInvoice.model_validate(_loads_json_object(content))
    except Exception as error:
        parse_error = f"{type(error).__name__}: {error}"
    if invoice is not None and not invoice.is_invoice:
        return {"status": "skipped_not_invoice", "detail": None, "invoice": None}
    normalized_date = _normalize_date(invoice.invoice_date) if invoice else None
    if not normalized_date:
        normalized_date = _find_date_in_text(str(text))
    if invoice is None or not normalized_date:
        snippet = (content or "")[:300]
        return {
            "status": "failed",
            "detail": (
                f"Date de facturation introuvable (OCR : {len(str(text))} caractères, "
                f"réponse modèle : {snippet!r}, erreur : {parse_error!r})"
            ),
            "invoice": None,
        }
    invoice.invoice_date = normalized_date
    if invoice.total_ttc is None and invoice.total_ht is None and not invoice.lines:
        return {"status": "failed", "detail": "Aucun montant trouvé dans la facture", "invoice": None}
    return {
        "status": "ok",
        "detail": None,
        "invoice": invoice.model_dump(),
        "warnings": _amount_warnings(invoice),
    }


def _same_amount(existing: dict[str, Any], extracted: ExtractedInvoice) -> bool:
    """Compare une facture existante Dolibarr aux montants extraits (tolérance 0,05).

    Sans totaux comparables, considère le doublon déjà fait (comportement prudent).
    """
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    existing_ttc = _to_float(existing.get("total_ttc"))
    existing_ht = _to_float(existing.get("total_ht"))
    if extracted.total_ttc is not None and existing_ttc is not None:
        return abs(existing_ttc - extracted.total_ttc) <= 0.05
    if extracted.total_ht is not None and existing_ht is not None:
        return abs(existing_ht - extracted.total_ht) <= 0.05
    return True


@workflows.activity(name="dolibarr-create-supplier-invoice")
async def create_supplier_invoice(
    supplier_id: int,
    supplier_name: str,
    invoice: dict[str, Any],
    document_name: str,
    replace_mismatched_duplicates: bool = True,
) -> dict[str, Any]:
    dolibarr = _dolibarr_client()
    extracted = ExtractedInvoice.model_validate(invoice)
    supplier_ref = (extracted.supplier_ref or "").strip()
    duplicate_note: str | None = None

    if supplier_ref:
        existing = await dolibarr.list_supplier_invoices(
            thirdparty_ids=str(supplier_id),
            sqlfilters=f"(t.ref_supplier:=:'{supplier_ref}')",
            limit=10,
        )
        if existing:
            if any(_same_amount(candidate, extracted) for candidate in existing):
                return {
                    "status": "skipped_duplicate",
                    "invoice_id": int(existing[0]["id"]),
                    "invoice_ref": existing[0].get("ref"),
                    "detail": f"Facture fournisseur '{supplier_ref}' déjà présente dans Dolibarr",
                }
            old = existing[0]
            if not replace_mismatched_duplicates:
                return {
                    "status": "skipped_duplicate",
                    "invoice_id": int(old["id"]),
                    "invoice_ref": old.get("ref"),
                    "detail": (
                        f"Doublon avec total différent conservé : existant "
                        f"{old.get('total_ttc', old.get('total_ht', '?'))} €, extrait "
                        f"{extracted.total_ttc if extracted.total_ttc is not None else extracted.total_ht} €"
                    ),
                }
            duplicate_note = (
                f"Doublon total différent : ancienne facture {old.get('ref', old.get('id'))} "
                f"(total {old.get('total_ttc', old.get('total_ht', '?'))} €) conservée, "
                "à corriger manuellement dans Dolibarr"
            )

    payload: dict[str, Any] = {
        "ref": "auto",
        "socid": supplier_id,
        "note": f"Importé depuis {document_name} par le workflow Mistral",
    }
    invoice_timestamp = _date_to_timestamp(extracted.invoice_date)
    if invoice_timestamp is not None:
        payload["date"] = invoice_timestamp
    if supplier_ref:
        payload["ref_supplier"] = supplier_ref

    created = await dolibarr.create_supplier_invoice(payload)
    invoice_id = _created_id(created)
    if invoice_id is None:
        raise ValueError(f"Création de la facture impossible : {created!r}")
    invoice_ref = created.get("ref") if isinstance(created, dict) else None

    try:
        for line in _compute_lines(extracted):
            await dolibarr.add_supplier_invoice_line(invoice_id, line)
        await dolibarr.validate_supplier_invoice(invoice_id)
    except Exception:
        await dolibarr.delete_supplier_invoice(invoice_id)
        raise

    fetched = await dolibarr.get_supplier_invoice(invoice_id)
    if isinstance(fetched, dict) and fetched.get("ref"):
        invoice_ref = fetched["ref"]

    return {
        "status": "created",
        "invoice_id": invoice_id,
        "invoice_ref": invoice_ref,
        "detail": duplicate_note,
    }


@workflows.activity(name="library-delete-document")
async def delete_library_document(
    library_id: str,
    document_id: str,
) -> dict[str, Any]:
    client = get_mistral_client(run_as=RunAs.AUTO)
    await client.beta.libraries.documents.delete_async(
        library_id=library_id, document_id=document_id
    )
    return {"status": "deleted"}


@workflows.activity(name="dolibarr-classify-supplier-invoice-paid")
async def classify_supplier_invoice_paid(
    invoice_id: int,
    comment: str,
) -> dict[str, Any]:
    dolibarr = _dolibarr_client()
    await dolibarr.mark_supplier_invoice_paid(invoice_id, close_note=comment)
    return {"status": "paid"}


@workflows.activity(name="dolibarr-pay-supplier-invoice")
async def pay_supplier_invoice(
    invoice_id: int,
    payment_date: str,
    payment_mode_id: int,
    bank_account_id: int,
    comment: str,
) -> dict[str, Any]:
    dolibarr = _dolibarr_client()
    payment_timestamp = _date_to_timestamp(payment_date)
    payment = await dolibarr.add_supplier_invoice_payment(
        invoice_id,
        {
            "datepaye": payment_timestamp or payment_date,
            "payment_mode_id": payment_mode_id,
            "closepaidinvoices": "yes",
            "accountid": bank_account_id,
            "comment": comment,
        },
    )
    amount = None
    if isinstance(payment, dict):
        for key in ("amount", "montant", "total"):
            if payment.get(key) is not None:
                amount = float(payment[key])
                break
    return {"status": "paid", "amount_paid": amount, "detail": None}


async def _fetch_pdf(url: str) -> bytes:
    import httpx

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


@workflows.activity(name="dolibarr-attach-invoice-pdf")
async def attach_invoice_pdf(
    library_id: str,
    document_id: str,
    document_name: str,
    invoice_ref: str,
) -> dict[str, Any]:
    client = get_mistral_client(run_as=RunAs.AUTO)
    signed_url = await client.beta.libraries.documents.get_signed_url_async(
        library_id=library_id, document_id=document_id
    )
    content = await _fetch_pdf(signed_url)
    dolibarr = _dolibarr_client()
    await dolibarr.upload_supplier_invoice_document(
        ref=invoice_ref,
        filename=document_name,
        content_base64=base64.b64encode(content).decode("ascii"),
    )
    return {"status": "attached"}


@workflows.workflow.define(
    name="dolibarr-supplier-invoices-import",
    workflow_display_name="Import factures fournisseurs Dolibarr",
    workflow_description=(
        "Importe un dossier de PDF de factures fournisseurs dans Dolibarr : chaque facture "
        "est créée, validée, puis réglée par prélèvement à sa date de facturation sur le "
        "compte bancaire choisi (par défaut le compte 1)."
    ),
    on_behalf_of=True,
)
class SupplierInvoiceImportWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: SupplierInvoiceImportInput) -> ImportResult:
        supplier = await resolve_supplier(
            input.supplier_name,
            input.create_missing_supplier and not input.dry_run,
            allow_missing=input.dry_run,
        )
        payment_mode = await resolve_payment_mode(input.payment_mode_code, input.payment_mode_id)
        bank_account = await resolve_bank_account(input.bank_account_id)
        library = await list_library_pdfs(input.library_name_or_id, input.document_ids)
        library_id = library["library_id"]
        pdfs = library["pdfs"]
        if input.max_documents is not None:
            pdfs = pdfs[: input.max_documents]

        results: list[DocumentResult] = []
        for pdf in pdfs:
            results.append(
                await _process_document(
                    library_id=library_id,
                    document=pdf,
                    supplier_id=supplier["id"],
                    supplier_name=supplier["name"],
                    payment_mode_id=payment_mode["id"],
                    bank_account_id=bank_account["id"],
                    dry_run=input.dry_run,
                    replace_mismatched_duplicates=input.replace_mismatched_duplicates,
                )
            )

        return ImportResult(
            supplier=supplier["name"],
            supplier_id=supplier["id"],
            supplier_created=supplier["created"],
            bank_account_id=bank_account["id"],
            payment_mode=payment_mode["label"],
            documents=results,
        )


async def _process_document(
    library_id: str,
    document: dict[str, Any],
    supplier_id: int,
    supplier_name: str,
    payment_mode_id: int,
    bank_account_id: int,
    dry_run: bool = False,
    replace_mismatched_duplicates: bool = True,
) -> DocumentResult:
    document_id = document["id"]
    document_name = document["name"]
    created_id: int | None = None
    created_ref: str | None = None
    try:
        extracted = await extract_invoice_data(library_id, document_id, document_name)
        if extracted["status"] != "ok":
            return DocumentResult(
                document=document_name,
                status=extracted["status"],
                detail=extracted.get("detail"),
            )
        invoice = extracted["invoice"]
        warnings = list(extracted.get("warnings") or [])

        if dry_run:
            detail = json.dumps(
                {
                    "date": invoice.get("invoice_date"),
                    "lignes": invoice.get("lines", []),
                    "total_ht": invoice.get("total_ht"),
                    "total_ttc": invoice.get("total_ttc"),
                    "devise": invoice.get("currency"),
                },
                ensure_ascii=False,
            )
            if warnings:
                detail = f"{warnings} | {detail}"
            return DocumentResult(
                document=document_name,
                status="dry_run",
                invoice_ref=invoice.get("supplier_ref"),
                detail=detail,
            )

        created = await create_supplier_invoice(
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            invoice=invoice,
            document_name=document_name,
            replace_mismatched_duplicates=replace_mismatched_duplicates,
        )
        if created["status"] != "created":
            return DocumentResult(
                document=document_name,
                status=created["status"],
                invoice_id=created.get("invoice_id"),
                invoice_ref=created.get("invoice_ref"),
                detail=created.get("detail"),
            )
        created_id = created["invoice_id"]
        created_ref = created.get("invoice_ref")

        total_ttc = float(invoice.get("total_ttc") or 0.0)
        total_ht = float(invoice.get("total_ht") or 0.0)
        if total_ttc == 0.0 and total_ht == 0.0:
            await classify_supplier_invoice_paid(
                invoice_id=created_id,
                comment=f"Montant nul : classement payé sans règlement ({document_name})",
            )
            result = DocumentResult(
                document=document_name,
                status="paid",
                invoice_id=created_id,
                invoice_ref=created_ref,
                amount_paid=0.0,
                detail="Montant nul : aucun règlement requis, classée payée (settopaid)",
            )
        else:
            paid = await pay_supplier_invoice(
                invoice_id=created_id,
                payment_date=invoice["invoice_date"],
                payment_mode_id=payment_mode_id,
                bank_account_id=bank_account_id,
                comment=f"Prélèvement automatique - facture {created_ref or document_name}",
            )
            result = DocumentResult(
                document=document_name,
                status="paid",
                invoice_id=created_id,
                invoice_ref=created_ref,
                amount_paid=paid.get("amount_paid") or total_ttc or None,
            )

        duplicate_note = created.get("detail")
        if warnings:
            note = f"Avertissement extracteur : {' ; '.join(warnings)}"
            result.detail = f"{result.detail} | {note}" if result.detail else note
        if duplicate_note:
            result.detail = f"{result.detail} | {duplicate_note}" if result.detail else duplicate_note

        if created_ref:
            try:
                await attach_invoice_pdf(
                    library_id=library_id,
                    document_id=document_id,
                    document_name=document_name,
                    invoice_ref=created_ref,
                )
            except Exception as error:
                note = f"Liaison du PDF impossible : {_error_text(error)}"
                result.detail = f"{result.detail} | {note}" if result.detail else note

        if result.status == "paid":
            try:
                await delete_library_document(library_id=library_id, document_id=document_id)
            except Exception as error:
                note = f"Suppression du PDF de la bibliothèque impossible : {_error_text(error)}"
                result.detail = f"{result.detail} | {note}" if result.detail else note
        return result
    except Exception as error:
        return DocumentResult(
            document=document_name,
            status="failed",
            invoice_id=created_id,
            invoice_ref=created_ref,
            detail=_error_text(error),
        )
