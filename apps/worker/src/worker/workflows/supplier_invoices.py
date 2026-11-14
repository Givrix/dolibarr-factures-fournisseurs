"""Import de factures fournisseurs PDF dans Dolibarr, réglées par prélèvement.

Le workflow prend en entrée une bibliothèque Mistral contenant des PDF de factures
fournisseurs et un nom de fournisseur. Pour chaque PDF :
  1. extraction des données de la facture (OCR bibliothèque + modèle structuré) ;
  2. création de la facture fournisseur dans Dolibarr (avec anti-doublon) ;
  3. validation de la facture ;
  4. règlement par prélèvement à la date de facturation, sur le compte bancaire choisi.
"""

import json
import re
from typing import Any

import mistralai.workflows as workflows
import pydantic
from mistralai.workflows import Depends
from mistralai.workflows.client import get_mistral_client
from mistralai.workflows.core.auth.run_as import RunAs
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)
from mistralai.workflows.plugins.mistralai.connectors.exceptions import ConnectorToolCallError

DOLIBARR_CONNECTOR_NAME = "dolibarr_anelosimus_c428"
dolibarr_connector = connector(DOLIBARR_CONNECTOR_NAME)

TOOL_LIST_THIRDPARTIES = "listThirdparties"
TOOL_CREATE_THIRDPARTIES = "createThirdparties"
TOOL_LIST_SUPPLIER_INVOICES = "listSupplierinvoices"
TOOL_CREATE_SUPPLIER_INVOICES = "createSupplierinvoices"
TOOL_CREATE_SUPPLIER_INVOICE_LINE = "supplierinvoicesCreateLine"
TOOL_VALIDATE_SUPPLIER_INVOICE = "supplierinvoicesValidate"
TOOL_ADD_SUPPLIER_INVOICE_PAYMENT = "supplierinvoicesAddPayment"
TOOL_REMOVE_SUPPLIER_INVOICE = "removeSupplierinvoices"
TOOL_LIST_PAYMENT_TYPES = "setupRetrievePaymentTypes"
TOOL_LIST_BANK_ACCOUNTS = "listBankaccounts"

EXTRACTION_MODEL = "mistral-medium-latest"
DEFAULT_VAT_RATE = 20.0
DEFAULT_PAYMENT_MODE_CODE = "PRE"


class InvoiceLine(pydantic.BaseModel):
    description: str
    quantity: float = 1.0
    unit_price_ht: float | None = None
    unit_price_ttc: float | None = None
    vat_rate: float | None = None


class ExtractedInvoice(pydantic.BaseModel):
    is_invoice: bool = True
    invoice_date: str | None = pydantic.Field(
        default=None, description="Date de facturation au format YYYY-MM-DD"
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
    payment_mode_id: int | None = pydantic.Field(
        default=None,
        description=(
            "ID Dolibarr du mode de paiement (table llx_c_paiement). S'il est fourni, "
            "la lecture des dictionnaires est évitée (utile si l'API renvoie 403)"
        ),
    )
    dry_run: bool = pydantic.Field(
        default=False,
        description="Si vrai : extraction seule, aucune écriture dans Dolibarr",
    )


def _unwrap_tool_result(raw: Any) -> Any:
    """Normalise la réponse d'un outil connecteur en données Python."""
    if raw is None:
        return None
    if isinstance(raw, (list, int, float, bool)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    if isinstance(raw, dict):
        content = raw.get("content")
        if isinstance(content, list) and content:
            texts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
                else:
                    return raw
            if len(texts) == 1:
                return _unwrap_tool_result(texts[0])
            return [_unwrap_tool_result(t) for t in texts]
        return raw
    return raw


def _data_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("data", data)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


_EMPTY_LIST_ERROR = re.compile(r"HTTP error 404.*[Nn]o .+ found", re.DOTALL)
_FORBIDDEN_ERROR = re.compile(r"HTTP error 403", re.DOTALL)


async def _call_list(
    dolibarr: ToolCallClient,
    tool: str,
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Appel de liste Dolibarr ; l'API renvoie 404 « No ... found » quand la liste est vide."""
    try:
        raw = await dolibarr.call_tool(tool, arguments)
    except ConnectorToolCallError as error:
        text = json.dumps(error.details[0] if error.details else "", default=str)
        if _EMPTY_LIST_ERROR.search(text):
            return []
        raise
    return _data_list(_unwrap_tool_result(raw))


def _compute_lines(invoice: ExtractedInvoice) -> list[dict[str, Any]]:
    if invoice.lines:
        lines = []
        for line in invoice.lines:
            vat_rate = line.vat_rate if line.vat_rate is not None else DEFAULT_VAT_RATE
            pu_ht = line.unit_price_ht
            if pu_ht is None:
                if line.unit_price_ttc is not None:
                    pu_ht = line.unit_price_ttc / (1.0 + vat_rate / 100.0)
                else:
                    pu_ht = 0.0
            lines.append(
                {
                    "desc": line.description,
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
            "desc": f"Facture {invoice.supplier_ref or 'fournisseur'}",
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
    dolibarr: ToolCallClient = Depends(dolibarr_connector),
) -> dict[str, Any]:
    candidates = await _call_list(
        dolibarr,
        TOOL_LIST_THIRDPARTIES,
        {
            "sqlfilters": f"((t.nom:like:'{supplier_name}%') or (t.name_alias:like:'{supplier_name}%'))",
            "limit": 100,
        },
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

    raw = await dolibarr.call_tool(
        TOOL_CREATE_THIRDPARTIES,
        {"request_data": [json.dumps({"name": supplier_name, "fournisseur": 1})]},
    )
    created = _unwrap_tool_result(raw)
    if not isinstance(created, dict) or "id" not in created:
        raise ValueError(f"Création du fournisseur '{supplier_name}' impossible : {created!r}")
    return {"id": int(created["id"]), "name": created.get("name", supplier_name), "created": True}


@workflows.activity(name="dolibarr-resolve-payment-mode")
async def resolve_payment_mode(
    payment_mode_code: str,
    payment_mode_id: int | None = None,
    dolibarr: ToolCallClient = Depends(dolibarr_connector),
) -> dict[str, Any]:
    if payment_mode_id is not None:
        return {"id": payment_mode_id, "label": payment_mode_code}
    try:
        payment_types = await _call_list(dolibarr, TOOL_LIST_PAYMENT_TYPES, {"limit": 200})
    except ConnectorToolCallError as error:
        text = json.dumps(error.details[0] if error.details else "", default=str)
        if _FORBIDDEN_ERROR.search(text):
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
    raise ValueError(f"Mode de paiement '{payment_mode_code}' introuvable. Disponibles : {available}")


@workflows.activity(name="dolibarr-resolve-bank-account")
async def resolve_bank_account(
    bank_account_id: int,
    dolibarr: ToolCallClient = Depends(dolibarr_connector),
) -> dict[str, Any]:
    accounts = await _call_list(dolibarr, TOOL_LIST_BANK_ACCOUNTS, {"limit": 200})
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

    wanted_ids = set(document_ids)
    documents = (await client.beta.libraries.documents.list_async(library_id=library.id)).data
    pdfs = []
    for document in documents:
        if wanted_ids and document.id not in wanted_ids:
            continue
        if (document.extension or "").lower() != "pdf":
            continue
        pdfs.append({"id": document.id, "name": document.name})
    if wanted_ids and not pdfs:
        raise ValueError(f"Aucun PDF trouvé parmi les documents demandés dans '{library.name}'")
    return {"library_id": library.id, "library_name": library.name, "pdfs": pdfs}


@workflows.activity(name="library-extract-invoice-data")
async def extract_invoice_data(
    library_id: str,
    document_id: str,
    document_name: str,
) -> dict[str, Any]:
    client = get_mistral_client(run_as=RunAs.AUTO)
    text = await client.beta.libraries.documents.text_content_async(
        library_id=library_id, document_id=document_id
    )
    if isinstance(text, dict):
        text = text.get("content") or text.get("text") or json.dumps(text)
    if not text or not str(text).strip():
        return {"status": "failed", "detail": "Document illisible ou vide", "invoice": None}

    response = await client.chat.parse_async(
        response_format=ExtractedInvoice,
        model=EXTRACTION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu extrais les données d'une facture fournisseur à partir de son texte "
                    "(sortie OCR). Réponds uniquement à partir du document. Formats : dates en "
                    "YYYY-MM-DD, montants en nombres décimaux, prix unitaires hors taxes quand "
                    "ils sont disponibles, taux de TVA en pourcentage (20 pour 20 %). "
                    "is_invoice=false si le document n'est pas une facture."
                ),
            },
            {"role": "user", "content": f"Document : {document_name}\n\n{str(text)[:60000]}"},
        ],
    )
    invoice = response.choices[0].message.parsed
    if invoice is None or not invoice.is_invoice:
        return {"status": "skipped_not_invoice", "detail": None, "invoice": None}
    if not invoice.invoice_date:
        return {"status": "failed", "detail": "Date de facturation introuvable", "invoice": None}
    if invoice.total_ttc is None and invoice.total_ht is None and not invoice.lines:
        return {"status": "failed", "detail": "Aucun montant trouvé dans la facture", "invoice": None}
    return {"status": "ok", "detail": None, "invoice": invoice.model_dump()}


@workflows.activity(name="dolibarr-create-supplier-invoice")
async def create_supplier_invoice(
    supplier_id: int,
    supplier_name: str,
    invoice: dict[str, Any],
    document_name: str,
    dolibarr: ToolCallClient = Depends(dolibarr_connector),
) -> dict[str, Any]:
    extracted = ExtractedInvoice.model_validate(invoice)
    supplier_ref = (extracted.supplier_ref or "").strip()

    if supplier_ref:
        existing = await _call_list(
            dolibarr,
            TOOL_LIST_SUPPLIER_INVOICES,
            {
                "thirdparty_ids": str(supplier_id),
                "sqlfilters": f"(t.ref_supplier:=:'{supplier_ref}')",
                "limit": 10,
            },
        )
        if existing:
            return {
                "status": "skipped_duplicate",
                "invoice_id": int(existing[0]["id"]),
                "invoice_ref": existing[0].get("ref"),
                "detail": f"Facture fournisseur '{supplier_ref}' déjà présente dans Dolibarr",
            }

    payload: dict[str, Any] = {
        "ref": "auto",
        "socid": supplier_id,
        "date": extracted.invoice_date,
        "note": f"Importé depuis {document_name} par le workflow Mistral",
    }
    if supplier_ref:
        payload["ref_supplier"] = supplier_ref

    raw = await dolibarr.call_tool(
        TOOL_CREATE_SUPPLIER_INVOICES,
        {"request_data": [json.dumps(payload)]},
    )
    created = _unwrap_tool_result(raw)
    if not isinstance(created, dict) or "id" not in created:
        raise ValueError(f"Création de la facture impossible : {created!r}")
    invoice_id = int(created["id"])

    try:
        for line in _compute_lines(extracted):
            await dolibarr.call_tool(
                TOOL_CREATE_SUPPLIER_INVOICE_LINE,
                {"id": invoice_id, "request_data": [json.dumps(line)]},
            )
        await dolibarr.call_tool(TOOL_VALIDATE_SUPPLIER_INVOICE, {"id": invoice_id})
    except Exception:
        await dolibarr.call_tool(TOOL_REMOVE_SUPPLIER_INVOICE, {"id": invoice_id})
        raise

    return {
        "status": "created",
        "invoice_id": invoice_id,
        "invoice_ref": created.get("ref"),
        "detail": None,
    }


@workflows.activity(name="dolibarr-pay-supplier-invoice")
async def pay_supplier_invoice(
    invoice_id: int,
    payment_date: str,
    payment_mode_id: int,
    bank_account_id: int,
    comment: str,
    dolibarr: ToolCallClient = Depends(dolibarr_connector),
) -> dict[str, Any]:
    raw = await dolibarr.call_tool(
        TOOL_ADD_SUPPLIER_INVOICE_PAYMENT,
        {
            "id": invoice_id,
            "datepaye": payment_date,
            "payment_mode_id": payment_mode_id,
            "closepaidinvoices": "yes",
            "accountid": bank_account_id,
            "comment": comment,
        },
    )
    payment = _unwrap_tool_result(raw)
    amount = None
    if isinstance(payment, dict):
        for key in ("amount", "montant", "total"):
            if payment.get(key) is not None:
                amount = float(payment[key])
                break
    return {"status": "paid", "amount_paid": amount, "detail": None}


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
@uses_connectors(dolibarr_connector)
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
) -> DocumentResult:
    document_id = document["id"]
    document_name = document["name"]
    try:
        extracted = await extract_invoice_data(library_id, document_id, document_name)
        if extracted["status"] != "ok":
            return DocumentResult(
                document=document_name,
                status=extracted["status"],
                detail=extracted.get("detail"),
            )
        invoice = extracted["invoice"]

        if dry_run:
            return DocumentResult(
                document=document_name,
                status="dry_run",
                invoice_ref=invoice.get("supplier_ref"),
                detail=json.dumps(
                    {
                        "date": invoice.get("invoice_date"),
                        "lignes": invoice.get("lines", []),
                        "total_ht": invoice.get("total_ht"),
                        "total_ttc": invoice.get("total_ttc"),
                        "devise": invoice.get("currency"),
                    },
                    ensure_ascii=False,
                ),
            )

        created = await create_supplier_invoice(
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            invoice=invoice,
            document_name=document_name,
        )
        if created["status"] != "created":
            return DocumentResult(
                document=document_name,
                status=created["status"],
                invoice_id=created.get("invoice_id"),
                invoice_ref=created.get("invoice_ref"),
                detail=created.get("detail"),
            )

        paid = await pay_supplier_invoice(
            invoice_id=created["invoice_id"],
            payment_date=invoice["invoice_date"],
            payment_mode_id=payment_mode_id,
            bank_account_id=bank_account_id,
            comment=f"Prélèvement automatique - facture {created.get('invoice_ref') or document_name}",
        )
        return DocumentResult(
            document=document_name,
            status="paid",
            invoice_id=created["invoice_id"],
            invoice_ref=created.get("invoice_ref"),
            amount_paid=paid.get("amount_paid"),
        )
    except Exception as error:
        return DocumentResult(document=document_name, status="failed", detail=str(error))
