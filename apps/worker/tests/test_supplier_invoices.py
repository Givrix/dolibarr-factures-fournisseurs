import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment

from mistralai.workflows import testing as workflows_testing

import worker.workflows.supplier_invoices as si
from worker.dolibarr import DolibarrClient, DolibarrError
from worker.workflows.supplier_invoices import (
    SupplierInvoiceImportWorkflow,
    create_supplier_invoice,
    extract_invoice_data,
    list_library_pdfs,
    pay_supplier_invoice,
    resolve_bank_account,
    resolve_payment_mode,
    resolve_supplier,
)

REAL_ACTIVITIES = [
    resolve_supplier,
    resolve_payment_mode,
    resolve_bank_account,
    list_library_pdfs,
    extract_invoice_data,
    create_supplier_invoice,
    pay_supplier_invoice,
    si.classify_supplier_invoice_paid,
    si.attach_invoice_pdf,
    si.delete_library_document,
]

TASK_QUEUE = "test-task-queue"



class FakeDolibarr:
    """Imite worker.dolibarr.DolibarrClient avec une API REST qui répond comme Dolibarr."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.next_invoice_id = 101
        self.existing_invoices: list[dict[str, Any]] = []
        self.payments_by_invoice: dict[int, list[dict[str, Any]]] = {}

    def _record(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return arguments

    async def list_thirdparties(self, sqlfilters: str = "", limit: int = 100) -> list[dict[str, Any]]:
        self._record("listThirdparties", {"sqlfilters": sqlfilters, "limit": limit})
        return []

    async def create_thirdparty(self, payload: dict[str, Any]) -> Any:
        self._record("createThirdparties", payload)
        return 7

    async def list_payment_types(self, limit: int = 200) -> list[dict[str, Any]]:
        self._record("setupRetrievePaymentTypes", {"limit": limit})
        return [
            {"id": 2, "code": "VIR", "label": "Virement"},
            {"id": 3, "code": "PRE", "label": "Prélèvement"},
        ]

    async def list_bank_accounts(self, limit: int = 200) -> list[dict[str, Any]]:
        self._record("listBankaccounts", {"limit": limit})
        return [{"id": 1, "label": "Compte courant"}]

    async def list_supplier_invoices(
        self, thirdparty_ids: str = "", sqlfilters: str = "", limit: int = 10
    ) -> list[dict[str, Any]]:
        self._record(
            "listSupplierinvoices",
            {"thirdparty_ids": thirdparty_ids, "sqlfilters": sqlfilters, "limit": limit},
        )
        return self.existing_invoices

    async def create_supplier_invoice(self, payload: dict[str, Any]) -> Any:
        self._record("createSupplierinvoices", payload)
        invoice = {"id": self.next_invoice_id, "ref": f"FF{self.next_invoice_id}"}
        self.next_invoice_id += 1
        return invoice

    async def get_supplier_invoice(self, invoice_id: int) -> dict[str, Any]:
        self._record("getSupplierinvoices", {"id": invoice_id})
        return {"id": invoice_id, "ref": f"FF{invoice_id}"}

    async def upload_supplier_invoice_document(self, ref: str, filename: str, content_base64: str) -> Any:
        self._record("uploadSupplierInvoiceDocument", {"ref": ref, "filename": filename})
        return {"success": True}

    async def add_supplier_invoice_line(self, invoice_id: int, line: dict[str, Any]) -> Any:
        self._record("supplierinvoicesCreateLine", {"id": invoice_id, **line})
        return 1

    async def validate_supplier_invoice(self, invoice_id: int) -> Any:
        self._record("supplierinvoicesValidate", {"id": invoice_id})
        return {"success": {"code": 200, "message": "Invoice validated"}}

    async def add_supplier_invoice_payment(self, invoice_id: int, payload: dict[str, Any]) -> Any:
        self._record("supplierinvoicesAddPayment", {"id": invoice_id, **payload})
        return {"id": 1, "amount": 120.0}

    async def list_supplier_invoice_payments(self, invoice_id: int) -> list[dict[str, Any]]:
        self._record("supplierinvoicesGetPayments", {"id": invoice_id})
        return self.payments_by_invoice.get(invoice_id, [])

    async def reopen_supplier_invoice(self, invoice_id: int) -> Any:
        self._record("supplierinvoicesSettounpaid", {"id": invoice_id})
        return {"success": {"code": 200, "message": "Invoice reopened"}}

    async def set_supplier_invoice_draft(self, invoice_id: int) -> Any:
        self._record("supplierinvoicesSettodraft", {"id": invoice_id})
        return {"id": invoice_id, "status": "0"}

    async def mark_supplier_invoice_paid(self, invoice_id: int, close_note: str = "") -> Any:
        self._record("supplierinvoicesSettopaid", {"id": invoice_id, "close_note": close_note})
        return {"success": {"code": 200, "message": "Invoice classified as paid"}}

    async def delete_supplier_invoice(self, invoice_id: int) -> Any:
        self._record("removeSupplierinvoices", {"id": invoice_id})
        return {"success": {"code": 200, "message": "Supplier invoice deleted"}}

    def calls_to(self, suffix: str) -> list[dict[str, Any]]:
        return [args for name, args in self.calls if name.endswith(suffix)]


class FakeLibrary:
    id = "lib-1"
    name = "Factures"


class FakeDocument:
    def __init__(self, pdf: dict[str, Any]) -> None:
        self.id = pdf["id"]
        self.name = pdf["name"]
        self.extension = "pdf"


class FakeDocuments:
    def __init__(self, pdfs: list[dict[str, Any]], text_by_document: dict[str, str]) -> None:
        self.pdfs = pdfs
        self.text_by_document = text_by_document
        self.deleted: list[str] = []

    async def list_async(self, library_id: str):
        class _Resp:
            data = [FakeDocument(pdf) for pdf in self.pdfs]

        return _Resp()

    async def text_content_async(self, library_id: str, document_id: str):
        return self.text_by_document[document_id]

    async def get_signed_url_async(self, library_id: str, document_id: str):
        return f"https://signed.example/{document_id}.pdf"

    async def delete_async(self, library_id: str, document_id: str):
        self.deleted.append(document_id)


class FakeBeta:
    def __init__(self, documents: FakeDocuments) -> None:
        self.libraries = FakeLibrariesContainer(documents)


class FakeLibrariesContainer:
    def __init__(self, documents: FakeDocuments) -> None:
        self.documents = documents

    async def list_async(self):
        class _Resp:
            data = [FakeLibrary()]

        return _Resp()


class FakeChat:
    """Imite client.chat.complete_async : renvoie le JSON d'extraction dans message.content."""

    def __init__(self, extraction: dict[str, Any]) -> None:
        self.extraction = extraction
        self.content_override: str | None = None

    async def complete_async(self, **kwargs):
        content = self.content_override or json.dumps(self.extraction, ensure_ascii=False)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeMistralClient:
    def __init__(self, pdfs: list[dict[str, Any]], text_by_document: dict[str, str], extraction: dict[str, Any]) -> None:
        self.beta = FakeBeta(FakeDocuments(pdfs, text_by_document))
        self.chat = FakeChat(extraction)


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    dolibarr: FakeDolibarr,
    client: FakeMistralClient,
) -> None:
    monkeypatch.setattr(si, "_dolibarr_client", lambda: dolibarr)
    monkeypatch.setattr(si, "get_mistral_client", lambda **kwargs: client)

    async def fake_fetch_pdf(url: str) -> bytes:
        return b"%PDF-fake"

    monkeypatch.setattr(si, "_fetch_pdf", fake_fetch_pdf)


def base_fixtures() -> tuple[FakeDolibarr, FakeMistralClient]:
    dolibarr = FakeDolibarr()
    pdfs = [{"id": "doc-1", "name": "facture-a.pdf"}, {"id": "doc-2", "name": "facture-b.pdf"}]
    text_by_document = {"doc-1": "FACTURE", "doc-2": "FACTURE"}
    extraction = {
        "is_invoice": True,
        "invoice_date": "2025-06-15",
        "supplier_ref": "FA-2025-042",
        "lines": [
            {"description": "Prestation", "quantity": 2.0, "unit_price_ht": 50.0, "vat_rate": 20.0}
        ],
        "total_ht": 100.0,
        "total_ttc": 120.0,
        "currency": "EUR",
    }
    return dolibarr, FakeMistralClient(pdfs, text_by_document, extraction)


async def run_workflow(monkeypatch: pytest.MonkeyPatch, input_data: dict[str, Any]) -> Any:
    env = await WorkflowEnvironment.start_time_skipping()
    async with env, workflows_testing.create_test_worker(
        env,
        workflows=[SupplierInvoiceImportWorkflow],
        activities=REAL_ACTIVITIES,
    ):
        return await env.client.execute_workflow(
            SupplierInvoiceImportWorkflow.run,
            input_data,
            id="test-import-supplier-invoices",
            task_queue=TASK_QUEUE,
        )


async def test_imports_all_pdfs_and_pays_them(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert result["supplier"] == "ACME"
    assert result["supplier_id"] == 7
    assert result["supplier_created"] is True
    assert result["bank_account_id"] == 1
    assert result["payment_mode"] == "Prélèvement"
    assert [doc["status"] for doc in result["documents"]] == ["paid", "paid"]
    assert [doc["invoice_id"] for doc in result["documents"]] == [101, 102]
    assert [doc["invoice_ref"] for doc in result["documents"]] == ["FF101", "FF102"]
    assert [doc["amount_paid"] for doc in result["documents"]] == [120.0, 120.0]

    payments = dolibarr.calls_to("supplierinvoicesAddPayment")
    assert len(payments) == 2
    from datetime import datetime, timezone

    assert payments[0]["datepaye"] == int(
        datetime(2025, 6, 15, tzinfo=timezone.utc).timestamp()
    )
    assert payments[0]["payment_mode_id"] == 3
    assert payments[0]["accountid"] == 1
    assert payments[0]["closepaidinvoices"] == "yes"
    assert payments[0]["comment"].startswith("Prélèvement automatique")

    lines = dolibarr.calls_to("supplierinvoicesCreateLine")
    assert len(lines) == 2
    assert lines[0] == {
        "id": 101,
        "description": "Prestation",
        "qty": 2.0,
        "pu_ht": 50.0,
        "tva_tx": 20.0,
    }
    assert dolibarr.calls_to("supplierinvoicesValidate")

    uploads = dolibarr.calls_to("uploadSupplierInvoiceDocument")
    assert len(uploads) == 2
    assert uploads[0] == {"ref": "FF101", "filename": "facture-a.pdf"}

    assert client.beta.libraries.documents.deleted == ["doc-1", "doc-2"]


async def test_existing_supplier_is_reused(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()

    async def fake_list_thirdparties(self, sqlfilters: str = "", limit: int = 100):
        dolibarr._record("listThirdparties", {"sqlfilters": sqlfilters, "limit": limit})
        return [{"id": 12, "name": "acme"}]

    monkeypatch.setattr(FakeDolibarr, "list_thirdparties", fake_list_thirdparties)
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert result["supplier_id"] == 12
    assert result["supplier_created"] is False
    assert not dolibarr.calls_to("createThirdparties")


async def test_dolibarr_404_empty_list_creates_supplier(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/thirdparties") and request.method == "GET":
            return httpx.Response(404, json={"error": {"code": 404, "message": "Not Found: No third parties found"}})
        if request.url.path.endswith("/thirdparties") and request.method == "POST":
            return httpx.Response(200, json=7)
        if request.url.path.endswith("/setup/dictionary/payment_types"):
            return httpx.Response(200, json=[{"id": 3, "code": "PRE", "label": "Pr\u00e9l\u00e8vement"}])
        if request.url.path.endswith("/bankaccounts"):
            return httpx.Response(200, json=[{"id": 1, "label": "Compte courant"}])
        raise AssertionError(f"Appel inattendu : {request.method} {request.url.path}")

    real_client = DolibarrClient(
        base_url="https://dolibarr.example/htdocs/api/index.php",
        api_token="token",
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(si, "_dolibarr_client", lambda: real_client)
    monkeypatch.setattr(si, "get_mistral_client", lambda **kwargs: client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert result["supplier_created"] is True
    assert result["supplier_id"] == 7


async def test_dry_run_missing_supplier_does_not_fail(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {"library_name_or_id": "Factures", "supplier_name": "ACME", "dry_run": True},
    )

    assert result["supplier_id"] is None
    assert result["supplier_created"] is False
    assert not dolibarr.calls_to("createThirdparties")
    assert [doc["status"] for doc in result["documents"]] == ["dry_run", "dry_run"]


async def test_payment_mode_id_skips_dictionary_lookup(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {
            "library_name_or_id": "Factures",
            "supplier_name": "ACME",
            "payment_mode_id": 3,
        },
    )

    assert not dolibarr.calls_to("setupRetrievePaymentTypes")
    payments = dolibarr.calls_to("supplierinvoicesAddPayment")
    assert payments[0]["payment_mode_id"] == 3
    assert result["payment_mode"] == "PRE"


async def test_payment_mode_403_gives_clear_error(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()

    async def fake_list_payment_types(self, limit: int = 200):
        dolibarr._record("setupRetrievePaymentTypes", {"limit": limit})
        raise DolibarrError(403, "{'error': {'code': 403, 'message': 'Forbidden'}}")

    monkeypatch.setattr(FakeDolibarr, "list_payment_types", fake_list_payment_types)
    install_fakes(monkeypatch, dolibarr, client)

    with pytest.raises(WorkflowFailureError) as exc_info:
        await run_workflow(
            monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
        )

    messages = []
    error = exc_info.value
    while error is not None:
        messages.append(str(error))
        error = error.__cause__
    assert any("llx_c_paiement" in message for message in messages)
    assert not dolibarr.calls_to("createSupplierinvoices")


async def test_supplier_missing_fails_workflow(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    with pytest.raises(WorkflowFailureError):
        await run_workflow(
            monkeypatch,
            {
                "library_name_or_id": "Factures",
                "supplier_name": "INCONNU",
                "create_missing_supplier": False,
            },
        )
    assert not dolibarr.calls_to("createSupplierinvoices")


async def test_duplicate_invoice_is_skipped(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    dolibarr.existing_invoices = [{"id": 55, "ref": "FF2504-0042", "ref_supplier": "FA-2025-042"}]
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert [doc["status"] for doc in result["documents"]] == ["skipped_duplicate", "skipped_duplicate"]
    assert [doc["invoice_id"] for doc in result["documents"]] == [55, 55]
    assert not dolibarr.calls_to("supplierinvoicesAddPayment")


async def test_duplicate_with_different_total_is_replaced(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    dolibarr.existing_invoices = [
        {"id": 55, "ref": "FF2504-0042", "ref_supplier": "FA-2025-042", "total_ttc": 44.78}
    ]
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    doc = result["documents"][0]
    assert doc["status"] == "paid"
    assert "44.78" in (doc["detail"] or "")
    assert "FF2504-0042" in (doc["detail"] or "")
    assert "supprimée" in (doc["detail"] or "")

    names = [name for name, _ in dolibarr.calls]
    old_position = next(i for i, name in enumerate(names) if name == "removeSupplierinvoices" and dolibarr.calls[i][1]["id"] == 55)
    create_position = next(i for i, name in enumerate(names) if name == "createSupplierinvoices")
    assert old_position < create_position
    assert dolibarr.calls_to("supplierinvoicesSettounpaid")
    assert dolibarr.calls_to("supplierinvoicesSettodraft")


def test_same_amount_variants():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _same_amount

    extracted = ExtractedInvoice(total_ttc=120.0)
    assert _same_amount({"total_ttc": "120.00"}, extracted)
    assert not _same_amount({"total_ttc": 44.78}, extracted)
    assert _same_amount({"total_ht": 100.0}, ExtractedInvoice(total_ttc=None, total_ht=100.0))
    assert _same_amount({}, extracted)  # pas de total comparable : considéré identique


async def test_duplicate_with_matching_total_is_skipped(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    dolibarr.existing_invoices = [
        {"id": 55, "ref": "FF2504-0042", "ref_supplier": "FA-2025-042", "total_ttc": 120.0}
    ]
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert [doc["status"] for doc in result["documents"][:1]] == ["skipped_duplicate"]
    assert not dolibarr.calls_to("createSupplierinvoices")
    assert client.beta.libraries.documents.deleted[:1] == ["doc-1"]


async def test_duplicate_mismatch_kept_keeps_pdf(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    dolibarr.existing_invoices = [
        {"id": 55, "ref": "FF2504-0042", "ref_supplier": "FA-2025-042", "total_ttc": 44.78}
    ]
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {
            "library_name_or_id": "Factures",
            "supplier_name": "ACME",
            "replace_mismatched_duplicates": False,
        },
    )

async def test_duplicate_with_different_total_can_be_kept(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    dolibarr.existing_invoices = [
        {"id": 55, "ref": "FF2504-0042", "ref_supplier": "FA-2025-042", "total_ttc": 44.78}
    ]
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {
            "library_name_or_id": "Factures",
            "supplier_name": "ACME",
            "replace_mismatched_duplicates": False,
        },
    )

    assert [doc["status"] for doc in result["documents"][:1]] == ["skipped_duplicate"]
    assert "Doublon avec total différent" in (result["documents"][0]["detail"] or "")
    assert not dolibarr.calls_to("createSupplierinvoices")
    assert client.beta.libraries.documents.deleted[:1] == []



async def test_duplicate_with_payment_cannot_be_removed_autoratically(
    monkeypatch: pytest.MonkeyPatch,
):
    dolibarr, client = base_fixtures()
    dolibarr.existing_invoices = [
        {"id": 55, "ref": "FF2504-0042", "ref_supplier": "FA-2025-042", "total_ttc": 120.0}
    ]
    dolibarr.payments_by_invoice = {55: [{"amount": 120.0, "ref": "PAY2506-001"}]}
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {
            "library_name_or_id": "Factures",
            "supplier_name": "ACME",
            "force_reimport": True,
        },
    )

    doc = result["documents"][0]
    assert doc["status"] == "failed"
    assert "paiement" in (doc["detail"] or "").lower()
    assert "rouverture" in (doc["detail"] or "") or "rouvrez" in (doc["detail"] or "").lower()
    assert not doc["invoice_id"]
    assert not dolibarr.calls_to("createSupplierinvoices")
    assert not dolibarr.calls_to("supplierinvoicesSettodraft")


async def test_force_reimport_replaces_even_matching_duplicate(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    dolibarr.existing_invoices = [
        {"id": 55, "ref": "FF2504-0042", "ref_supplier": "FA-2025-042", "total_ttc": 120.0}
    ]
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {
            "library_name_or_id": "Factures",
            "supplier_name": "ACME",
            "force_reimport": True,
        },
    )

    doc = result["documents"][0]
    assert doc["status"] == "paid"
    assert dolibarr.calls_to("removeSupplierinvoices")
    assert dolibarr.calls_to("createSupplierinvoices")


async def test_created_totals_mismatch_fails_before_payment(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()

    async def totals_from_lines(self, invoice_id: int) -> dict[str, Any]:
        self._record("getSupplierinvoices", {"id": invoice_id})
        return {"id": invoice_id, "ref": "FF173", "total_ht": 62.78, "total_ttc": 75.34}

    monkeypatch.setattr(FakeDolibarr, "get_supplier_invoice", totals_from_lines)
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    doc = result["documents"][0]
    assert doc["status"] == "failed"
    assert "75.34" in (doc["detail"] or "")
    assert "120" in (doc["detail"] or "")
    removed_ids = [args["id"] for args in dolibarr.calls_to("removeSupplierinvoices")]
    assert 101 in removed_ids
    assert not dolibarr.calls_to("supplierinvoicesAddPayment")


async def test_extraction_failure_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    client.chat.extraction = {"is_invoice": True, "invoice_date": None}
    client.beta.libraries.documents.text_by_document = {"doc-1": "FACTURE sans date", "doc-2": "FACTURE sans date"}
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert [doc["status"] for doc in result["documents"]] == ["failed", "failed"]
    assert "Date" in result["documents"][0]["detail"]
    assert "réponse modèle" in result["documents"][0]["detail"]
    assert not dolibarr.calls_to("createSupplierinvoices")


async def test_extract_falls_back_to_date_found_in_ocr(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    client.chat.extraction = {"is_invoice": True, "invoice_date": None, "total_ttc": 23.99}
    client.beta.libraries.documents.text_by_document = {
        "doc-1": "OVHcloud\nRéférence de la facture : FR57265489\nDate d'émission : 01/06/2023\nTotal TTC 23,99 €",
        "doc-2": "OVHcloud\nFacture n°FR57265489 du 01 juin 2023\nTotal TTC 23,99 €",
    }
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {"library_name_or_id": "Factures", "supplier_name": "ACME", "dry_run": True},
    )

    assert [doc["status"] for doc in result["documents"]] == ["dry_run", "dry_run"]
    assert json.loads(result["documents"][0]["detail"])["date"] == "2023-06-01"
    assert json.loads(result["documents"][1]["detail"])["date"] == "2023-06-01"


async def test_extract_falls_back_to_date_with_short_year_in_ocr(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    client.chat.extraction = {"is_invoice": True, "invoice_date": None, "total_ttc": 23.99}
    client.beta.libraries.documents.text_by_document = {
        "doc-1": "OVHcloud\nFacture FR57628354\nEmis le 02/06/23\nTotal TTC 23,99 EUR",
        "doc-2": "ACME\nFacture A-123\nDate : 15.6.24\nTotal TTC 10,00 EUR",
    }
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {"library_name_or_id": "Factures", "supplier_name": "ACME", "dry_run": True},
    )

    assert [doc["status"] for doc in result["documents"]] == ["dry_run", "dry_run"]
    assert json.loads(result["documents"][0]["detail"])["date"] == "2023-06-02"
    assert json.loads(result["documents"][1]["detail"])["date"] == "2024-06-15"


async def test_extract_parses_json_surrounded_by_prose(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    client.chat.content_override = (
        'Voici les données : {"is_invoice": true, "invoice_date": "15/07/2025", '
        '"total_ttc": 12.0} — fin.'
    )
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {"library_name_or_id": "Factures", "supplier_name": "ACME", "dry_run": True},
    )

    assert [doc["status"] for doc in result["documents"]] == ["dry_run", "dry_run"]
    assert json.loads(result["documents"][0]["detail"])["date"] == "2025-07-15"


async def test_extract_salvages_invoice_with_invalid_line(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    client.chat.content_override = json.dumps(
        {
            "is_invoice": True,
            "invoice_date": "17 oct. 2018",
            "supplier_ref": "FR20181116753",
            "lines": [
                {
                    "description": "FORMULE BRONZE PREMIUM 30 000 emails",
                    "quantity": 1,
                    "unit_price_ht": 14.95,
                    "line_total_ht": 14.95,
                    "vat_rate": 20,
                },
                {"description": "Ligne corrompue", "quantity": "abc", "unit_price_ht": {"x": 1}},
            ],
            "total_ht": 14.95,
            "total_ttc": 17.94,
        },
        ensure_ascii=False,
    )
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {"library_name_or_id": "Factures", "supplier_name": "ACME", "dry_run": True},
    )

    doc = result["documents"][0]
    assert doc["status"] == "dry_run"
    assert doc["invoice_ref"] == "FR20181116753"
    payload = json.loads(doc["detail"])
    assert payload["date"] == "2018-10-17"
    assert [line["description"] for line in payload["lignes"]] == [
        "FORMULE BRONZE PREMIUM 30 000 emails"
    ]


async def test_extract_harmonizes_discount_vat_rate(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    client.chat.content_override = json.dumps(
        {
            "is_invoice": True,
            "invoice_date": "2021-10-01",
            "supplier_ref": "786350",
            "lines": [
                {"description": "Bronze D1", "quantity": 1, "unit_price_ht": 7.95, "line_total_ht": 7.95, "vat_rate": 20.0},
                {"description": "20% Off Non Profit", "quantity": 1, "unit_price_ht": -1.59, "line_total_ht": -1.59, "vat_rate": 0.0},
            ],
            "total_ht": 6.36,
            "total_ttc": 7.63,
        }
    )
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {"library_name_or_id": "Factures", "supplier_name": "ACME", "dry_run": True},
    )

    doc = result["documents"][0]
    assert doc["status"] == "dry_run"
    assert "Somme des lignes TTC" not in (doc["detail"] or "")
    assert "raligne a 20.0 %" in (doc["detail"] or "")
    json_part = (doc["detail"] or "").rsplit(" | ", 1)[-1]
    assert all(line["vat_rate"] == 20.0 for line in json.loads(json_part)["lignes"])


async def test_zero_amount_invoice_is_classified_paid(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    client.chat.extraction = {
        "is_invoice": True,
        "invoice_date": "2025-06-15",
        "supplier_ref": "FA-ZERO",
        "lines": [{"description": "Avoir", "quantity": 1.0, "unit_price_ht": 0.0, "vat_rate": 20.0}],
        "total_ht": 0.0,
        "total_ttc": 0.0,
        "currency": "EUR",
    }
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert [doc["status"] for doc in result["documents"]] == ["paid", "paid"]
    assert [doc["amount_paid"] for doc in result["documents"]] == [0.0, 0.0]
    assert "Montant nul" in result["documents"][0]["detail"]
    assert not dolibarr.calls_to("supplierinvoicesAddPayment")
    settopaid = dolibarr.calls_to("supplierinvoicesSettopaid")
    assert len(settopaid) == 2
    assert settopaid[0]["id"] == 101
    assert len(dolibarr.calls_to("uploadSupplierInvoiceDocument")) == 2
    assert client.beta.libraries.documents.deleted == ["doc-1", "doc-2"]


async def test_amount_mismatch_warning_is_surfaced_on_paid_result(
    monkeypatch: pytest.MonkeyPatch,
):
    dolibarr, client = base_fixtures()
    client.chat.extraction = {
        "is_invoice": True,
        "invoice_date": "2025-06-15",
        "supplier_ref": "FA-CONFUSION",
        "lines": [
            {"description": "Article A", "quantity": 1.0, "unit_price_ht": 10.33, "vat_rate": 20.0},
            {"description": "Article B", "quantity": 1.0, "unit_price_ht": 18.0, "vat_rate": 20.0},
        ],
        "total_ht": 44.78,
        "total_ttc": 53.74,
        "currency": "EUR",
    }
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert [doc["status"] for doc in result["documents"]] == ["paid", "paid"]
    assert (
        "Avertissement extracteur" in result["documents"][0]["detail"]
        or "Avertissement extracteur" in result["documents"][1]["detail"]
    )
    assert "44.78" in result["documents"][0]["detail"] or "44.78" in result["documents"][1]["detail"]


async def test_pdf_attachment_failure_does_not_fail_document(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    async def failing_upload(self, ref: str, filename: str, content_base64: str) -> Any:
        dolibarr._record("uploadSupplierInvoiceDocument", {"ref": ref, "filename": filename})
        raise DolibarrError(500, "upload error")

    monkeypatch.setattr(FakeDolibarr, "upload_supplier_invoice_document", failing_upload)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert [doc["status"] for doc in result["documents"]] == ["paid", "paid"]
    assert "Liaison du PDF impossible" in result["documents"][0]["detail"]


async def test_document_selection_accepts_filename_and_stem(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    async def run(document_ids: list[str]) -> Any:
        return await run_workflow(
            monkeypatch,
            {
                "library_name_or_id": "Factures",
                "supplier_name": "ACME",
                "dry_run": True,
                "document_ids": document_ids,
            },
        )

    by_name = await run(["facture-a.pdf"])
    assert [doc["document"] for doc in by_name["documents"]] == ["facture-a.pdf"]

    by_stem = await run(["facture-b"])
    assert [doc["document"] for doc in by_stem["documents"]] == ["facture-b.pdf"]


async def test_unknown_document_id_lists_available_documents(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    env = await WorkflowEnvironment.start_time_skipping()
    async with env, workflows_testing.create_test_worker(
        env,
        workflows=[SupplierInvoiceImportWorkflow],
        activities=REAL_ACTIVITIES,
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await env.client.execute_workflow(
                SupplierInvoiceImportWorkflow.run,
                {
                    "library_name_or_id": "Factures",
                    "supplier_name": "ACME",
                    "document_ids": ["inv_inconnu"],
                },
                id="test-import-unknown-document",
                task_queue=TASK_QUEUE,
            )
    cause: BaseException | None = exc_info.value
    message = str(exc_info.value)
    while cause is not None and "inv_inconnu" not in message:
        cause = cause.__cause__
        message = str(cause) if cause is not None else ""
    assert "inv_inconnu" in message
    assert "doc-1" in message and "facture-a.pdf" in message


async def test_max_documents_limits_processing(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {
            "library_name_or_id": "Factures",
            "supplier_name": "ACME",
            "max_documents": 1,
            "dry_run": True,
        },
    )

    assert [doc["document"] for doc in result["documents"]] == ["facture-a.pdf"]


async def test_client_sends_payload_as_body():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["apikey"] = request.headers.get("DOLAPIKEY")
        return httpx.Response(200, json=101)

    client = DolibarrClient(
        base_url="https://dolibarr.example/htdocs/api/index.php",
        api_token="token",
        transport=httpx.MockTransport(handler),
    )
    result = await client.create_supplier_invoice({"socid": 7, "ref": "auto", "date": 1750000000})

    assert result == 101
    assert seen["path"].endswith("/supplierinvoices")
    assert seen["apikey"] == "token"
    assert seen["body"]["socid"] == 7


def test_compute_lines_from_lines():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _compute_lines

    invoice = ExtractedInvoice(
        lines=[{"description": "Article", "quantity": 2.0, "unit_price_ht": 50.0, "vat_rate": 20.0}]
    )
    assert _compute_lines(invoice) == [
        {"description": "Article", "qty": 2.0, "pu_ht": 50.0, "tva_tx": 20.0}
    ]


def test_compute_lines_from_totals_only():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _compute_lines

    invoice = ExtractedInvoice(total_ttc=120.0)
    lines = _compute_lines(invoice)
    assert len(lines) == 1
    assert lines[0]["tva_tx"] == 20.0
    assert lines[0]["pu_ht"] == pytest.approx(100.0, rel=1e-3)


def test_compute_lines_uses_line_total_when_unit_price_missing():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _compute_lines

    invoice = ExtractedInvoice(
        lines=[
            {
                "description": "Article",
                "quantity": 4.0,
                "line_total_ht": 41.32,
                "vat_rate": 20.0,
            }
        ]
    )
    lines = _compute_lines(invoice)
    assert lines[0]["pu_ht"] == pytest.approx(10.33, rel=1e-3)


def test_amount_warnings_detect_lines_total_mismatch():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _amount_warnings

    invoice = ExtractedInvoice(
        lines=[
            {"description": "A", "quantity": 1.0, "unit_price_ht": 10.33},
            {"description": "B", "quantity": 1.0, "unit_price_ht": 18.0},
        ],
        total_ht=44.78,
    )
    warnings = _amount_warnings(invoice)
    assert len(warnings) == 1
    assert "28.33" in warnings[0]
    assert "44.78" in warnings[0]


def test_amount_warnings_detect_ttc_below_ht():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _amount_warnings

    invoice = ExtractedInvoice(total_ht=100.0, total_ttc=90.0)
    warnings = _amount_warnings(invoice)
    assert len(warnings) == 1
    assert "TTC" in warnings[0]


def test_amount_warnings_detect_lines_ttc_mismatch_without_ht():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _amount_warnings

    invoice = ExtractedInvoice(
        lines=[
            {"description": "A", "quantity": 1.0, "unit_price_ht": 44.78, "vat_rate": 20.0},
            {"description": "B", "quantity": 1.0, "unit_price_ht": 18.0, "vat_rate": 20.0},
        ],
        total_ttc=29.68,
    )
    warnings = _amount_warnings(invoice)
    assert len(warnings) == 1
    assert "75.34" in warnings[0]
    assert "29.68" in warnings[0]


def test_compute_lines_prefers_displayed_line_total_over_inconsistent_unit_price():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _compute_lines

    invoice = ExtractedInvoice(
        lines=[
            {
                "description": "Messages P1",
                "quantity": 19470.0,
                "unit_price_ht": 0.00231,
                "line_total_ht": 10.33,
                "vat_rate": 20.0,
            }
        ]
    )
    lines = _compute_lines(invoice)
    assert lines[0]["pu_ht"] == pytest.approx(10.33 / 19470, rel=1e-4)


def test_discount_line_balances_totals():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _amount_warnings, _compute_lines

    invoice = ExtractedInvoice(
        lines=[
            {"description": "Messages P1", "quantity": 1.0, "line_total_ht": 10.33, "vat_rate": 20.0},
            {"description": "Essential D1", "quantity": 1.0, "line_total_ht": 18.0, "vat_rate": 20.0},
            {"description": "Crédit appliqué", "quantity": 1.0, "line_total_ht": -3.6, "vat_rate": 20.0},
        ],
        total_ttc=29.68,
    )
    lines = _compute_lines(invoice)
    assert lines[2]["pu_ht"] == pytest.approx(-3.6)
    assert _amount_warnings(invoice) == []


def test_created_totals_ht_subtotal_does_not_block_when_ttc_matches():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _check_created_totals

    extracted = ExtractedInvoice(
        lines=[
            {"description": "A", "quantity": 1.0, "line_total_ht": 10.33, "vat_rate": 20.0},
            {"description": "B", "quantity": 1.0, "line_total_ht": 18.0, "vat_rate": 20.0},
            {"description": "Crédit", "quantity": 1.0, "line_total_ht": -3.6, "vat_rate": 20.0},
        ],
        total_ht=28.33,
        total_ttc=29.68,
    )
    fetched = {"total_ht": 24.73, "total_ttc": 29.68}
    _check_created_totals(fetched, extracted)  # ne doit pas lever


def test_amount_warnings_empty_when_consistent():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _amount_warnings

    invoice = ExtractedInvoice(
        lines=[{"description": "A", "quantity": 2.0, "unit_price_ht": 50.0}],
        total_ht=100.0,
        total_ttc=120.0,
    )
    assert _amount_warnings(invoice) == []


def test_normalize_date_formats():
    from worker.workflows.supplier_invoices import _normalize_date

    assert _normalize_date("2016-12-28") == "2016-12-28"
    assert _normalize_date("28/12/2016") == "2016-12-28"
    assert _normalize_date("01/06/2023") == "2023-06-01"
    assert _normalize_date("28.12.2016") == "2016-12-28"
    assert _normalize_date("28 Décembre 2016") == "2016-12-28"
    assert _normalize_date("1er Mars 2025") == "2025-03-01"
    assert _normalize_date("3 janvier 2024") == "2024-01-03"
    assert _normalize_date("28 December 2016") == "2016-12-28"
    assert _normalize_date("June 15, 2025") == "2025-06-15"
    assert _normalize_date("17 oct. 2018") == "2018-10-17"
    assert _normalize_date("1er nov. 2018") == "2018-11-01"
    assert _normalize_date(None) is None
    assert _normalize_date("") is None
    assert _normalize_date("sans date") is None
    assert _normalize_date("120.0") is None


def test_invoice_line_quantity_null_becomes_one():
    line = si.InvoiceLine.model_validate(
        {"description": "FORMULE BRONZE PREMIUM", "quantity": None, "unit_price_ht": 14.95}
    )
    assert line.quantity == 1.0


def test_vat_harmonization_aligns_discount_line_from_invoice_total():
    invoice = si.ExtractedInvoice.model_validate(
        {
            "lines": [
                {"description": "Bronze D1", "quantity": 1, "unit_price_ht": 7.95, "line_total_ht": 7.95, "vat_rate": 20.0},
                {"description": "20% Non Profit Discount", "quantity": 1, "unit_price_ht": -1.59, "line_total_ht": -1.59, "vat_rate": 0.0},
            ],
            "total_ht": 6.36,
            "total_ttc": 7.63,
        }
    )
    notes = si._harmonize_vat_rates(invoice)
    assert invoice.lines[1].vat_rate == 20.0
    assert ["Taux de TVA de la ligne '20% Non Profit Discount' raligne a 20.0 % (total TTC de la facture)"] == notes
    assert si._amount_warnings(invoice) == []
    lines = si._compute_lines(invoice)
    assert all(line["tva_tx"] == 20.0 for line in lines)


def test_vat_harmonization_kept_out_when_total_disagrees():
    invoice = si.ExtractedInvoice.model_validate(
        {
            "lines": [
                {"description": "Article", "quantity": 1, "unit_price_ht": 10.0, "line_total_ht": 10.0, "vat_rate": 20.0},
                {"description": "Livraison", "quantity": 1, "unit_price_ht": 6.36, "line_total_ht": 6.36, "vat_rate": 0.0},
            ],
            "total_ht": 16.36,
            "total_ttc": 18.36,
        }
    )
    assert si._harmonize_vat_rates(invoice) == []
    assert invoice.lines[1].vat_rate == 0.0


def test_vat_harmonization_requires_single_nonzero_rate():
    invoice = si.ExtractedInvoice.model_validate(
        {
            "lines": [
                {"description": "A", "quantity": 1, "unit_price_ht": 5.0, "line_total_ht": 5.0, "vat_rate": 20.0},
                {"description": "B", "quantity": 1, "unit_price_ht": 5.0, "line_total_ht": 5.0, "vat_rate": 10.0},
                {"description": "Remise", "quantity": 1, "unit_price_ht": -1.0, "line_total_ht": -1.0, "vat_rate": 0.0},
            ],
            "total_ht": 9.0,
            "total_ttc": 11.4,
        }
    )
    assert si._harmonize_vat_rates(invoice) == []
    assert invoice.lines[2].vat_rate == 0.0


def test_compute_lines_uses_default_quantity_when_null():
    invoice = si.ExtractedInvoice.model_validate(
        {
            "lines": [{"description": "Forfait", "quantity": None, "unit_price_ht": 14.95}],
            "total_ht": 14.95,
            "total_ttc": 17.94,
        }
    )
    lines = si._compute_lines(invoice)
    assert lines[0]["qty"] == 1.0
    assert lines[0]["pu_ht"] == 14.95


def test_amounts_as_french_strings_are_coerced():
    invoice = si.ExtractedInvoice.model_validate(
        {
            "lines": [
                {
                    "description": "A",
                    "quantity": 1,
                    "unit_price_ht": "14,95",
                    "line_total_ht": "1 234,56",
                }
            ],
            "total_ht": "1\u00a0234,56",
            "total_ttc": "1\u202f480,00",
        }
    )
    assert invoice.lines[0].unit_price_ht == 14.95
    assert invoice.lines[0].line_total_ht == 1234.56
    assert invoice.total_ht == 1234.56
    assert invoice.total_ttc == 1480.0


async def test_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch,
        {"library_name_or_id": "Factures", "supplier_name": "ACME", "dry_run": True},
    )

    assert [doc["status"] for doc in result["documents"]] == ["dry_run", "dry_run"]
    assert "2025-06-15" in result["documents"][0]["detail"]
    assert not dolibarr.calls_to("createThirdparties")
    assert not dolibarr.calls_to("createSupplierinvoices")
    assert not dolibarr.calls_to("supplierinvoicesAddPayment")
