import json
from typing import Any

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment

from mistralai.workflows import testing as workflows_testing
from mistralai.workflows.plugins.mistralai.connectors import ToolCallClient
from mistralai.workflows.plugins.mistralai.connectors.exceptions import ConnectorToolCallError

import worker.workflows.supplier_invoices as si
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
]

TASK_QUEUE = "test-task-queue"


class FakeDolibarr:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.next_invoice_id = 101
        self.existing_invoices: list[dict[str, Any]] = []

    def __call__(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((tool_name, arguments or {}))
        if tool_name.endswith("listThirdparties"):
            return {"content": [{"type": "text", "text": json.dumps([])}]}
        if tool_name.endswith("createThirdparties"):
            return {"content": [{"type": "text", "text": json.dumps({"id": 7, "name": "ACME"})}]}
        if tool_name.endswith("listSupplierinvoices"):
            return {"content": [{"type": "text", "text": json.dumps(self.existing_invoices)}]}
        if tool_name.endswith("createSupplierinvoices"):
            invoice = {"id": self.next_invoice_id, "ref": f"FF{self.next_invoice_id}"}
            self.next_invoice_id += 1
            return {"content": [{"type": "text", "text": json.dumps(invoice)}]}
        if tool_name.endswith("supplierinvoicesCreateLine"):
            return {"content": [{"type": "text", "text": json.dumps({"id": 1})}]}
        if tool_name.endswith("supplierinvoicesValidate"):
            return {"content": [{"type": "text", "text": json.dumps({"id": 1})}]}
        if tool_name.endswith("supplierinvoicesAddPayment"):
            return {"content": [{"type": "text", "text": json.dumps({"id": 1, "amount": 120.0})}]}
        if tool_name.endswith("setupRetrievePaymentTypes"):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            [
                                {"id": 2, "code": "VIR", "label": "Virement"},
                                {"id": 3, "code": "PRE", "label": "Prélèvement"},
                            ]
                        ),
                    }
                ]
            }
        if tool_name.endswith("listBankaccounts"):
            return {
                "content": [
                    {"type": "text", "text": json.dumps([{"id": 1, "label": "Compte courant"}])}
                ]
            }
        raise AssertionError(f"Outil Dolibarr inattendu : {tool_name}")

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

    async def list_async(self, library_id: str):
        class _Resp:
            data = [FakeDocument(pdf) for pdf in self.pdfs]

        return _Resp()

    async def text_content_async(self, library_id: str, document_id: str):
        return self.text_by_document[document_id]


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
    def __init__(self, extraction: dict[str, Any]) -> None:
        self.extraction = extraction

    async def parse_async(self, response_format, **kwargs):
        extraction = self.extraction

        class _Message:
            parsed = response_format.model_validate(extraction)

        class _Choice:
            message = _Message()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class FakeMistralClient:
    def __init__(self, pdfs: list[dict[str, Any]], text_by_document: dict[str, str], extraction: dict[str, Any]) -> None:
        self.beta = FakeBeta(FakeDocuments(pdfs, text_by_document))
        self.chat = FakeChat(extraction)


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    dolibarr: FakeDolibarr,
    client: FakeMistralClient,
) -> None:
    async def fake_call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        return dolibarr(tool_name, arguments)

    monkeypatch.setattr(ToolCallClient, "call_tool", fake_call_tool)
    monkeypatch.setattr(si, "get_mistral_client", lambda **kwargs: client)


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

    payments = dolibarr.calls_to("supplierinvoicesAddPayment")
    assert len(payments) == 2
    assert payments[0]["datepaye"] == "2025-06-15"
    assert payments[0]["payment_mode_id"] == 3
    assert payments[0]["accountid"] == 1
    assert payments[0]["closepaidinvoices"] == "yes"
    assert payments[0]["comment"].startswith("Prélèvement automatique")

    lines = dolibarr.calls_to("supplierinvoicesCreateLine")
    assert len(lines) == 2
    assert json.loads(lines[0]["request_data"][0]) == {
        "desc": "Prestation",
        "qty": 2.0,
        "pu_ht": 50.0,
        "tva_tx": 20.0,
    }
    assert dolibarr.calls_to("supplierinvoicesValidate")


async def test_existing_supplier_is_reused(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()

    async def fake_call_tool(self, tool_name, arguments=None):
        if tool_name.endswith("listThirdparties"):
            return {
                "content": [{"type": "text", "text": json.dumps([{"id": 12, "name": "acme"}])}]
            }
        return dolibarr(tool_name, arguments)

    monkeypatch.setattr(ToolCallClient, "call_tool", fake_call_tool)
    monkeypatch.setattr(si, "get_mistral_client", lambda **kwargs: client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert result["supplier_id"] == 12
    assert result["supplier_created"] is False
    assert not dolibarr.calls_to("createThirdparties")


async def test_dolibarr_404_empty_list_creates_supplier(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()

    async def fake_call_tool(self, tool_name, arguments=None):
        if tool_name.endswith("listThirdparties"):
            raise ConnectorToolCallError(
                "dolibarr_anelosimus_c428",
                response={
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Error calling tool 'listThirdparties': HTTP error 404: Not Found - "
                                "{'error': {'code': 404, 'message': 'Not Found: No third parties found'}}"
                            ),
                        }
                    ]
                },
            )
        return dolibarr(tool_name, arguments)

    monkeypatch.setattr(ToolCallClient, "call_tool", fake_call_tool)
    monkeypatch.setattr(si, "get_mistral_client", lambda **kwargs: client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert result["supplier_created"] is True
    assert dolibarr.calls_to("createThirdparties")


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


async def test_extraction_failure_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()
    client.chat.extraction = {"is_invoice": True, "invoice_date": None}
    install_fakes(monkeypatch, dolibarr, client)

    result = await run_workflow(
        monkeypatch, {"library_name_or_id": "Factures", "supplier_name": "ACME"}
    )

    assert [doc["status"] for doc in result["documents"]] == ["failed", "failed"]
    assert "Date" in result["documents"][0]["detail"]
    assert not dolibarr.calls_to("createSupplierinvoices")


def test_compute_lines_from_lines():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _compute_lines

    invoice = ExtractedInvoice(
        lines=[{"description": "Article", "quantity": 2.0, "unit_price_ht": 50.0, "vat_rate": 20.0}]
    )
    assert _compute_lines(invoice) == [
        {"desc": "Article", "qty": 2.0, "pu_ht": 50.0, "tva_tx": 20.0}
    ]


def test_compute_lines_from_totals_only():
    from worker.workflows.supplier_invoices import ExtractedInvoice, _compute_lines

    invoice = ExtractedInvoice(total_ttc=120.0)
    lines = _compute_lines(invoice)
    assert len(lines) == 1
    assert lines[0]["tva_tx"] == 20.0
    assert lines[0]["pu_ht"] == pytest.approx(100.0, rel=1e-3)


def test_unwrap_tool_result_handles_mcp_content():
    from worker.workflows.supplier_invoices import _unwrap_tool_result

    raw = {"content": [{"type": "text", "text": json.dumps([{"id": "1", "name": "ACME"}])}]}
    assert _unwrap_tool_result(raw) == [{"id": "1", "name": "ACME"}]


async def test_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch):
    dolibarr, client = base_fixtures()

    async def fake_call_tool(self, tool_name, arguments=None):
        if tool_name.endswith("listThirdparties"):
            return {
                "content": [{"type": "text", "text": json.dumps([{"id": 7, "name": "ACME"}])}]
            }
        return dolibarr(tool_name, arguments)

    monkeypatch.setattr(ToolCallClient, "call_tool", fake_call_tool)
    monkeypatch.setattr(si, "get_mistral_client", lambda **kwargs: client)

    result = await run_workflow(
        monkeypatch,
        {"library_name_or_id": "Factures", "supplier_name": "ACME", "dry_run": True},
    )

    assert [doc["status"] for doc in result["documents"]] == ["dry_run", "dry_run"]
    assert "2025-06-15" in result["documents"][0]["detail"]
    assert not dolibarr.calls_to("createThirdparties")
    assert not dolibarr.calls_to("createSupplierinvoices")
    assert not dolibarr.calls_to("supplierinvoicesAddPayment")
