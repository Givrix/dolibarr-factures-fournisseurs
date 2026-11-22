"""Client REST direct pour l'API Dolibarr.

Le connecteur MCP généré depuis le swagger Restler déclare les corps de
requête comme tableaux de chaînes et déforme tout format d'argument : l'API
reçoit un corps vide ou enveloppé et refuse toute création (« socid field
missing »), alors que le même appel en HTTP direct aboutit. Ce client appelle
donc l'API REST directement avec la clé API Dolibarr (DOLAPIKEY).

Configuration (variables d'environnement) :
- DOLIBARR_BASE_URL : URL de l'API, ex. https://dolibarr.example.org/htdocs/api/index.php
- DOLIBARR_API_TOKEN : clé API de l'utilisateur Dolibarr (DOLAPIKEY)
"""

import os
from typing import Any

import httpx


class DolibarrError(RuntimeError):
    """Erreur HTTP renvoyée par l'API Dolibarr."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code} : {body[:500]}")


class DolibarrClient:
    """Client HTTP minimal pour l'API REST Dolibarr."""

    def __init__(
        self,
        base_url: str | None = None,
        api_token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("DOLIBARR_BASE_URL", "")).rstrip("/")
        self.api_token = api_token or os.environ.get("DOLIBARR_API_TOKEN", "")
        if not self.base_url or not self.api_token:
            raise ValueError(
                "Configuration Dolibarr manquante : renseignez DOLIBARR_BASE_URL "
                "(ex. https://dolibarr.example.org/htdocs/api/index.php) et "
                "DOLIBARR_API_TOKEN (clé DOLAPIKEY de l'utilisateur) dans l'environnement"
            )
        self._transport = transport

    async def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"DOLAPIKEY": self.api_token, "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=60, transport=self._transport) as client:
            response = await client.request(method, url, params=params, json=json_body, headers=headers)
        if response.status_code >= 400:
            raise DolibarrError(response.status_code, response.text)
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    async def list_thirdparties(self, sqlfilters: str = "", limit: int = 100) -> list[dict[str, Any]]:
        return await self._list("thirdparties", {"sqlfilters": sqlfilters, "limit": limit})

    async def create_thirdparty(self, payload: dict[str, Any]) -> Any:
        return await self.request("POST", "thirdparties", json_body=payload)

    async def list_payment_types(self, limit: int = 200) -> list[dict[str, Any]]:
        return await self._list("setup/dictionary/payment_types", {"limit": limit})

    async def list_bank_accounts(self, limit: int = 200) -> list[dict[str, Any]]:
        return await self._list("bankaccounts", {"limit": limit})

    async def list_supplier_invoices(
        self,
        thirdparty_ids: str = "",
        sqlfilters: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return await self._list(
            "supplierinvoices",
            {"thirdparty_ids": thirdparty_ids, "sqlfilters": sqlfilters, "limit": limit},
        )

    async def create_supplier_invoice(self, payload: dict[str, Any]) -> Any:
        return await self.request("POST", "supplierinvoices", json_body=payload)

    async def get_supplier_invoice(self, invoice_id: int) -> dict[str, Any]:
        return await self.request("GET", f"supplierinvoices/{invoice_id}")

    async def upload_supplier_invoice_document(self, ref: str, filename: str, content_base64: str) -> Any:
        return await self.request(
            "POST",
            "documents/upload",
            json_body={
                "filename": filename,
                "modulepart": "supplier_invoice",
                "ref": ref,
                "filecontent": content_base64,
                "fileencoding": "base64",
                "overwriteifexists": 1,
            },
        )

    async def add_supplier_invoice_line(self, invoice_id: int, line: dict[str, Any]) -> Any:
        return await self.request("POST", f"supplierinvoices/{invoice_id}/lines", json_body=line)

    async def validate_supplier_invoice(self, invoice_id: int) -> Any:
        return await self.request(
            "POST",
            f"supplierinvoices/{invoice_id}/validate",
            params={"idwarehouse": 0, "notrigger": 0},
        )

    async def add_supplier_invoice_payment(self, invoice_id: int, payload: dict[str, Any]) -> Any:
        return await self.request("POST", f"supplierinvoices/{invoice_id}/payments", json_body=payload)

    async def mark_supplier_invoice_paid(self, invoice_id: int, close_note: str = "") -> Any:
        return await self.request(
            "POST",
            f"supplierinvoices/{invoice_id}/settopaid",
            json_body={"close_code": "", "close_note": close_note},
        )

    async def list_supplier_invoice_payments(self, invoice_id: int) -> list[dict[str, Any]]:
        return await self._list(f"supplierinvoices/{invoice_id}/payments", {})

    async def reopen_supplier_invoice(self, invoice_id: int) -> Any:
        return await self.request(
            "POST", f"supplierinvoices/{invoice_id}/settounpaid", json_body={}
        )

    async def set_supplier_invoice_draft(self, invoice_id: int) -> Any:
        return await self.request(
            "POST",
            f"supplierinvoices/{invoice_id}/settodraft",
            params={"idwarehouse": -1, "notrigger": 0},
        )

    async def delete_supplier_invoice(self, invoice_id: int) -> Any:
        return await self.request("DELETE", f"supplierinvoices/{invoice_id}")

    async def _list(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            data = await self.request("GET", path, params=params)
        except DolibarrError as error:
            if error.status_code == 404:
                return []
            raise
        if isinstance(data, dict):
            data = data.get("data", data)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []
