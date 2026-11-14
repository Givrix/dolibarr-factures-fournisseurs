"""Sonde : identifie le format d'arguments que le connecteur Dolibarr transmet intact.

Appelle ``createSupplierinvoices`` avec ``socid = -1`` : aucun objet ne peut être
créé (tiers inexistant), mais Dolibarr distingue deux cas :
  - « socid field missing » : le corps de la requête est perdu ou déformé ;
  - toute autre erreur (ex. tiers introuvable) : le corps est transmis tel quel.

Le format gagnant est celui à utiliser dans le workflow (déjà couvert par la
cascade de :func:`worker.workflows.supplier_invoices._create_arg_candidates`).

Usage :
    bash tools/uv.sh run --all-packages python scripts/probe_create_formats.py

Nécessite MISTRAL_API_KEY (variable d'environnement ou .env à la racine du dépôt).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env_file = REPO_ROOT / ".env"
    if env_file.exists() and not os.environ.get("MISTRAL_API_KEY"):
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


PAYLOAD = {
    "socid": -1,
    "date": 1685577600,
    "ref_supplier": "PROBE-SONDE",
    "multicurrency_code": "EUR",
}


def _error_text(result) -> str:
    metadata = getattr(result, "metadata", None)
    mcp_meta = getattr(metadata, "mcp_meta", None)
    if mcp_meta is not None and not getattr(mcp_meta, "is_error", False):
        return ""
    content = getattr(result, "content", None) or []
    return " ".join(getattr(block, "text", "") for block in content)


def _classify(text: str) -> str:
    if "field missing" in text or "Invalid params" in text or "invalid params" in text:
        return "CORPS PERDU/DEFORME"
    if text:
        return "corps transmis (erreur métier attendue)"
    return "succès inattendu"


async def main() -> int:
    load_env()
    if not os.environ.get("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY manquant : renseignez-le dans .env ou en variable d'environnement.")
        return 1

    from mistralai.workflows.client import get_mistral_client
    from mistralai.workflows.core.auth import RunAs
    from worker.workflows.supplier_invoices import DOLIBARR_CONNECTOR_NAME, _create_arg_candidates

    client = get_mistral_client(api_key=os.environ["MISTRAL_API_KEY"], run_as=RunAs.DEPLOYMENT)
    candidates = _create_arg_candidates(PAYLOAD, {})
    print(f"Sonde {DOLIBARR_CONNECTOR_NAME} / createSupplierinvoices avec socid=-1\n")
    for index, arguments in enumerate(candidates, start=1):
        label = json.dumps(
            {key: (value if not isinstance(value, (dict, list)) else "<...>") for key, value in arguments.items()},
            ensure_ascii=False,
        )
        try:
            result = await client.beta.connectors.call_tool_async(
                connector_id_or_name=DOLIBARR_CONNECTOR_NAME,
                tool_name="createSupplierinvoices",
                arguments=arguments,
            )
            text = _error_text(result)
        except Exception as error:
            parts = [str(error)]
            current = error.__cause__
            while current is not None:
                parts.append(str(current))
                current = current.__cause__
            text = " | ".join(part for part in parts if part)
        print(f"[{index}/{len(candidates)}] {label}")
        print(f"    -> {_classify(text)}")
        print(f"    {text[:400]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
