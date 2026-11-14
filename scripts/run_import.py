"""Lance une exécution du workflow d'import de factures fournisseurs.

Usage :
    uv run python scripts/run_import.py --library "Factures fournisseurs" --supplier "ACME"
    uv run python scripts/run_import.py --library "Factures fournisseurs" --supplier "ACME" --dry-run
    uv run python scripts/run_import.py --library "lib-abc123" --supplier "ACME" --document-id doc-1 doc-2

Nécessite MISTRAL_API_KEY (variable d'environnement ou .env à la racine du dépôt)
et un worker en cours d'exécution (`mistral apps dev`, lancé depuis la racine du dépôt).

Lancer via le workspace uv :
    bash tools/uv.sh run --all-packages python scripts/run_import.py --help
"""

import argparse
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


async def main() -> int:
    parser = argparse.ArgumentParser(description="Import de factures fournisseurs Dolibarr")
    parser.add_argument("--library", required=True, help="Nom ou ID de la bibliothèque Mistral")
    parser.add_argument("--supplier", required=True, help="Nom du fournisseur dans Dolibarr")
    parser.add_argument("--bank-account", type=int, default=1, help="ID du compte bancaire (défaut : 1)")
    parser.add_argument("--payment-mode", default="PRE", help="Code du mode de paiement (défaut : PRE)")
    parser.add_argument("--document-id", nargs="*", default=[], help="IDs de documents à traiter (tous si vide)")
    parser.add_argument("--no-create-supplier", action="store_true", help="Ne pas créer le fournisseur absent")
    parser.add_argument("--dry-run", action="store_true", help="Extraction seule, aucune écriture dans Dolibarr")
    parser.add_argument("--deployment", default=None, help="Nom du déploiement (défaut : DEPLOYMENT_NAME du .env)")
    args = parser.parse_args()

    load_env()
    if not os.environ.get("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY manquant : renseignez-le dans .env ou en variable d'environnement.")
        return 1

    from mistralai.workflows.client import get_mistral_client

    client = get_mistral_client(api_key=os.environ["MISTRAL_API_KEY"])
    kwargs = {}
    deployment = args.deployment or os.environ.get("DEPLOYMENT_NAME")
    if deployment:
        kwargs["deployment_name"] = deployment

    execution = await client.workflows.execute_workflow_async(
        workflow_identifier="dolibarr-supplier-invoices-import",
        input={
            "library_name_or_id": args.library,
            "supplier_name": args.supplier,
            "bank_account_id": args.bank_account,
            "payment_mode_code": args.payment_mode,
            "create_missing_supplier": not args.no_create_supplier,
            "document_ids": args.document_id,
            "dry_run": args.dry_run,
        },
        wait_for_result=True,
        **kwargs,
    )
    result = execution.result if hasattr(execution, "result") else execution
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
