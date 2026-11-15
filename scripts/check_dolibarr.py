"""Vérifie la connexion à l'API REST Dolibarr utilisée par le workflow.

Lit DOLIBARR_BASE_URL et DOLIBARR_API_TOKEN (variable d'environnement ou .env
à la racine du dépôt), puis effectue des lectures sans écriture :
  - liste des fournisseurs (thirdparties) ;
  - modes de paiement (recherche du code PRE) ;
  - comptes bancaires.

Usage :
    bash tools/uv.sh run --all-packages python scripts/check_dolibarr.py
"""

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


async def main() -> int:
    load_env()
    from worker.dolibarr import DolibarrClient, DolibarrError

    base_url = os.environ.get("DOLIBARR_BASE_URL", "")
    if not base_url or not os.environ.get("DOLIBARR_API_TOKEN"):
        print(
            "Configuration manquante : renseignez DOLIBARR_BASE_URL et "
            "DOLIBARR_API_TOKEN dans .env (voir .env.example)."
        )
        return 1
    print(f"Base URL : {base_url}")

    dolibarr = DolibarrClient()
    try:
        thirdparties = await dolibarr.list_thirdparties(limit=5)
        print(f"Tiers (5 premiers) : {len(thirdparties)}")
        for thirdparty in thirdparties:
            print(f"  - {thirdparty.get('id')}: {thirdparty.get('name') or thirdparty.get('nom')}")

        payment_types = await dolibarr.list_payment_types()
        prelevement = [pt for pt in payment_types if str(pt.get("code", "")).upper() == "PRE"]
        print(f"Modes de paiement : {len(payment_types)} ; prélèvement (PRE) :")
        for pt in prelevement:
            print(f"  - id {pt.get('id')}: {pt.get('label')}")

        accounts = await dolibarr.list_bank_accounts()
        print(f"Comptes bancaires : {len(accounts)}")
        for account in accounts:
            print(f"  - id {account.get('id')}: {account.get('label')}")

        invoices = await dolibarr.list_supplier_invoices(limit=3)
        print(f"Factures fournisseurs (3 dernières) : {len(invoices)}")
        return 0
    except DolibarrError as error:
        print(f"Échec : {error}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
