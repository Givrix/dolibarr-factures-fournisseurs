"""Lance une exécution du workflow d'import de factures fournisseurs.

Usage :
    uv run python scripts/run_import.py --library "Factures fournisseurs" --supplier "ACME" --pdf-dir ./factures
    uv run python scripts/run_import.py --library "Factures fournisseurs" --supplier "ACME" --dry-run
    uv run python scripts/run_import.py --library "lib-abc123" --supplier "ACME" --document-id doc-1 doc-2
    uv run python scripts/run_import.py --library "Factures fournisseurs" --supplier "ACME" --limit 2 --dry-run

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
import time
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


async def upload_pdfs(client, library_name_or_id: str, pdf_dir: str) -> tuple[str, list[str]]:
    """Crée la bibliothèque si absente, téléverse les PDF, attend l'OCR. Retourne (library_id, document_ids)."""
    libraries = (await client.beta.libraries.list_async()).data
    library = None
    for candidate in libraries:
        if candidate.id == library_name_or_id or candidate.name == library_name_or_id:
            library = candidate
            break
    if library is None:
        library = await client.beta.libraries.create_async(name=library_name_or_id)
        print(f"Bibliothèque créée : {library.name} ({library.id})")
    else:
        print(f"Bibliothèque trouvée : {library.name} ({library.id})")

    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf")) + sorted(Path(pdf_dir).glob("*.PDF"))
    if not pdf_paths:
        print(f"Aucun PDF trouvé dans {pdf_dir}")
        return library.id, []

    document_ids = []
    for pdf_path in pdf_paths:
        document = await client.beta.libraries.documents.upload_async(
            library_id=library.id,
            file={
                "file_name": pdf_path.name,
                "content": pdf_path.read_bytes(),
                "content_type": "application/pdf",
            },
        )
        document_ids.append(document.id)
        print(f"Téléversé : {pdf_path.name} ({document.id})")

    print("Traitement OCR en cours...")
    deadline = time.monotonic() + 300
    pending = set(document_ids)
    while pending and time.monotonic() < deadline:
        await asyncio.sleep(3)
        for document_id in list(pending):
            document = await client.beta.libraries.documents.get_async(
                library_id=library.id, document_id=document_id
            )
            status = getattr(document, "process_status", None)
            if status in ("done", "error", "missing_content", "noop"):
                pending.discard(document_id)
                print(f"  {document_id} : {status}")
    if pending:
        print(f"Traitement toujours en cours après 300 s pour : {', '.join(sorted(pending))}")
    return library.id, document_ids


async def main() -> int:
    parser = argparse.ArgumentParser(description="Import de factures fournisseurs Dolibarr")
    parser.add_argument("--library", required=True, help="Nom ou ID de la bibliothèque Mistral")
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Dossier local de PDF à téléverser dans la bibliothèque avant l'import",
    )
    parser.add_argument("--supplier", required=True, help="Nom du fournisseur dans Dolibarr")
    parser.add_argument("--bank-account", type=int, default=1, help="ID du compte bancaire (défaut : 1)")
    parser.add_argument("--payment-mode", default="PRE", help="Code du mode de paiement (défaut : PRE)")
    parser.add_argument(
        "--payment-mode-id",
        type=int,
        default=None,
        help="ID Dolibarr du mode de paiement (évite la lecture des dictionnaires, si 403)",
    )
    parser.add_argument("--document-id", nargs="*", default=[], help="IDs de documents à traiter (tous si vide)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre de PDF traités (pratique pour les tests)",
    )
    parser.add_argument("--no-create-supplier", action="store_true", help="Ne pas créer le fournisseur absent")
    parser.add_argument("--dry-run", action="store_true", help="Extraction seule, aucune écriture dans Dolibarr")
    parser.add_argument("--deployment", default=None, help="Nom du déploiement (défaut : DEPLOYMENT_NAME du .env)")
    args = parser.parse_args()

    load_env()
    if not os.environ.get("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY manquant : renseignez-le dans .env ou en variable d'environnement.")
        return 1

    from mistralai.workflows.client import get_mistral_client
    from mistralai.workflows.core.auth import RunAs

    client = get_mistral_client(api_key=os.environ["MISTRAL_API_KEY"], run_as=RunAs.DEPLOYMENT)
    kwargs = {}
    deployment = args.deployment or os.environ.get("DEPLOYMENT_NAME")
    if deployment:
        kwargs["deployment_name"] = deployment

    document_ids = args.document_id
    if args.pdf_dir:
        _, uploaded_ids = await upload_pdfs(client, args.library, args.pdf_dir)
        if uploaded_ids:
            document_ids = uploaded_ids

    execution = await client.workflows.execute_workflow_async(
        workflow_identifier="dolibarr-supplier-invoices-import",
        input={
            "library_name_or_id": args.library,
            "supplier_name": args.supplier,
            "bank_account_id": args.bank_account,
            "payment_mode_code": args.payment_mode,
            "payment_mode_id": args.payment_mode_id,
            "create_missing_supplier": not args.no_create_supplier,
            "document_ids": document_ids,
            "max_documents": args.limit,
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
