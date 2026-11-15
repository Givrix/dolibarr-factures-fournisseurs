# dolibarr-factures-fournisseurs

Workflow Mistral qui importe un dossier de PDF de factures fournisseurs dans Dolibarr :
chaque facture est créée, validée, puis réglée **par prélèvement** à sa **date de
facturation**, sur le **compte bancaire 1** (paramétrable).

## Fonctionnement

Entrée du workflow `dolibarr-supplier-invoices-import` :

```json
{
  "library_name_or_id": "Factures fournisseurs",
  "supplier_name": "ACME",
  "bank_account_id": 1,
  "payment_mode_code": "PRE",
  "create_missing_supplier": true,
  "document_ids": [],
  "dry_run": false
}
```

| Paramètre | Défaut | Rôle |
|---|---|---|
| `library_name_or_id` | — | Bibliothèque Mistral contenant les PDF de factures |
| `supplier_name` | — | Nom du fournisseur dans Dolibarr |
| `bank_account_id` | `1` | Compte bancaire Dolibarr à débiter |
| `payment_mode_code` | `PRE` | Mode de paiement Dolibarr (prélèvement) |
| `create_missing_supplier` | `true` | Crée le tiers fournisseur s'il n'existe pas |
| `document_ids` | `[]` | Restreint l'import à certains documents (tous les PDF si vide) |
| `max_documents` | `null` | Limite le nombre de PDF traités, dans l'ordre de la bibliothèque |
| `payment_mode_id` | `null` | ID Dolibarr du mode de paiement (évite la lecture des dictionnaires) |
| `dry_run` | `false` | Si vrai : extraction seule, aucune écriture dans Dolibarr |

Pour chaque PDF de la bibliothèque :

1. **Extraction** : le texte OCR de la bibliothèque est analysé par `mistral-medium-latest`
   (sortie structurée : date, référence fournisseur, lignes, totaux, TVA).
2. **Anti-doublon** : si une facture fournisseur avec la même `ref_supplier` existe déjà
   pour ce tiers, le document est marqué `skipped_duplicate`.
3. **Création** : facture fournisseur créée (`ref: auto`), lignes ajoutées, facture validée.
   En cas d'échec (lignes/validation), la facture brouillon est supprimée.
4. **Règlement** : paiement ajouté avec `closepaidinvoices=yes`, à la date de facturation,
   sur le compte bancaire choisi, avec le mode de paiement trouvé dans le dictionnaire
   Dolibarr (code `PRE`, sinon libellé contenant « prélèvement »).

Statuts par document : `paid`, `dry_run`, `skipped_duplicate`, `skipped_not_invoice`, `failed`.

Pour un premier essai sans risque, lancez le workflow avec `"dry_run": true` : il extrait
les données de chaque PDF et retourne ce qui serait créé, sans rien écrire dans Dolibarr.

## Prérequis

1. **Bibliothèque Mistral** : créez une bibliothèque dans Mistral Studio et téléversez-y
   les PDF de factures. Partagez-la avec le workspace si le workflow tourne sans identité
   utilisateur (schedule/webhook).
2. **API Dolibarr** : le worker appelle l'API REST de Dolibarr directement (le
   connecteur MCP généré depuis le swagger Restler déforme les corps POST et est
   donc abandonné). Renseignez dans l'environnement du worker :
   - `DOLIBARR_BASE_URL` : URL de l'API, ex. `https://dolibarr.anelosimus.info/htdocs/api/index.php` ;
   - `DOLIBARR_API_TOKEN` : clé API de l'utilisateur Dolibarr (envoyée en header `DOLAPIKEY`).
   L'utilisateur API doit avoir les droits sur les tiers, les factures fournisseurs,
   les banques et la lecture des dictionnaires (setup).
3. **Dolibarr** : le dictionnaire des modes de paiement doit contenir le prélèvement
   (code `PRE`) et le compte bancaire `1` doit exister.

## Développement

```bash
bash tools/install.sh          # bun install + uv sync (workspace complet)
bash tools/uv.sh run --package worker pytest
```

## Test local

Prérequis sur la machine : [uv](https://docs.astral.sh/uv/getting-started/installation/),
[Bun](https://bun.sh/docs/installation) et la CLI Mistral (`mistral`).

Les commandes se lancent **depuis la racine du dépôt** (là où se trouvent `apps.json`
et `package.json`), pas depuis `apps/`.

```bash
# 1. Clé API + nom de déploiement
cp .env.example .env            # puis renseignez MISTRAL_API_KEY, DOLIBARR_BASE_URL et DOLIBARR_API_TOKEN
# DEPLOYMENT_NAME (prérempli : dolibarr-factures-fournisseurs-dev) est requis :
# il nomme le déploiement du worker et la file de tâche ; le worker s'y
# enregistre et les exécutions y sont routées. Gardez la même valeur partout.

# 2. Dépendances (bun + uv)
bash tools/install.sh

# 3. Worker local avec rechargement (laisser tourner)
mistral apps dev

# 4. Dans un autre terminal, premier essai sans écriture dans Dolibarr :
# --pdf-dir téléverse les PDF du dossier dans la bibliothèque Mistral
# (créée si elle n'existe pas) avant de lancer l'import.
bash tools/uv.sh run --all-packages python scripts/run_import.py \
    --library "Factures fournisseurs" --supplier "ACME" \
    --pdf-dir ./factures --dry-run

# 5. Import réel (--limit pour traiter seulement les N premiers PDF, pratique en test)
bash tools/uv.sh run --all-packages python scripts/run_import.py \
    --library "Factures fournisseurs" --supplier "ACME" --limit 2
```

## Dépannage

**Vérifier la connexion Dolibarr** : le script suivant lit la configuration du `.env`
et effectue des lectures sans écriture (tiers, modes de paiement, comptes bancaires) :

```bash
bash tools/uv.sh run --all-packages python scripts/check_dolibarr.py
```

**Erreur 403 sur les modes de paiement** : la lecture des dictionnaires Dolibarr
(`GET /setup/dictionary/payment_types`) exige que l'utilisateur API ait un droit de lecture
factures, propales ou commandes clients (source Dolibarr 24.0, `api_setup.class.php`).
Deux options :

- accorder le droit dans Dolibarr : Utilisateurs & Groupes → permissions de l'utilisateur API ;
- ou passer l'ID directement : `--payment-mode-id <id>` (ID visible dans Dolibarr,
  Tiers → dictionnaires → modes de paiement, ou table `llx_c_paiement`).

## Déploiement

Le déploiement géré se fait depuis un dépôt GitHub (`deploy/docker/Dockerfile.worker`).
Nom de déploiement conseillé : `dolibarr-factures-fournisseurs-worker`. Pensez à définir
`DOLIBARR_BASE_URL` et `DOLIBARR_API_TOKEN` dans les settings de l'app déployée.
