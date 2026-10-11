# One-time Google Cloud setup for automatic deploys

The `deploy` job in `.github/workflows/ci.yml` pushes the tested image to
Artifact Registry and updates the two Cloud Run services whenever something
that goes into an image changes on `main`. It logs in to Google **without any
stored password or key file**: GitHub proves to Google, for each run, "I am the
`main` branch of this repository", and Google lets it act as a small service
account (Workload Identity Federation).

Until the four GitHub variables at the bottom exist, the deploy job is skipped.

Run these once, in PowerShell, logged in with `gcloud auth login`.

```powershell
$PROJECT = "hospitality-forecasting-1"
$NUMBER  = "56220375160"
$REGION  = "us-central1"
$REPO    = "kevinpinoarias/hospitality-forecasting-system"
$SA      = "github-deployer@$PROJECT.iam.gserviceaccount.com"
$RUNTIME = "$NUMBER-compute@developer.gserviceaccount.com"   # the account the services run as
```

## 1. Switch on the two Google services login needs
```powershell
gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project $PROJECT
```

## 2. Create the deployer account
```powershell
gcloud iam service-accounts create github-deployer --display-name "GitHub Actions deployer" --project $PROJECT
```

## 3. Give it only what it needs
```powershell
# push images to our one registry
gcloud artifacts repositories add-iam-policy-binding hospitality-repo --location $REGION --project $PROJECT --member "serviceAccount:$SA" --role roles/artifactregistry.writer

# update our two services - and nothing else in the project
gcloud run services add-iam-policy-binding hospitality-forecasting-api --region $REGION --project $PROJECT --member "serviceAccount:$SA" --role roles/run.developer
gcloud run services add-iam-policy-binding hospitality-forecasting-assistant --region $REGION --project $PROJECT --member "serviceAccount:$SA" --role roles/run.developer

# let it start revisions that run as the services' own account
gcloud iam service-accounts add-iam-policy-binding $RUNTIME --project $PROJECT --member "serviceAccount:$SA" --role roles/iam.serviceAccountUser
```

## 4. Teach Google to recognise GitHub - but only this repo's `main`
```powershell
gcloud iam workload-identity-pools create github --location global --project $PROJECT --display-name "GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-oidc --location global --workload-identity-pool github --project $PROJECT --issuer-uri "https://token.actions.githubusercontent.com" --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" --attribute-condition "assertion.repository=='$REPO' && assertion.ref=='refs/heads/main'"
```

## 5. Let that GitHub identity act as the deployer account
```powershell
gcloud iam service-accounts add-iam-policy-binding $SA --project $PROJECT --role roles/iam.workloadIdentityUser --member "principalSet://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"
```

## 6. Print the value GitHub needs
```powershell
gcloud iam workload-identity-pools providers describe github-oidc --location global --workload-identity-pool github --project $PROJECT --format "value(name)"
```

## 7. Tell GitHub (Settings > Secrets and variables > Actions > **Variables** tab)
None of these is a secret, so they are variables, not secrets.

| Name | Value |
|---|---|
| `GCP_PROJECT_ID` | `hospitality-forecasting-1` |
| `GCP_REGION` | `us-central1` |
| `GCP_DEPLOY_SA` | `github-deployer@hospitality-forecasting-1.iam.gserviceaccount.com` |
| `GCP_WIF_PROVIDER` | the line printed by step 6 (starts `projects/56220375160/...`) |

## 8. Optional: approve each deploy by hand
Settings > Environments > `production` > tick **Required reviewers** and add
yourself. Each deploy then waits for your click.

## Undoing it
Delete the `GCP_WIF_PROVIDER` variable and deploys stop at once. To remove
the access entirely: `gcloud iam service-accounts delete $SA --project $PROJECT`
and `gcloud iam workload-identity-pools delete github --location global --project $PROJECT`.
