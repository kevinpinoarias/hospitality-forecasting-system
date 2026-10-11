#!/usr/bin/env bash
#
# Deploy one Cloud Run service safely: the new revision is started WITHOUT
# any live traffic, tested at its own private address, and only then given
# the traffic. If the test fails the script stops and the live revision is
# untouched, so there is nothing to roll back.
#
# Usage:  deploy_service.sh SERVICE IMAGE api|page
#   api   test with scripts/smoke_test.py (/health, /predict, /errors/ranking)
#   page  test that the web page answers
# Needs PROJECT and REGION in the environment, and gcloud already logged in.

set -euo pipefail

service="$1"
image="$2"
check="$3"
project="${PROJECT:?PROJECT is not set}"
region="${REGION:?REGION is not set}"

gcloud_run() { gcloud run "$@" --project "$project" --region "$region" --quiet; }

# Whatever happens, do not leave the temporary "candidate" address behind.
trap 'gcloud_run services update-traffic "$service" --remove-tags candidate >/dev/null 2>&1 || true' EXIT

echo "Deploying $image to $service with no traffic..."
gcloud_run deploy "$service" --image "$image" --no-traffic --tag candidate

url=$(gcloud_run services describe "$service" --format=json |
  jq -r '.status.traffic[] | select(.tag == "candidate") | .url')
[ -n "$url" ] && [ "$url" != "null" ] || { echo "Could not find the candidate address" >&2; exit 1; }
echo "Candidate revision is at $url"

if [ "$check" = "api" ]; then
  python3 scripts/smoke_test.py "$url"
else
  curl -fsS --retry 6 --retry-delay 5 --retry-all-errors -o /dev/null "$url"
  echo "Page answered."
fi

echo "Test passed - sending live traffic to the new revision."
gcloud_run services update-traffic "$service" --to-latest
echo "$service is live."

# Housekeeping: keep the newest few revisions (for rollback) and delete the
# rest. Cloud Run refuses to delete a revision that is serving traffic, and a
# failure here must never fail a deploy that already succeeded.
keep=4
gcloud_run revisions list --service "$service" --sort-by="~metadata.creationTimestamp" --format="value(metadata.name)" |
  tail -n +"$((keep + 1))" |
  while read -r revision; do
    echo "Deleting old revision $revision"
    gcloud_run revisions delete "$revision" || echo "Could not delete $revision - leaving it."
  done || true
