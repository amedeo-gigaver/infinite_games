#!/usr/bin/env bash
set -o errexit  # Exit immediately if a command exits with a non-zero status
set -o nounset  # Treat unset variables as an error and exit immediately
set -o pipefail

# Settings
PUBLIC_REPO="https://github.com/amedeo-gigaver/infinite_games.git"
PRIVATE_REPO="https://github.com/infinite-mech/infinite_games.git"
RELEASE_DATE=$(date +"%Y-%m-%d-%H-%M")
NEW_BRANCH="sync-main-$RELEASE_DATE"
echo "Public release branch: $NEW_BRANCH"

trap 'echo "Error occurred in script at line: $LINENO"' ERR

# Ensure the token is set
if [[ -z "$PUBLIC_REPO_TOKEN" ]]; then
    echo "Error: PUBLIC_REPO_TOKEN is not set"
    exit 1
fi

# Inject the token into the repository URLs
PUBLIC_REPO_AUTH="https://${PUBLIC_REPO_TOKEN}@${PUBLIC_REPO#https://}"
PRIVATE_REPO_AUTH="https://${PUBLIC_REPO_TOKEN}@${PRIVATE_REPO#https://}"

echo "Clone the public repo"
git clone --quiet --branch main "$PUBLIC_REPO_AUTH" public-repo
cd public-repo || exit 1
git checkout -b "$NEW_BRANCH"

echo "Remove old files from this new branch"
git rm -r --cached . > /dev/null
rm -rf ./* > /dev/null

echo "Clone the private repo and copy the content"
cd ..
git clone --quiet --branch main "$PRIVATE_REPO_AUTH" private-repo
rsync -a --exclude='.git' ./private-repo/ ./public-repo/ > /dev/null

echo "Add and commit new private files in the public repo"
cd ./public-repo || exit 1
git add .
git commit -m "Release $RELEASE_DATE" > /dev/null

echo "Push the new branch to the public repo"
git push "$PUBLIC_REPO_AUTH" "$NEW_BRANCH"

echo "Changes were pushed to branch $NEW_BRANCH in the public repo - please manually open a PR!"
