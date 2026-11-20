#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repository_url="https://github.com/VIIWQX/3dv_group6.git"
cd -- "$repository_root"

if [[ ! -d .git ]]; then
  git init
  git symbolic-ref HEAD refs/heads/main
fi
if git remote get-url origin >/dev/null 2>&1; then
  existing_url="$(git remote get-url origin)"
  if [[ "$existing_url" != "$repository_url" && "$existing_url" != "git@github.com:VIIWQX/3dv_group6.git" ]]; then
    echo "Existing origin differs from the Group 6 repository: $existing_url" >&2
    exit 1
  fi
else
  git remote add origin "$repository_url"
fi
git add --all
if ! git diff --cached --quiet; then
  git -c user.name='Group 6' -c user.email='73103714+VIIWQX@users.noreply.github.com' commit -m 'Add Group 6 image-to-NeRF pipeline and ten-object results'
fi
git branch -M main
git push --set-upstream origin main
