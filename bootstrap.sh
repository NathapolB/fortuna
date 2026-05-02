#!/usr/bin/env bash
# bootstrap.sh — Run ONCE to initialize git, venv, and push to GitHub.
# After running this, delete this file (or it will stay in git history, which is fine).
#
# Pre-requisites:
#   - python3.12 installed (check: python3.12 --version)
#   - gh CLI installed and authenticated (check: gh auth status)
#
# Usage:
#   cd ~/projects/fortuna
#   bash bootstrap.sh

set -euo pipefail

echo "=== Project Fortuna Bootstrap ==="
echo ""

# 1. Verify we are NOT inside iCloud Drive
if echo "$PWD" | grep -q "com~apple~CloudDocs"; then
    echo "ERROR: You are inside iCloud Drive. Move to ~/projects/fortuna/ first."
    exit 1
fi

echo "Path check: OK ($PWD)"

# 2. Create data/.nosync sentinel
mkdir -p data/raw/cache
touch data/.nosync
echo "Created data/.nosync sentinel"

# 3. Create logs directory
mkdir -p logs
echo "Created logs/"

# 4. Git init (if not already a repo)
if [ ! -d .git ]; then
    git init
    git checkout -b main
    echo "Git initialized"
else
    echo "Git already initialized"
fi

# 5. Feature branch
BRANCH="feat/phase-0-1-bootstrap"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
echo "On branch: $BRANCH"

# 6. Create venv — prefer 3.12, fall back to 3.11, 3.13, then python3
if [ ! -d .venv ]; then
    PYBIN=""
    for cand in python3.12 python3.11 python3.13 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            PYBIN="$cand"
            break
        fi
    done
    if [ -z "$PYBIN" ]; then
        echo "ERROR: No suitable Python found. Install python3.12 via: brew install python@3.12"
        exit 1
    fi
    PYVER=$("$PYBIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Using Python: $PYBIN ($PYVER)"
    "$PYBIN" -m venv .venv
    echo "Created .venv with $PYBIN"
else
    echo ".venv already exists"
fi

# 7. Install dependencies
echo ""
echo "Installing dependencies..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo ""
echo "Verifying torch CPU build:"
.venv/bin/python -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')"

# 8. Create GitHub repo (private)
echo ""
if gh repo view fortuna --json name &>/dev/null 2>&1; then
    echo "GitHub repo 'fortuna' already exists — adding remote"
    git remote add origin "$(gh repo view fortuna --json sshUrl -q .sshUrl)" 2>/dev/null || echo "Remote already set"
else
    echo "Creating private GitHub repo 'fortuna'..."
    gh repo create fortuna --private --source=. --remote=origin
fi

# 9. Initial commit and push
echo ""
echo "Making initial commit..."
git add -A
git commit -m "chore: bootstrap project structure (Phase 0)" || echo "Nothing to commit"

echo ""
echo "Pushing to origin/$BRANCH..."
git push -u origin "$BRANCH"

echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. python scripts/backfill.py --start 2005-01-01 --end 2026-04-30"
echo "  3. python tests/create_fixtures.py"
echo "  4. pytest tests/test_store_dedup.py tests/test_parser.py -v"
echo "  5. python scripts/eda.py"
echo "  6. Open PR: gh pr create --title 'Phase 0 + 1 bootstrap' --body 'See SPEC §8'"
echo ""
echo "After Phase 1 acceptance criteria are all checked:"
echo "  gh pr merge --squash"
echo "  Nash: enable branch protection on GitHub: Settings → Branches → force-push protection"
