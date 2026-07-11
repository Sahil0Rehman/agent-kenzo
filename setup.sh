#!/usr/bin/env bash
# Agent Kenzo — one-command setup for macOS and Linux.
#
# What this does, step by step:
#   1. Checks you have Python 3.12 or newer.
#   2. Creates an isolated Python environment in ./.venv (so Kenzo's
#      dependencies never clash with anything else on your machine).
#   3. Installs Kenzo and everything it needs into that environment.
#   4. Walks you through setting up an API key (optional, can skip and do
#      later).
#   5. Optionally builds the VS Code extension.
#
# You do not need to know anything about Python to run this — just open a
# terminal in this folder and run:
#
#   ./setup.sh
#
# If you get a "permission denied" error, run this once first:
#   chmod +x setup.sh

set -euo pipefail

BOLD="$(tput bold 2>/dev/null || true)"
GREEN="$(tput setaf 2 2>/dev/null || true)"
YELLOW="$(tput setaf 3 2>/dev/null || true)"
RED="$(tput setaf 1 2>/dev/null || true)"
RESET="$(tput sgr0 2>/dev/null || true)"

info()  { echo "${BOLD}==>${RESET} $1"; }
ok()    { echo "${GREEN}✓${RESET} $1"; }
warn()  { echo "${YELLOW}!${RESET} $1"; }
fail()  { echo "${RED}✗ $1${RESET}"; exit 1; }

cd "$(dirname "${BASH_SOURCE[0]}")"

echo ""
echo "${BOLD}Agent Kenzo — Setup${RESET}"
echo "This will take a few minutes the first time."
echo ""

# --- 1. Find a usable Python (3.12+) ---------------------------------------
info "Step 1/5: Checking for Python 3.12 or newer..."

PYTHON_BIN=""
for candidate in python3.13 python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")"
        major="${version%%.*}"
        minor="${version##*.}"
        if [ "$major" -eq 3 ] && [ "$minor" -ge 12 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    fail "Python 3.12+ was not found on your system.
   Install it from https://www.python.org/downloads/ (or via your OS's
   package manager, e.g. 'brew install python@3.12' on macOS), then run
   this script again."
fi
ok "Found $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN")"

# --- 2. Create the virtual environment --------------------------------------
info "Step 2/5: Setting up an isolated environment in ./.venv ..."
if [ -d ".venv" ]; then
    ok "./.venv already exists — reusing it."
else
    "$PYTHON_BIN" -m venv .venv
    ok "Created ./.venv"
fi

VENV_PY="./.venv/bin/python"
VENV_PIP="./.venv/bin/pip"

# --- 3. Install Kenzo --------------------------------------------------------
info "Step 3/5: Installing Kenzo and its dependencies (this can take a few minutes)..."
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PIP" install -e ".[dev]" --quiet
ok "Kenzo installed."

KENZO_BIN="./.venv/bin/kenzo"
if [ ! -x "$KENZO_BIN" ]; then
    fail "Installation finished but the 'kenzo' command was not created. Something went wrong — check the output above for errors."
fi
ok "'kenzo' command is ready."

# --- 4. API key setup (optional) --------------------------------------------
echo ""
info "Step 4/5: API key setup"
echo "Kenzo needs at least one AI provider's API key to actually run tasks"
echo "(it can index/analyze a repo without one, but can't call a model)."
echo ""
echo "Supported out of the box:"
echo "  1) Nemotron (via NVIDIA)     — env var: NVIDIA_API_KEY"
echo "  2) Gemini (Google)           — env var: GEMINI_API_KEY"
echo "  3) Any OpenAI-compatible API — env var: OPENAI_API_KEY"
echo ""
read -r -p "Set up a key now? [y/N] " setup_key
if [[ "$setup_key" =~ ^[Yy]$ ]]; then
    read -r -p "Which provider? [1=Nemotron / 2=Gemini / 3=OpenAI-compatible]: " provider_choice
    case "$provider_choice" in
        1) provider_name="nemotron" ;;
        2) provider_name="gemini" ;;
        3) provider_name="openai_compatible" ;;
        *) provider_name="" ;;
    esac
    if [ -n "$provider_name" ]; then
        read -r -s -p "Paste your API key (input hidden): " api_key
        echo ""
        if [ -n "$api_key" ]; then
            "$KENZO_BIN" config set "providers.${provider_name}.api_key" "$api_key" >/dev/null
            ok "Saved to ~/.kenzo/config.yaml (run 'kenzo config show' any time to confirm — keys are masked)."
        else
            warn "No key entered — skipping. You can set one later, see below."
        fi
    else
        warn "Not a valid choice — skipping. You can set one later, see below."
    fi
else
    warn "Skipped. You can do this any time later with:"
    echo "     ./.venv/bin/kenzo config set providers.nemotron.api_key YOUR_KEY_HERE"
    echo "   or by exporting an environment variable (NVIDIA_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)"
    echo "   before running kenzo."
fi

# --- 5. VS Code extension (optional) ----------------------------------------
echo ""
info "Step 5/5: VS Code extension (optional)"
read -r -p "Set up the VS Code extension too? [y/N] " setup_vscode
if [[ "$setup_vscode" =~ ^[Yy]$ ]]; then
    if ! command -v npm >/dev/null 2>&1; then
        warn "npm was not found — install Node.js from https://nodejs.org/ (any LTS version) and re-run this script, or run the commands below yourself later:"
        echo "     cd vscode-extension && npm install && npm run compile"
    else
        (cd vscode-extension && npm install --silent && npm run compile --silent)
        ok "VS Code extension built."
        echo ""
        echo "   To run it: open the 'vscode-extension' folder in VS Code and press F5"
        echo "   (this launches a second VS Code window with Kenzo's chat/changes/memory"
        echo "   panels active — the standard way to run an unpublished extension)."
    fi
else
    echo "   Skipped. Instructions are in README.md under 'VS Code Extension' whenever you want it."
fi

# --- Done --------------------------------------------------------------------
echo ""
echo "${GREEN}${BOLD}Setup complete!${RESET}"
echo ""
echo "Next steps:"
echo "  1. Activate the environment in your terminal:"
echo "       source .venv/bin/activate"
echo "  2. Go into any project folder you want Kenzo to work on and run:"
echo "       kenzo init"
echo "       kenzo"
echo ""
echo "  (Or skip activating and just use ./.venv/bin/kenzo directly.)"
echo ""
echo "See README.md for the full command reference."
