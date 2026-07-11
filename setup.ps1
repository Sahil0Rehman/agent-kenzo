# Agent Kenzo -- one-command setup for Windows.
#
# What this does, step by step:
#   1. Checks you have Python 3.12 or newer.
#   2. Creates an isolated Python environment in .\.venv (so Kenzo's
#      dependencies never clash with anything else on your machine).
#   3. Installs Kenzo and everything it needs into that environment.
#   4. Walks you through setting up an API key (optional, can skip and do
#      later).
#   5. Optionally builds the VS Code extension.
#
# You do not need to know anything about Python to run this.
#
# HOW TO RUN THIS SCRIPT:
#   1. Right-click this file's folder, choose "Open in Terminal" (or open
#      PowerShell and 'cd' into this folder).
#   2. Run:  .\setup.ps1
#   3. If Windows blocks it with a "running scripts is disabled" error, run
#      this once first (it only affects your current terminal session):
#        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Info($msg)  { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Fail($msg)  { Write-Host "[FAILED] $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Agent Kenzo -- Setup" -ForegroundColor White
Write-Host "This will take a few minutes the first time."
Write-Host ""

# --- 1. Find a usable Python (3.12+) ----------------------------------------
Write-Info "Step 1/5: Checking for Python 3.12 or newer..."

$pythonCmd = $null
$candidates = @(
    @{ Cmd = "py"; Args = @("-3.13") },
    @{ Cmd = "py"; Args = @("-3.12") },
    @{ Cmd = "py"; Args = @("-3") },
    @{ Cmd = "python"; Args = @() },
    @{ Cmd = "python3"; Args = @() }
)

foreach ($candidate in $candidates) {
    $exists = Get-Command $candidate.Cmd -ErrorAction SilentlyContinue
    if ($null -eq $exists) { continue }
    try {
        $versionArgs = $candidate.Args + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        $versionOutput = & $candidate.Cmd @versionArgs 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($versionOutput)) { continue }
        $parts = $versionOutput.Trim().Split(".")
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        if ($major -eq 3 -and $minor -ge 12) {
            $pythonCmd = $candidate
            break
        }
    } catch {
        continue
    }
}

if ($null -eq $pythonCmd) {
    Write-Fail "Python 3.12+ was not found on your system.`n   Install it from https://www.python.org/downloads/ -- during install, MAKE SURE to check the box that says 'Add python.exe to PATH' -- then close and reopen your terminal and run this script again."
}
$foundVersionArgs = $pythonCmd.Args + @("--version")
$foundVersion = & $pythonCmd.Cmd @foundVersionArgs
Write-Ok "Found $foundVersion"

# --- 2. Create the virtual environment --------------------------------------
Write-Info "Step 2/5: Setting up an isolated environment in .\.venv ..."
if (Test-Path ".venv") {
    Write-Ok ".\.venv already exists -- reusing it."
} else {
    $venvArgs = $pythonCmd.Args + @("-m", "venv", ".venv")
    & $pythonCmd.Cmd @venvArgs
    if ($LASTEXITCODE -ne 0) { Write-Fail "Could not create the virtual environment." }
    Write-Ok "Created .\.venv"
}

$VenvPy = ".\.venv\Scripts\python.exe"
$KenzoBin = ".\.venv\Scripts\kenzo.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Fail "Virtual environment looks broken (python.exe missing). Delete the .venv folder and run this script again."
}

# --- 3. Install Kenzo --------------------------------------------------------
Write-Info "Step 3/5: Installing Kenzo and its dependencies (this can take a few minutes)..."
& $VenvPy -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "Could not upgrade pip." }
& $VenvPy -m pip install -e ".[dev]" --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "Installation failed -- scroll up for the real error from pip." }
Write-Ok "Kenzo installed."

if (-not (Test-Path $KenzoBin)) {
    Write-Fail "Installation finished but the 'kenzo' command was not created. Something went wrong."
}
Write-Ok "'kenzo' command is ready."

# --- 4. API key setup (optional) --------------------------------------------
Write-Host ""
Write-Info "Step 4/5: API key setup"
Write-Host "Kenzo needs at least one AI provider's API key to actually run tasks"
Write-Host "(it can index/analyze a repo without one, but can't call a model)."
Write-Host ""
Write-Host "Supported out of the box:"
Write-Host "  1) Nemotron (via NVIDIA)     - env var: NVIDIA_API_KEY"
Write-Host "  2) Gemini (Google)           - env var: GEMINI_API_KEY"
Write-Host "  3) Any OpenAI-compatible API - env var: OPENAI_API_KEY"
Write-Host ""
$setupKey = Read-Host "Set up a key now? [y/N]"
if ($setupKey -match "^[Yy]") {
    $providerChoice = Read-Host "Which provider? [1=Nemotron / 2=Gemini / 3=OpenAI-compatible]"
    $providerName = ""
    if ($providerChoice -eq "1") { $providerName = "nemotron" }
    elseif ($providerChoice -eq "2") { $providerName = "gemini" }
    elseif ($providerChoice -eq "3") { $providerName = "openai_compatible" }

    if ($providerName -ne "") {
        $secureKey = Read-Host "Paste your API key (input hidden)" -AsSecureString
        $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
        $apiKey = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

        if ([string]::IsNullOrWhiteSpace($apiKey)) {
            Write-Warn "No key entered -- skipping. You can set one later, see below."
        } else {
            & $KenzoBin config set "providers.$providerName.api_key" "$apiKey" | Out-Null
            Write-Ok "Saved to %USERPROFILE%\.kenzo\config.yaml (run 'kenzo config show' any time to confirm -- keys are masked)."
        }
    } else {
        Write-Warn "Not a valid choice -- skipping. You can set one later, see below."
    }
} else {
    Write-Warn "Skipped. You can do this any time later with:"
    Write-Host "     .\.venv\Scripts\kenzo.exe config set providers.nemotron.api_key YOUR_KEY_HERE"
    Write-Host "   or by setting an environment variable (NVIDIA_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY)"
    Write-Host "   before running kenzo."
}

# --- 5. VS Code extension (optional) ----------------------------------------
Write-Host ""
Write-Info "Step 5/5: VS Code extension (optional)"
$setupVscode = Read-Host "Set up the VS Code extension too? [y/N]"
if ($setupVscode -match "^[Yy]") {
    $npmExists = Get-Command npm -ErrorAction SilentlyContinue
    if ($null -eq $npmExists) {
        Write-Warn "npm was not found -- install Node.js from https://nodejs.org/ (any LTS version) and re-run this script, or run these commands yourself later:"
        Write-Host "     cd vscode-extension; npm install; npm run compile"
    } else {
        Push-Location "vscode-extension"
        npm install --silent
        if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Fail "npm install failed -- scroll up for the real error." }
        npm run compile --silent
        if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Fail "npm run compile failed -- scroll up for the real error." }
        Pop-Location
        Write-Ok "VS Code extension built."
        Write-Host ""
        Write-Host "   To run it: open the 'vscode-extension' folder in VS Code and press F5"
        Write-Host "   (this launches a second VS Code window with Kenzo's chat/changes/memory"
        Write-Host "   panels active -- the standard way to run an unpublished extension)."
    }
} else {
    Write-Host "   Skipped. Instructions are in README.md under 'VS Code Extension' whenever you want it."
}

# --- Done --------------------------------------------------------------------
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate the environment in your terminal:"
Write-Host "       .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Go into any project folder you want Kenzo to work on and run:"
Write-Host "       kenzo init"
Write-Host "       kenzo"
Write-Host ""
Write-Host "  (Or skip activating and just use .\.venv\Scripts\kenzo.exe directly.)"
Write-Host ""
Write-Host "See README.md for the full command reference."
