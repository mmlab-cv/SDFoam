param(
    [string]$Python = "",
    [string]$TorchRequirements = "requirements-cu132.txt"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if ($Python -eq "") {
    uv venv .venv
} else {
    uv venv .venv --python $Python
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

uv pip install --python $venvPython -r $TorchRequirements
uv pip install --python $venvPython .

& $venvPython -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')"
