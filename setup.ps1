# One-time setup: creates .venv and installs all dependencies (Windows PowerShell)
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Setup complete. Next: .\run_pipeline.ps1  then  .\start_dashboard.ps1"
