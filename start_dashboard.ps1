# Starts the API + dashboard at http://127.0.0.1:8000  (Ctrl+C to stop)
Start-Job { Start-Sleep 8; Start-Process "http://127.0.0.1:8000" } | Out-Null
& .\.venv\Scripts\python.exe -m uvicorn twin.api.server:app --host 127.0.0.1 --port 8000
