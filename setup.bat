@echo off
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
%PY% -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if not exist .env copy .env.example .env
echo Setup complete. Edit .env if you want LLM-enhanced match reasoning.
