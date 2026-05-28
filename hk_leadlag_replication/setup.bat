@echo off
REM Setup script for HK-LeadLag - Windows
REM Usage: double-click or run in cmd

echo === Creating virtual environment ===
python -m venv .venv
if errorlevel 1 (
    echo Failed to create venv. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

echo === Activating venv and installing dependencies ===
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo === Running tests ===
pytest -q --tb=short

echo.
echo === Setup complete. ===
echo To activate later: .venv\Scripts\activate.bat
echo To run the notebook: jupyter lab notebooks/01_walkthrough.ipynb
echo To run the main pipeline: python main.py --config configs/simulation_default.yaml
pause
