@REM Set current working directory
cd /D "%~dp0"

@echo --------
@echo Activate venv
@echo --------
@echo
@call .venv\Scripts\activate.bat

@echo --------
@echo Starting script
@echo --------
@echo
python client_example.py
cd ..