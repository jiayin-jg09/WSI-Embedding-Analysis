@echo off
REM =============================================================================
REM WSI Embedding Analysis Pipeline - Launcher (Windows)
REM =============================================================================
REM Usage: run.bat [--cancer-type CHOL] [--n-folds 5]
REM =============================================================================

set SCRIPT_DIR=%~dp0

REM Check for virtual environment
if not exist "%SCRIPT_DIR%venv" (
    echo No virtual environment found.
    set /p answer="Create one? (y/n): "
    if /i "%answer%"=="y" (
        echo Creating virtual environment...
        python -m venv "%SCRIPT_DIR%venv"
        call "%SCRIPT_DIR%venv\Scripts\activate.bat"
        echo Installing dependencies...
        pip install -r "%SCRIPT_DIR%requirements.txt"
    ) else (
        echo Proceeding without virtual environment...
    )
) else (
    call "%SCRIPT_DIR%venv\Scripts\activate.bat"
    echo Activated virtual environment.
)

REM Check that data exists
if not exist "%SCRIPT_DIR%CLINICAL_FULL.parquet" (
    echo ERROR: CLINICAL_FULL.parquet not found in package directory.
    echo Place the clinical data file in: %SCRIPT_DIR%
    exit /b 1
)

echo.
echo ==============================================
echo Running WSI Embedding Analysis Pipeline
echo ==============================================
echo.

python "%SCRIPT_DIR%wsi_embedding_analysis.py" ^
    --embeddings-dir "%SCRIPT_DIR%embeddings" ^
    --clinical-data "%SCRIPT_DIR%CLINICAL_FULL.parquet" ^
    --output-dir "%SCRIPT_DIR%results" ^
    --cancer-type CHOL ^
    %*

echo.
echo Done! Results saved to: %SCRIPT_DIR%results\
pause
