@echo off
setlocal
cd /d "%~dp0.."

rem ===================================================================
rem  Build the Windows executable: a folder holding SOMTrack.exe, and a
rem  zip of it for a GitHub release.
rem
rem  Everything is built outside the source tree -- by default in
rem  %LOCALAPPDATA%\SOMTrack-build -- so a synced or backed-up project
rem  folder does not pick up ~1 GB of build files.  Set SOMTRACK_BUILD_DIR
rem  to put it somewhere else.
rem
rem    packaging\build_exe.bat
rem ===================================================================

if "%SOMTRACK_BUILD_DIR%"=="" set "SOMTRACK_BUILD_DIR=%LOCALAPPDATA%\SOMTrack-build"
set "VENV=%SOMTRACK_BUILD_DIR%\venv"
set "PYEXE=%VENV%\Scripts\python.exe"

if not exist "%PYEXE%" (
    echo  Creating the build environment in %VENV% ...
    py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
)
if not exist "%PYEXE%" (
    echo  [X] Could not create a virtual environment. Is Python 3.10+ installed?
    exit /b 1
)

"%PYEXE%" -m pip install --quiet --upgrade pip
"%PYEXE%" -m pip install --quiet -r packaging\requirements-exe.txt || exit /b 1

rem the version number, for the name of the zip
"%PYEXE%" packaging\version.py > "%SOMTRACK_BUILD_DIR%\version.txt" || exit /b 1
set /p VERSION=<"%SOMTRACK_BUILD_DIR%\version.txt"

set "PYTHONPATH=%CD%"
"%PYEXE%" -m PyInstaller --noconfirm --clean ^
    --distpath "%SOMTRACK_BUILD_DIR%\dist" --workpath "%SOMTRACK_BUILD_DIR%\work" ^
    packaging\somtrack.spec || exit /b 1

copy /y LICENSE "%SOMTRACK_BUILD_DIR%\dist\SOMTrack\LICENSE.txt" >nul
copy /y README.md "%SOMTRACK_BUILD_DIR%\dist\SOMTrack\README.md" >nul
copy /y README_TW.md "%SOMTRACK_BUILD_DIR%\dist\SOMTrack\README_TW.md" >nul

set "ZIP=%SOMTRACK_BUILD_DIR%\SOMTrack-%VERSION%-windows-x64.zip"
if exist "%ZIP%" del "%ZIP%"
powershell -NoProfile -Command "Compress-Archive -Path '%SOMTRACK_BUILD_DIR%\dist\SOMTrack' -DestinationPath '%ZIP%'" || exit /b 1

echo.
echo  Built:  %SOMTRACK_BUILD_DIR%\dist\SOMTrack\SOMTrack.exe
echo  Zip:    %ZIP%
endlocal
