@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem ===================================================================
rem  SOMTrack 3.0 installer
rem
rem  Checks for a usable Python, builds a private virtual environment in
rem  .venv, and installs the dependencies into it.  Nothing is installed
rem  system-wide, so this cannot disturb another Python project on the
rem  same machine, and deleting the .venv folder undoes everything.
rem
rem  Safe to run again: it reuses an existing .venv and only installs
rem  what is missing or out of date.
rem ===================================================================

set "VENV=.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
set "NEED=3.10"

echo.
echo  ==========================================================
echo   SOMTrack 3.0  --  installer
echo  ==========================================================
echo.

rem -------------------------------------------------------------------
rem  1. Find a Python that is new enough
rem -------------------------------------------------------------------
echo  [1/6] Looking for Python %NEED% or newer...

set "PYCMD="

rem The py launcher knows about every Python installed.  Ask it for the
rem newest 3.x rather than naming versions: a list of versions to try goes
rem stale the moment a new Python is released, and would quietly skip it.
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYCMD=py -3"
)

if not defined PYCMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" >nul 2>&1
        if not errorlevel 1 set "PYCMD=python"
    )
)

if not defined PYCMD (
    echo.
    echo  [X] No Python %NEED% or newer was found.
    echo.
    echo      Install it from  https://www.python.org/downloads/
    echo      and tick "Add python.exe to PATH" in the installer.
    echo.
    echo      If Python is already installed, close this window, open a
    echo      new one and try again: a fresh window is needed to pick up
    echo      a PATH that changed during installation.
    echo.
    goto :fail
)

for /f "delims=" %%V in ('%PYCMD% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%V"
for /f "delims=" %%V in ('%PYCMD% -c "import sys;print(sys.executable)"') do set "PYPATH=%%V"
echo        found Python !PYVER!
echo        at !PYPATH!

rem A 32-bit Python cannot address enough memory for the permutation tests,
rem and several of the dependencies publish no 32-bit packages at all.
%PYCMD% -c "import sys;sys.exit(0 if sys.maxsize > 2**32 else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [X] That is a 32-bit Python. SOMTrack needs the 64-bit build:
    echo      several of its dependencies publish no 32-bit packages.
    echo      Install the "Windows installer ^(64-bit^)" from python.org.
    echo.
    goto :fail
)

rem -------------------------------------------------------------------
rem  2. Create the virtual environment
rem -------------------------------------------------------------------
echo.
echo  [2/6] Preparing the virtual environment in %VENV%\ ...

if exist "%PYEXE%" (
    "%PYEXE%" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo        the existing %VENV% is broken, rebuilding it
        rmdir /s /q "%VENV%" 2>nul
    ) else (
        echo        reusing the existing environment
    )
)

if not exist "%PYEXE%" (
    %PYCMD% -m venv "%VENV%"
    if errorlevel 1 (
        echo.
        echo  [X] Could not create the virtual environment.
        echo      On a managed machine this is usually a permissions
        echo      problem, or the folder is being synchronised
        echo      ^(OneDrive, pCloud, Dropbox^) while files are written.
        echo      Try a plain local folder such as C:\SOMTrack instead.
        echo.
        goto :fail
    )
    echo        created
)

if not exist "%PYEXE%" (
    echo.
    echo  [X] The virtual environment was created but has no python.exe.
    echo      Delete the .venv folder and run this script again.
    echo.
    goto :fail
)

rem -------------------------------------------------------------------
rem  3. pip itself
rem -------------------------------------------------------------------
echo.
echo  [3/6] Updating the package installer...
"%PYEXE%" -m pip install --quiet --upgrade pip setuptools wheel
if errorlevel 1 (
    echo  [!] Could not update pip. Carrying on with the version that is
    echo      already there; if the next step fails, this is why.
)

rem -------------------------------------------------------------------
rem  4. Required packages
rem -------------------------------------------------------------------
echo.
echo  [4/6] Installing the required packages. The first run downloads
echo        about 500 MB and takes a few minutes; later runs take
echo        seconds.
echo.
"%PYEXE%" -m pip install numpy pandas scipy scikit-learn matplotlib joblib openpyxl PySide6
if errorlevel 1 (
    echo.
    echo  [X] Installing the required packages failed.
    echo.
    echo      The usual causes, in order:
    echo        - no internet connection, or a proxy that blocks pip
    echo        - a company firewall intercepting https to pypi.org
    echo        - not enough disk space
    echo.
    echo      Behind a proxy, run these two lines and try again:
    echo        set HTTPS_PROXY=http://your.proxy:port
    echo        install.bat
    echo.
    goto :fail
)

rem -------------------------------------------------------------------
rem  5. Optional packages (small)
rem -------------------------------------------------------------------
echo.
echo  [5/6] Installing the optional packages ^(extra projections and
echo        colour palettes^). SOMTrack runs without these: any that
echo        fail are skipped, and the report records the omission.
echo.
for %%P in (pacmap umap-learn glasbey cmcrameri) do (
    echo        - %%P
    "%PYEXE%" -m pip install --quiet %%P
    if errorlevel 1 echo          not installed, SOMTrack will skip it
)

rem -------------------------------------------------------------------
rem  6. PyTorch, for SLISEMAP
rem
rem  Separate from step 5 because it is by far the biggest download here:
rem  a CUDA build is around 3 GB against 250 MB for the CPU one.  Set
rem  SOMTRACK_SKIP_TORCH=1 to skip it, or SOMTRACK_TORCH_CPU=1 to take
rem  the small CPU build even on a machine with a GPU.
rem
rem  Which build is right is a real decision -- an RTX 50-series card
rem  needs CUDA 12.8 or newer, and a mismatched build installs and
rem  imports cleanly before failing the first time it touches the GPU --
rem  so it is made in tools\install_torch.py, where it can be tested.
rem -------------------------------------------------------------------
echo.
if "%SOMTRACK_SKIP_TORCH%"=="1" (
    echo  [6/6] Skipping PyTorch ^(SOMTRACK_SKIP_TORCH=1^). SLISEMAP will
    echo        not be available; everything else works.
) else (
    echo  [6/6] Installing PyTorch and SLISEMAP.
    if "%SOMTRACK_TORCH_CPU%"=="1" (
        "%PYEXE%" tools\install_torch.py --cpu-only
    ) else (
        "%PYEXE%" tools\install_torch.py
    )
    if errorlevel 1 (
        echo  [!] SLISEMAP is not available. Everything else is unaffected.
    )
)

rem -------------------------------------------------------------------
rem  Report what is actually usable, by asking SOMTrack itself
rem -------------------------------------------------------------------
echo.
echo  ==========================================================
"%PYEXE%" -c "from somtrack.analysis import all_methods as m; ok=[s.label for s in m(available_only=True)]; no=[s.label for s in m() if not s.available()]; print('   Projections ready: ' + ', '.join(ok)); print('   Not installed    : ' + (', '.join(no) if no else 'none'))"
if errorlevel 1 (
    echo   [X] The packages installed, but SOMTrack does not start.
    echo       Run  start.bat  to see the full error message.
    echo  ==========================================================
    goto :fail
)
echo  ==========================================================
echo.
echo   Installation finished.
echo.
echo   Double-click  start.bat  to open SOMTrack.
echo.
echo   To remove everything, delete the .venv folder in this
echo   directory. Nothing was installed anywhere else.
echo.
pause
endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1
