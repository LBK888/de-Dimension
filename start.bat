@echo off
setlocal
cd /d "%~dp0"

rem ===================================================================
rem  SOMTrack 3.0 launcher
rem
rem  Opens the desktop app using the virtual environment that
rem  install.bat built.  Any argument is passed through to the command
rem  line, so this doubles as a general entry point:
rem
rem    start.bat                     the desktop app
rem    start.bat methods             the projections, with their citations
rem    start.bat metrics             the locomotion metric catalogue
rem    start.bat demo .\demo_data    a synthetic data set to try it on
rem    start.bat run data.csv --features-table -o results
rem ===================================================================

set "PYCON=.venv\Scripts\python.exe"
set "PYWIN=.venv\Scripts\pythonw.exe"

if not exist "%PYCON%" (
    echo.
    echo  [X] SOMTrack is not installed yet.
    echo.
    echo      Double-click  install.bat  first. It only has to be done
    echo      once, and takes a few minutes.
    echo.
    pause
    endlocal
    exit /b 1
)

rem -------------------------------------------------------------------
rem  With arguments: a command-line request.  Keep the console so the
rem  output is readable, and pass the exit code through.
rem -------------------------------------------------------------------
if not "%~1"=="" (
    "%PYCON%" -m somtrack %*
    call :hold
    endlocal
    exit /b
)

rem -------------------------------------------------------------------
rem  No arguments: open the window.
rem
rem  Check that it can actually start *before* handing over to
rem  pythonw.exe.  pythonw has no console, which is what keeps a black
rem  box from sitting behind the app, but it also discards every error
rem  message -- so a failure would look like nothing happening at all.
rem  Importing first means a broken install reports itself here, with
rem  the traceback visible.
rem -------------------------------------------------------------------
"%PYCON%" -c "import somtrack.ui.app" 2>&1
if errorlevel 1 (
    echo.
    echo  [X] SOMTrack could not start. The error is above.
    echo.
    echo      If it names a missing module, run install.bat again.
    echo      If it mentions Qt or a platform plugin, the machine may
    echo      need updated graphics drivers, or may be a remote desktop
    echo      session without hardware acceleration.
    echo.
    pause
    endlocal
    exit /b 1
)

if exist "%PYWIN%" (
    start "SOMTrack" "%PYWIN%" -m somtrack
) else (
    "%PYCON%" -m somtrack
)

endlocal
exit /b 0

rem -------------------------------------------------------------------
:hold
rem `exit /b %errorlevel%` inside a parenthesised block would expand at
rem parse time, long before the command ran, and always report the code
rem from before.  A subroutine is parsed when it is called, so the value
rem here is the real one.
set "RC=%errorlevel%"
echo.
pause
exit /b %RC%
