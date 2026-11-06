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
rem  This deliberately uses python.exe and NOT pythonw.exe.  pythonw has no
rem  console, which looks tidier, but worker processes cannot be started
rem  under it on Windows: the permutation test dies with a
rem  TerminatedWorkerError minutes into a run, after the expensive part is
rem  already done.  Pointing the interpreter, multiprocessing and loky at
rem  python.exe does not fix it, and neither does repairing the stdio
rem  handles -- the only thing that works is not being pythonw.
rem
rem  The console starts minimised and the app hides it when it can, so it
rem  is at worst a taskbar entry rather than a window over the app.
rem
rem  Check it can start first: an error here is visible, and a broken
rem  install would otherwise look like nothing happening at all.
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

rem Launch it minimised.  Two things that do NOT work, both tried and
rem measured, so nobody repeats them:
rem
rem   - pythonw.exe: no console at all, but loky worker processes die under
rem     it, so the permutation test fails minutes into a run.
rem   - powershell Start-Process -WindowStyle Hidden: hides the console, but
rem     Windows applies that show-state to the process's *first* ShowWindow
rem     call, so Qt's main window never appears either.
rem
rem So: a minimised console, which the app then hides itself if it finds it
rem owns it alone.
start "SOMTrack" /min "%PYCON%" -m somtrack

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
