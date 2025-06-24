
@echo off
setlocal enabledelayedexpansion

if "%~3"=="" (
    echo Usage: %~n0 INPUT_FOLDER INPUT_SUFFIX OUTPUT_FOLDER
    goto :eof
)

set "inFolder=%~1"
set "inSuffix=%~2"
set "outFolder=%~3"

:: Define the different mode-specific arguments
set count=0

@REM :: Names and params (separated by a special delimiter like ||)
set "args[0]=ours_ol||-m 0 -a left right"
set "args[1]=ours_extrapolate||-m 0 -l 11 -a left right"
set "args[2]=backward_bidirectional_ol||-m 1 -ns -hb -a"
set "args[3]=splatting_ol||-m 3 -ns -l 11 -a mid"
set "args[4]=timewarp_ol||-m 4 -a mid"
set NARGS=5

@REM set "args[0]=ours_l1||-m 0 -l 1 -a left right"
@REM set "args[1]=ours_l2||-m 0 -l 2 -a left right"
@REM set "args[2]=ours_l4||-m 0 -l 4 -a left right"
@REM set "args[3]=ours_l8||-m 0 -l 8 -a left right"
@REM set NARGS=4

@REM set "args[0]=ours_holy||-m 0 -l 11 -hb -ns -a"
@REM set "args[1]=ours_noaux_ns_holy||-m 0 -ns -hb -a"
@REM set "args[2]=ours_noaux_ns||-m 0 -ns -a"
@REM set "args[3]=ours_aux_ns||-m 0 -ns -a left right"
@REM set NARGS=4

@REM set "args[0]=timewarp_ol||-m 4 -a mid"
@REM set NARGS=1

:loop
if !count! GEQ !NARGS! goto :after_loop

set "line=!args[%count%]!"
echo ------ Processing item !count!: !line! -------

for /f "tokens=1,2 delims=||" %%A in ("!line!") do (
    @REM echo Running command %%A with params %%B
    call .\scripts\run.bat "%inFolder%" "%inSuffix%" client_ol "%outFolder%" %%A "%%B" -r -v -vmaf -e -d 

    @REM Stereo Evaluation
    @REM call .\scripts\run.bat "%inFolder%" "%inSuffix%" stereoleft_ol "%outFolder%" "%%A_left" "%%B" -r -v -e -d
    @REM call .\scripts\run.bat "%inFolder% " "%inSuffix%" stereoright_ol "%outFolder%" "%%A_right" "%%B" -r -v -e -d
)

set /a count+=1
goto loop

:after_loop

echo ------ Done Processing Methods -------
echo ------ Rendering Server/Client videos -------
@REM call .\scripts\run.bat "%inFolder%" "%inSuffix%" client_ol "%outFolder%" clientserver_ol "" -c -s