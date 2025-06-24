@echo off

if "%~5"=="" (
    echo Usage: %~n0 INPUT_FOLDER INPUT_SUFFIX CLIENT_NAME OUTPUT_FOLDER OUTPUT_SUFFIX RENDER_ARGS
    goto :eof
)

set "inFolder=%~1"
set "inSuffix=%~2"
set "clientName=%~3"
set "outFolder=%~4"
set "outSuffix=%~5"
set "renderArgs=%~6"

set "inClient=%inFolder%/%inSuffix%/%clientName%"
set "inFull=%inFolder%/%inSuffix%"
set "outFull=%outFolder%/%outSuffix%"
set "inClientOriginal=%inClient:_compressed=%"

set "inClientOriginal=%inClientOriginal:_60fps=%"
set "inClientOriginal=%inClientOriginal:_30fps=%"
set "inClientOriginal=%inClientOriginal:_20fps=%"
set "inClientOriginal=%inClientOriginal:_200=%"
set "inClientOriginal=%inClientOriginal:_100=%"
set "inClientOriginal=%inClientOriginal:_10=%"
set "inClientOriginal=%inClientOriginal:_20=%"
set "inClientOriginal=%inClientOriginal:_40=%"
set "inClientOriginal=%inClientOriginal:_60=%"
set "inClientOriginal=%inClientOriginal:_80=%"


set "FFMPEG=C:\Users\icguser\Documents\Development\ffmpeg\bin\ffmpeg.exe"
set "FFMPEG_ARGS1=-v quiet -stats -y -framerate 120"
set "FFMPEG_ARGS2=-c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 25 -pix_fmt nv12"

REM --- Initialize optional flags (default off) ---
set "FLAG_RENDER=0"
set "FLAG_VIDEO=0"
set "FLAG_VMAF=0"
set "FLAG_EVAL=0"
set "FLAG_DELETE=0"
set "FLAG_CLIENT=0"
set "FLAG_SERVER=0"

REM --- Shift away the first 6 arguments ---
shift
shift
shift
shift
shift
shift

REM --- Loop through any additional optional flags ---
:parse_optional
if "%~1"=="" goto done_parse

REM Check each flag (case-insensitive)
if /I "%~1"=="-r" (
    set "FLAG_RENDER=1"
    goto shift_optional
)
if /I "%~1"=="-v" (
    set "FLAG_VIDEO=1"
    goto shift_optional
)
if /I "%~1"=="-vmaf" (
    set "FLAG_VMAF=1"
    goto shift_optional
)
if /I "%~1"=="-e" (
    set "FLAG_EVAL=1"
    goto shift_optional
)
if /I "%~1"=="-d" (
    set "FLAG_DELETE=1"
    goto shift_optional
)
if /I "%~1"=="-c" (
    set "FLAG_CLIENT=1"
    goto shift_optional
)
if /I "%~1"=="-s" (
    set "FLAG_SERVER=1"
    goto shift_optional
)

echo Warning: Unrecognized option: %~1

:shift_optional
shift
goto parse_optional

:done_parse

IF "%FLAG_RENDER%"=="1" .\build\Release\splinter.exe "%inFull%" "%outFull%" -c "%inClientOriginal%" -w %renderArgs%
IF "%FLAG_VIDEO%"=="1" %FFMPEG% %FFMPEG_ARGS1% -i %outFull%/%%04d.png %FFMPEG_ARGS2% %outFolder%/%outSuffix%.mp4
IF "%FLAG_VMAF%"=="1" %FFMPEG% -r 120 -i %outFull%/%%04d.png -r 120 -i %inClientOriginal%/color/%%04d.png -lavfi "[0:v][1:v]libvmaf=log_path=%outFull%/vmaf.json:log_fmt=json" -f null - 
IF "%FLAG_EVAL%"=="1" python3 .\scripts\eval.py %inClientOriginal%/color %outFull% -ff 12 -ft -12 
IF "%FLAG_DELETE%"=="1" del /q "%outFull%\*.png"

IF "%FLAG_CLIENT%"=="1" %FFMPEG% %FFMPEG_ARGS1% -i %inClient%/color/%%04d.png %FFMPEG_ARGS2% %outFolder%/client_%clientName%.mp4
IF "%FLAG_SERVER%"=="1" .\scripts\make_video.bat 12 "%inFull%\main\color" "%outFolder%/server.mp4"