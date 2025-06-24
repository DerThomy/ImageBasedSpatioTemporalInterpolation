@echo off
REM make_video.bat
REM Usage: make_video.bat STRIDE INPUT_FOLDER OUTPUT_FILE
REM Example: make_video.bat 6 "C:\MyImages" "C:\Videos\output.mp4"

REM Check for required arguments
if "%~3"=="" (
    echo Usage: %~n0 STRIDE INPUT_FOLDER OUTPUT_FILE
    goto :eof
)

set "stride=%~1"
set "inputfolder=%~2"
set "output=%~3"

REM Verify that the input folder exists
if not exist "%inputfolder%" (
    echo Input folder "%inputfolder%" does not exist.
    goto :eof
)

REM Enable delayed expansion for variable substring processing
setlocal EnableDelayedExpansion

REM Temporary file names
set "selected=temp_selected.txt"
set "filelist=images.txt"

REM Remove temporary files if they already exist
if exist "%selected%" del "%selected%"
if exist "%filelist%" del "%filelist%"

REM Loop over PNG files in the input folder (sorted by name)
for /F "delims=" %%i in ('dir /B /ON "%inputfolder%\*.png"') do (
    REM %%i is something like 0000.png
    set "fname=%%i"
    REM Extract the first 4 characters (assumed to be the number)
    set "numstr=!fname:~0,4!"
    REM Use a trick to convert the string (which may have leading zeros)
    REM to a decimal number (e.g., "0006" becomes 6)
    set /A num=1!numstr! - 10000
    REM Compute remainder: note that in batch files, use "%%" for modulus
    set /A rem=num %% stride
    if !rem! EQU 0 (
        REM Write the full path to the file (without quotes) into our temporary list
        echo %inputfolder%\%%i>>"%selected%"
    )
)

REM Check that at least one file was selected
if not exist "%selected%" (
    echo No images selected.
    goto :eof
)

REM Count the number of lines (files) in the selected list.
REM 'find /c /v ""' outputs a line like "---------- TEMP_SELECTED.TXT: 3"
for /F "tokens=2 delims=:" %%A in ('find /c /v "" "%selected%"') do set count=%%A
REM Remove any spaces from the count value
set count=%count: =%
if "%count%"=="0" (
    echo No images selected.
    goto :eof
)

REM Create the images.txt file for ffmpeg’s concat demuxer.
REM Each entry (except the last) is followed by a "duration" line.
set "lineNum=0"
for /F "delims=" %%A in (%selected%) do (
    set /A lineNum+=1
    echo file '%%A'>>"%filelist%"
    echo duration 0.1 >> "%filelist%"
)

REM Run ffmpeg to create a video from the image list.
REM -f concat tells ffmpeg to use the concat demuxer.
REM -safe 0 allows absolute paths.
REM -vsync vfr preserves the individual image durations.
REM -pix_fmt yuv420p ensures wide compatibility.
C:\Users\icguser\Documents\Development\ffmpeg\bin\ffmpeg.exe -v quiet -stats -y -f concat -safe 0 -i "%filelist%" -vsync vfr -pix_fmt yuv420p "%output%"

REM Clean up temporary files
del "%selected%"
del "%filelist%"

endlocal
