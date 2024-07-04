@echo off

IF NOT EXIST "bin" MKDIR bin

REM Compile DLL
echo [!] Compiling 'SlimeFinder.dll'
nvcc --shared -o bin/SlimeFinder.dll src/SlimeFinder.cu

IF %ERRORLEVEL% NEQ 0 (
	pause
	exit /b %ERRORLEVEL%
)

REM Compile program with linked lib.dll
echo [!] Compiling 'SlimeFinder.exe'
g++ -O3 -o bin/SlimeFinder.exe -L. -l bin/SlimeFinder src/main.cpp src/UserIO.cpp

IF %ERRORLEVEL% NEQ 0 (
	pause
	exit /b %ERRORLEVEL%
) ELSE (
	echo [!] Built 'SlimeFinder.exe' successfully...
	pause
)
