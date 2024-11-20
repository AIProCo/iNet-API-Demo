#!/bin/bash
# if the FORCE flag is set to true, the script will delete all files in the build directory
# and rebuild the project from scratch
# Initialize variables for flags
FORCE=false
DEBUG=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -force)    # Handle the -force flag
            FORCE=true
            ;;
        -debug)    # Handle the -debug flag
            DEBUG=true
            ;;
        -h|--help) # Handle the help flag
            echo "Usage: script.sh [-force] [-debug] [-h|--help]"
            exit 0
            ;;
        *)         # Handle unknown arguments
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
    shift # Move to the next argument
done

echo "change to the directory that contains this _build.sh file"
cd "$(dirname "$0")"

# check if build directory exists, if not create it
if [ ! -d "build" ]; then
	echo "create build directory"
	mkdir build
fi
# change to build directory
cd build

if [ "$FORCE" = true ]; then
	echo "FORCE build: remove all old files"
	rm -rf *
fi

echo "run cmake"

if [ "$DEBUG" = true ]; then
    echo "DEBUG build"
    cmake -S .. -B . -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Debug
else
    echo "RELEASE build"
    cmake -S .. -B . -DCMAKE_C_COMPILER=/usr/bin/gcc-13 -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release
fi

echo "run make"
make -j$(nproc)

echo "back to <generator_dll> directory"
cd ..

