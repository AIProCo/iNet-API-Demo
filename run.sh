#!/bin/bash
BUILD=false
# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
  -b) # Handle the -force flag
    BUILD=true
    ;;
  -h | --help) # Handle the help flag
    echo "Usage: script.sh [-b] [-h|--help]"
    echo '-b: force to rebuild the project'
    exit 0
    ;;
  *) # Handle unknown arguments
    echo "Unknown option: $1"
    echo "Use -h or --help for usage information."
    exit 1
    ;;
  esac
  shift # Move to the next argument
done

if [ "$BUILD" = true ]; then
  ./build.sh -force
fi

cp -r videos bin &&
  cp -r inputs bin &&
  ./bin/client

