#!/usr/bin/env bash

set -e

# Check if jq is installed
if ! [ -x "$(command -v jq)" ]; then
    echo "jq is not installed" >& 2
    exit 1
fi

# Clean previous packages
if [ -d "pkg" ]; then
    rm -rf pkg
fi

if [ -d "pkg-node" ]; then
    rm -rf pkg-node
fi

# Build for both targets
wasm-pack build -t nodejs -d pkg-node
wasm-pack build -t browser -d pkg-browser

rm pkg-node/package.json
rm pkg-browser/package.json
