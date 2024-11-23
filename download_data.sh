#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SEPARATOR="${YELLOW}===================================================${NC}"

if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo -e "${RED}Error:${NC} Please install ${YELLOW}wget${NC} or ${YELLOW}curl${NC} to download the data."
    echo -e "${YELLOW}To install them on Ubuntu, run:${NC}"
    echo -e "  sudo apt update && sudo apt install -y wget curl"
    exit 1
fi

if ! command -v 7z &> /dev/null; then
    echo -e "${RED}Error:${NC} 7z is not installed. Please install it to proceed."
    echo -e "${YELLOW}To install it on Ubuntu, run:${NC}"
    echo -e "  sudo apt update && sudo apt install -y p7zip-full"
    exit 1
fi

TMPL_BASE_URL="https://storage.googleapis.com/polish_landscape/dataset/simulation_landscapes_0145.7z"
TMPL_DATA="simulation_landscapes_0145"

echo -e "${SEPARATOR}"
echo -e "${YELLOW}Downloading data...${NC}"
if ! $CMD "$TMPL_BASE_URL"; then
    echo -e "${RED}Failed to download data from $TMPL_BASE_URL${NC}"
    exit 1
fi
echo -e "${GREEN}Data downloaded successfully.${NC}"

echo -e "${SEPARATOR}"
echo -e "${YELLOW}Extracting the data into directory...${NC}"
if ! 7z x "${TMPL_DATA}.7z"; then
    echo -e "${RED}Failed to extract dataset.${NC}"
    exit 1
fi
echo -e "${GREEN}Data extracted successfully.${NC}"

rm "${TMPL_DATA}.7z"
echo -e "${GREEN}Temporary file removed: ${TMPL_dataset_SPN}.7z${NC}"

cd ..
echo -e "${SEPARATOR}"
echo -e "${GREEN}All operations completed successfully. The data is ready to use.${NC}"