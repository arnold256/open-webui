#!/bin/bash

# Open WebUI - Build and Push Script
# This script builds Docker images and pushes them to GitHub Container Registry

set -e  # Exit on error

# Load GitHub token from .env file
if [ -f .env ]; then
    GITHUB_TOKEN=$(grep "^GITHUB=" .env | cut -d "'" -f 2)
fi

# Configuration
GITHUB_USERNAME="arnold256"
VERSION=$(node -p "require('./package.json').version")
IMAGE_NAME="open-webui"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Open WebUI - Build and Push Script${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Version: ${YELLOW}${VERSION}${NC}"
echo -e "GitHub Username: ${YELLOW}${GITHUB_USERNAME}${NC}"
echo ""

# Check if GitHub token is available
if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${RED}ERROR: GitHub token not found!${NC}"
    echo "Please set GITHUB token in .env file or provide it manually."
    echo ""
    echo "To create a token:"
    echo "1. Go to: https://github.com/settings/tokens"
    echo "2. Generate new token (classic)"
    echo "3. Select scopes: write:packages, read:packages"
    echo "4. Add it to .env file as: GITHUB='your_token_here'"
    exit 1
fi

# Login to GitHub Container Registry
echo -e "${YELLOW}Logging in to GitHub Container Registry...${NC}"
echo "$GITHUB_TOKEN" | docker login ghcr.io -u ${GITHUB_USERNAME} --password-stdin

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to login to GitHub Container Registry${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Authentication successful${NC}"
echo ""

# Build variants
echo -e "${YELLOW}What would you like to build?${NC}"
echo "1) Standard (latest)"
echo "2) CUDA (GPU support)"
echo "3) Ollama (bundled with Ollama)"
echo "4) All variants"
read -p "Enter choice [1-4]: " choice

# Function to build and push
build_and_push() {
    local variant=$1
    local build_args=$2
    local tag_suffix=$3
    
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Building: ${variant}${NC}"
    echo -e "${GREEN}======================================${NC}"
    
    if [ -z "$tag_suffix" ]; then
        TAGS="-t ghcr.io/${GITHUB_USERNAME}/${IMAGE_NAME}:latest -t ghcr.io/${GITHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
    else
        TAGS="-t ghcr.io/${GITHUB_USERNAME}/${IMAGE_NAME}:${tag_suffix} -t ghcr.io/${GITHUB_USERNAME}/${IMAGE_NAME}:${VERSION}-${tag_suffix}"
    fi
    
    echo -e "${YELLOW}Building with tags: ${TAGS}${NC}"
    
    if [ -z "$build_args" ]; then
        docker buildx build ${TAGS} --push .
    else
        docker buildx build ${TAGS} ${build_args} --push .
    fi
    
    echo -e "${GREEN}✓ ${variant} build complete and pushed!${NC}"
}

# Build based on choice
case $choice in
    1)
        build_and_push "Standard" "" ""
        ;;
    2)
        build_and_push "CUDA" "--build-arg USE_CUDA=true" "cuda"
        ;;
    3)
        build_and_push "Ollama" "--build-arg USE_OLLAMA=true" "ollama"
        ;;
    4)
        build_and_push "Standard" "" ""
        build_and_push "CUDA" "--build-arg USE_CUDA=true" "cuda"
        build_and_push "Ollama" "--build-arg USE_OLLAMA=true" "ollama"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}✓ All builds completed successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "View your images at:"
echo "https://github.com/${GITHUB_USERNAME}?tab=packages"
echo ""
echo "Pull your images with:"
echo "  docker pull ghcr.io/${GITHUB_USERNAME}/${IMAGE_NAME}:latest"
echo "  docker pull ghcr.io/${GITHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo ""
