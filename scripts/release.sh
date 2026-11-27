#!/usr/bin/env bash
# release.sh - Rerelease to PyPI and Conda with auto version detection
# Usage: ./release.sh

set -e  # Exit on error

# 0️⃣ Detect version from pyproject.toml
if [ ! -f pyproject.toml ]; then
    echo "❌ pyproject.toml not found. Please run from the project root."
    exit 1
fi

VERSION=$(grep -m1 '^version =' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
if [ -z "$VERSION" ]; then
    echo "❌ Could not detect version from pyproject.toml"
    exit 1
fi

echo "🚀 Starting release process for version $VERSION..."

# 1️⃣ Update meta.yaml version
if [ -f conda-recipe/meta.yaml ]; then
    sed -i.bak "s/^  version: .*/  version: \"$VERSION\"/" conda-recipe/meta.yaml
    echo "✅ Updated version in conda-recipe/meta.yaml"
fi

# 2️⃣ Clean old builds
echo "🧹 Cleaning old build artifacts..."
rm -rf dist/ build/ *.egg-info

# 3️⃣ Build & upload to PyPI
echo "📦 Building and uploading to PyPI..."
python -m build
twine upload dist/*

# 4️⃣ Build & upload to Conda
echo "📦 Building and uploading to Conda..."
conda build conda-recipe/
PKG_FILE=$(conda build conda-recipe/ --output)
anaconda upload "$PKG_FILE"

echo "🎉 Release $VERSION complete!"
