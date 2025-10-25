#!/bin/bash
# TinyBrain Development Environment Setup Script

set -e

echo "🧠 TinyBrain Development Setup"
echo "================================"
echo ""

# Check macOS version
echo "Checking system requirements..."
os_version=$(sw_vers -productVersion)
echo "✓ macOS version: $os_version"

# Check Xcode
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Xcode not found. Please install Xcode 16+ from the App Store."
    exit 1
fi

xcode_version=$(xcodebuild -version | head -n 1)
echo "✓ $xcode_version"

# Check Swift version
swift_version=$(swift --version | head -n 1)
echo "✓ $swift_version"

# Install SwiftFormat if not present
if ! command -v swiftformat &> /dev/null; then
    echo ""
    echo "Installing SwiftFormat..."
    if command -v brew &> /dev/null; then
        brew install swiftformat
    else
        echo "⚠️  Homebrew not found. Please install SwiftFormat manually:"
        echo "   https://github.com/nicklockwood/SwiftFormat"
    fi
else
    echo "✓ SwiftFormat installed"
fi

# Install SwiftLint if not present
if ! command -v swiftlint &> /dev/null; then
    echo ""
    echo "Installing SwiftLint..."
    if command -v brew &> /dev/null; then
        brew install swiftlint
    else
        echo "⚠️  Homebrew not found. Please install SwiftLint manually:"
        echo "   https://github.com/realm/SwiftLint"
    fi
else
    echo "✓ SwiftLint installed"
fi

echo ""
echo "Resolving Swift Package dependencies..."
swift package resolve

echo ""
echo "Building TinyBrain..."
swift build

echo ""
echo "Running tests..."
swift test

echo ""
echo "✅ Setup complete! You're ready to build TinyBrain."
echo ""
echo "Quick commands:"
echo "  make build       - Build all targets"
echo "  make test        - Run tests"
echo "  make lint        - Run linting"
echo "  make docs        - Generate documentation"
echo ""

