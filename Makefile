# TinyBrain Makefile
# Convenient shortcuts for common development tasks

.PHONY: help build test lint format docs clean setup

# Default target
help:
	@echo "🧠 TinyBrain Build System"
	@echo "========================="
	@echo ""
	@echo "Available targets:"
	@echo "  make build        - Build all targets"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Run SwiftLint"
	@echo "  make format       - Format code with SwiftFormat"
	@echo "  make docs         - Generate documentation"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make setup        - Set up development environment"
	@echo "  make bench        - Build and run benchmarks"
	@echo ""

# Build all targets
build:
	@echo "🔨 Building TinyBrain..."
	swift build

# Build in release mode
build-release:
	@echo "🔨 Building TinyBrain (Release)..."
	swift build -c release

# Run tests
test:
	@echo "🧪 Running tests..."
	swift test

# Run tests with code coverage
test-coverage:
	@echo "🧪 Running tests with coverage..."
	swift test --enable-code-coverage

# Run SwiftLint
lint:
	@echo "🔍 Running SwiftLint..."
	@if command -v swiftlint >/dev/null 2>&1; then \
		swiftlint lint --strict; \
	else \
		echo "⚠️  SwiftLint not installed. Run 'make setup' first."; \
		exit 1; \
	fi

# Format code with SwiftFormat
format:
	@echo "✨ Formatting code..."
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat . --config .swiftformat; \
	else \
		echo "⚠️  SwiftFormat not installed. Run 'make setup' first."; \
		exit 1; \
	fi

# Check formatting without making changes
format-check:
	@echo "🔍 Checking code format..."
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat . --config .swiftformat --lint; \
	else \
		echo "⚠️  SwiftFormat not installed. Run 'make setup' first."; \
		exit 1; \
	fi

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	swift package generate-documentation

# Preview documentation
docs-preview:
	@echo "📚 Previewing documentation..."
	swift package --disable-sandbox preview-documentation --target TinyBrainRuntime

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	swift package clean
	rm -rf .build
	rm -rf *.xcodeproj

# Set up development environment
setup:
	@echo "🔧 Setting up development environment..."
	@chmod +x Scripts/setup.sh
	@./Scripts/setup.sh

# Build and run benchmarks
bench: build-release
	@echo "⚡ Running benchmarks..."
	.build/release/tinybrain-bench --help

# Resolve dependencies
resolve:
	@echo "📦 Resolving dependencies..."
	swift package resolve

# Update dependencies
update:
	@echo "📦 Updating dependencies..."
	swift package update

# Generate Xcode project
xcode:
	@echo "🔨 Generating Xcode project..."
	swift package generate-xcodeproj
	@echo "✅ Open TinyBrain.xcodeproj"

# Run all quality checks (lint + format-check + test)
check: lint format-check test
	@echo "✅ All checks passed!"

# Pre-commit hook
pre-commit: format lint test
	@echo "✅ Pre-commit checks passed!"

