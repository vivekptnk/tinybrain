#!/bin/bash
# TinyBrain Benchmark Automation Script
# TB-007 Phase 4: Run benchmarks across devices

set -e

echo "🧠 TinyBrain Benchmark Runner"
echo "=============================="
echo ""

# Configuration
BENCH_BIN=".build/release/tinybrain-bench"
SCENARIOS="benchmarks/scenarios.yml"
OUTPUT_DIR="benchmarks/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if benchmark binary exists
if [ ! -f "$BENCH_BIN" ]; then
    echo -e "${YELLOW}⚠️  Benchmark binary not found. Building in release mode...${NC}"
    swift build -c release
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Detect device
DEVICE=$(uname -m)
OS=$(uname -s)
HOSTNAME=$(hostname -s)

echo "Device: $HOSTNAME ($DEVICE)"
echo "OS: $OS"
echo ""

# Run benchmarks
if [ -f "$SCENARIOS" ]; then
    echo -e "${GREEN}Running scenarios from: $SCENARIOS${NC}"
    echo ""
    
    # JSON output
    JSON_FILE="$OUTPUT_DIR/benchmark_${HOSTNAME}_${TIMESTAMP}.json"
    $BENCH_BIN --scenario "$SCENARIOS" --output json > "$JSON_FILE"
    echo -e "${GREEN}✅ JSON results: $JSON_FILE${NC}"
    
    # Markdown output
    MD_FILE="$OUTPUT_DIR/benchmark_${HOSTNAME}_${TIMESTAMP}.md"
    echo "# Benchmark Results: $HOSTNAME" > "$MD_FILE"
    echo "**Date:** $(date)" >> "$MD_FILE"
    echo "**Device:** $DEVICE" >> "$MD_FILE"
    echo "" >> "$MD_FILE"
    $BENCH_BIN --scenario "$SCENARIOS" --output markdown >> "$MD_FILE"
    echo -e "${GREEN}✅ Markdown results: $MD_FILE${NC}"
    
else
    echo -e "${YELLOW}⚠️  No scenario file found at $SCENARIOS${NC}"
    echo "Running simple benchmark..."
    
    $BENCH_BIN --demo --tokens 50 --output json > "$OUTPUT_DIR/simple_${TIMESTAMP}.json"
    echo -e "${GREEN}✅ Simple benchmark complete${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Benchmark complete!${NC}"
echo "Results saved to: $OUTPUT_DIR"

