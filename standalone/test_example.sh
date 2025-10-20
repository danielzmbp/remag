#!/bin/bash
# Example testing script for HyenaDNA eukaryote predictor
# Uses the comm_16 test database for validation

set -e  # Exit on error

echo "HyenaDNA Eukaryote Predictor - Test Example"
echo "==========================================="
echo ""

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DB="${SCRIPT_DIR}/../test_db/comm_16"
CONTIGS_FILE="${TEST_DB}/contigs.fasta"

# Check if test database exists
if [[ ! -d "$TEST_DB" ]]; then
    echo "Error: Test database not found at $TEST_DB"
    exit 1
fi

if [[ ! -f "$CONTIGS_FILE" ]]; then
    echo "Error: Contigs file not found at $CONTIGS_FILE"
    exit 1
fi

echo "Test database: $TEST_DB"
echo "Input file: $CONTIGS_FILE"
echo ""

# Output files
OUTPUT_DIR="${SCRIPT_DIR}/test_results"
mkdir -p "$OUTPUT_DIR"

OUTPUT_FASTA="${OUTPUT_DIR}/eukaryotes.fasta"
OUTPUT_TABLE="${OUTPUT_DIR}/predictions.tsv"

echo "Output directory: $OUTPUT_DIR"
echo "Output FASTA: $OUTPUT_FASTA"
echo "Output table: $OUTPUT_TABLE"
echo ""

# Run prediction with default settings
echo "Running prediction with default settings..."
python "${SCRIPT_DIR}/predict_eukaryotes.py" \
    --input "$CONTIGS_FILE" \
    --output-fasta "$OUTPUT_FASTA" \
    --output-table "$OUTPUT_TABLE" \
    --device auto \
    --batch-size 64

echo ""
echo "==========================================="
echo "Test Results"
echo "==========================================="

# Show statistics
echo ""
echo "Predictions table (first 10 rows):"
head -11 "$OUTPUT_TABLE"

echo ""
echo "Summary statistics:"
total_predictions=$(tail -n +2 "$OUTPUT_TABLE" | wc -l)
eukaryotic=$(grep -c "eukaryote$" "$OUTPUT_TABLE" || true)
non_eukaryotic=$(grep -c "non_eukaryote$" "$OUTPUT_TABLE" || true)
fasta_count=$(grep -c "^>" "$OUTPUT_FASTA")

echo "  Total predictions: $total_predictions"
echo "  Eukaryotic: $eukaryotic"
echo "  Non-eukaryotic: $non_eukaryotic"
echo "  Filtered sequences (in FASTA): $fasta_count"

echo ""
echo "Filtered FASTA (first 10 sequences):"
head -30 "$OUTPUT_FASTA" | head -20

echo ""
echo "Test complete! Results saved to $OUTPUT_DIR"
