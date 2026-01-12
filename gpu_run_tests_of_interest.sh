#!/bin/bash
#
# Pallas Test Runner - Tests of Interest (Quiet Mode)
#
# Usage:
#   ./tpu_run_tests_of_interest.sh           # Run all tests
#   ./tpu_run_tests_of_interest.sh --cpu     # Run only CPU tests
#   ./tpu_run_tests_of_interest.sh --cuda    # Run only CUDA tests
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================
# Tests of Interest
# ============================================================

TESTS_CPU=(
    "PallasTestsCPU::test_different_shapes"
    "PallasTestsCPU::test_llama3"
    "PallasTestsCPU::test_max_reduction"
    "PallasTestsCPU::test_min_reduction"
    "PallasTestsCPU::test_nanogpt"
    "PallasTestsCPU::test_prod_reduction"
    "PallasTestsCPU::test_sum_reduction"
    "PallasTestsCPU::test_arange_multi_output"
    "PallasTestsCPU::test_strided_int_pallas"
    "PallasTestsCPU::test_strided_offset_pallas"
    "PallasTestsCPU::test_strided_2d_pallas"
    "PallasTestsCPU::test_stride_non_contiguous_1d"
    "PallasTestsCPU::test_stride_non_contiguous_2d_row_stride"
    "PallasTestsCPU::test_stride_non_contiguous_2d_col_stride"
    "PallasTestsCPU::test_stride_non_contiguous_2d_both_stride"
    "PallasTestsCPU::test_stride_non_contiguous_2d_transpose"
    "PallasTestsCPU::test_stride_non_contiguous_3d"
    "PallasTestsCPU::test_stride_non_contiguous_permuted"
    "PallasTestsCPU::test_stride_non_contiguous_channels_last"
    "PallasTestsCPU::test_stride_non_contiguous_diagonal"
    "PallasTestsCPU::test_stride_non_contiguous_as_strided"
    "PallasTestsCPU::test_stride_non_contiguous_select_stride"
    "PallasTestsCPU::test_stride_non_contiguous_unsqueeze"
    "PallasTestsCPU::test_stride_non_contiguous_dtypes"
    "PallasTestsCPU::test_stride_expanded_tensors"
    "PallasTestsCPU::test_stride_multiple_inputs"
    "PallasTestsCPU::test_contiguous_index_validation"
    "PallasTestsCPU::test_complex_indexing_gather"
    "PallasTestsCPU::test_complex_indexing_2d"
)

TESTS_CUDA=(
    "PallasTestsCUDA::test_different_shapes"
    "PallasTestsCUDA::test_llama3"
    "PallasTestsCUDA::test_max_reduction"
    "PallasTestsCUDA::test_min_reduction"
    "PallasTestsCUDA::test_nanogpt"
    "PallasTestsCUDA::test_prod_reduction"
    "PallasTestsCUDA::test_sum_reduction"
    "PallasTestsCUDA::test_arange_multi_output"
    "PallasTestsCUDA::test_strided_int_pallas"
    "PallasTestsCUDA::test_strided_offset_pallas"
    "PallasTestsCUDA::test_strided_2d_pallas"
    "PallasTestsCUDA::test_stride_non_contiguous_1d"
    "PallasTestsCUDA::test_stride_non_contiguous_2d_row_stride"
    "PallasTestsCUDA::test_stride_non_contiguous_2d_col_stride"
    "PallasTestsCUDA::test_stride_non_contiguous_2d_both_stride"
    "PallasTestsCUDA::test_stride_non_contiguous_2d_transpose"
    "PallasTestsCUDA::test_stride_non_contiguous_3d"
    "PallasTestsCUDA::test_stride_non_contiguous_permuted"
    "PallasTestsCUDA::test_stride_non_contiguous_channels_last"
    "PallasTestsCUDA::test_stride_non_contiguous_diagonal"
    "PallasTestsCUDA::test_stride_non_contiguous_as_strided"
    "PallasTestsCUDA::test_stride_non_contiguous_select_stride"
    "PallasTestsCUDA::test_stride_non_contiguous_unsqueeze"
    "PallasTestsCUDA::test_stride_non_contiguous_dtypes"
    "PallasTestsCUDA::test_stride_expanded_tensors"
    "PallasTestsCUDA::test_stride_multiple_inputs"
    "PallasTestsCUDA::test_contiguous_index_validation"
    "PallasTestsCUDA::test_complex_indexing_gather"
    "PallasTestsCUDA::test_complex_indexing_2d"
)

# Parse arguments
RUN_CPU=false
RUN_CUDA=false

if [[ $# -eq 0 ]]; then
    RUN_CPU=true
    RUN_CUDA=true
else
    for arg in "$@"; do
        case $arg in
            --cpu) RUN_CPU=true ;;
            --cuda) RUN_CUDA=true ;;
            *) echo "Usage: $0 [--cpu] [--cuda]"; exit 1 ;;
        esac
    done
fi

# Build test list
TESTS_TO_RUN=()
[[ "$RUN_CPU" == true ]] && TESTS_TO_RUN+=("${TESTS_CPU[@]}")
[[ "$RUN_CUDA" == true ]] && TESTS_TO_RUN+=("${TESTS_CUDA[@]}")

# Environment setup
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export ENABLE_AOT_AUTOGRAD_CACHE=0

# Clear caches (quiet)
rm -rf "${HOME}/.cache/torch/inductor" 2>/dev/null || true
rm -rf "${HOME}/.cache/torch/_dynamo" 2>/dev/null || true
python -c "import torch._dynamo; torch._dynamo.reset()" 2>/dev/null || true

cd "${SCRIPT_DIR}"

# Run tests
PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=()
TOTAL=${#TESTS_TO_RUN[@]}

echo "Running ${TOTAL} tests..."
echo ""

for i in "${!TESTS_TO_RUN[@]}"; do
    TEST="${TESTS_TO_RUN[$i]}"
    NUM=$((i + 1))

    # Run test quietly, capture exit code
    if python -m pytest "test/inductor/test_pallas.py::${TEST}" -x -q 2>&1 >/dev/null; then
        echo -e "[${NUM}/${TOTAL}] ${GREEN}PASS${NC} ${TEST}"
        PASSED=$((PASSED + 1))
    else
        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 5 ]]; then
            echo -e "[${NUM}/${TOTAL}] ${YELLOW}SKIP${NC} ${TEST}"
            SKIPPED=$((SKIPPED + 1))
        else
            echo -e "[${NUM}/${TOTAL}] ${RED}FAIL${NC} ${TEST}"
            FAILED=$((FAILED + 1))
            FAILED_TESTS+=("${TEST}")
        fi
    fi
done

# Summary
echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASSED} passed${NC}, ${RED}${FAILED} failed${NC}, ${YELLOW}${SKIPPED} skipped${NC}"
echo "========================================"

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed tests:"
    for TEST in "${FAILED_TESTS[@]}"; do
        echo "  - ${TEST}"
    done
    exit 1
fi
