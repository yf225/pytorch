#!/bin/bash
#
# TPU Pallas Test Runner - Tests of Interest (Quiet Mode)
#
# Usage:
#   ./tpu_run_tests_of_interest.sh                    # Run all tests
#   ./tpu_run_tests_of_interest.sh --cpu              # Run only CPU tests
#   ./tpu_run_tests_of_interest.sh --tpu              # Run only TPU tests
#   ./tpu_run_tests_of_interest.sh -k "pattern"       # Run tests matching pattern
#   ./tpu_run_tests_of_interest.sh --tpu -k "llama"   # Run TPU tests matching "llama"
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

TESTS_TPU=(
    "PallasTestsTPU::test_different_shapes"
    "PallasTestsTPU::test_llama3"
    "PallasTestsTPU::test_max_reduction"
    "PallasTestsTPU::test_min_reduction"
    "PallasTestsTPU::test_nanogpt"
    "PallasTestsTPU::test_prod_reduction"
    "PallasTestsTPU::test_sum_reduction"
    "PallasTestsTPU::test_arange_multi_output"
    "PallasTestsTPU::test_strided_int_pallas"
    "PallasTestsTPU::test_strided_offset_pallas"
    "PallasTestsTPU::test_strided_2d_pallas"
    "PallasTestsTPU::test_stride_non_contiguous_1d"
    "PallasTestsTPU::test_stride_non_contiguous_2d_row_stride"
    "PallasTestsTPU::test_stride_non_contiguous_2d_col_stride"
    "PallasTestsTPU::test_stride_non_contiguous_2d_both_stride"
    "PallasTestsTPU::test_stride_non_contiguous_2d_transpose"
    "PallasTestsTPU::test_stride_non_contiguous_3d"
    "PallasTestsTPU::test_stride_non_contiguous_permuted"
    "PallasTestsTPU::test_stride_non_contiguous_channels_last"
    "PallasTestsTPU::test_stride_non_contiguous_diagonal"
    "PallasTestsTPU::test_stride_non_contiguous_as_strided"
    "PallasTestsTPU::test_stride_non_contiguous_select_stride"
    "PallasTestsTPU::test_stride_non_contiguous_unsqueeze"
    "PallasTestsTPU::test_stride_non_contiguous_dtypes"
    "PallasTestsTPU::test_stride_expanded_tensors"
    "PallasTestsTPU::test_stride_multiple_inputs"
    "PallasTestsTPU::test_contiguous_index_validation"
    "PallasTestsTPU::test_complex_indexing_gather"
    "PallasTestsTPU::test_complex_indexing_2d"
)

# Parse arguments
RUN_CPU=false
RUN_TPU=false
K_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            RUN_CPU=true
            shift
            ;;
        --tpu)
            RUN_TPU=true
            shift
            ;;
        -k)
            K_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--cpu] [--tpu] [-k <filter>]"
            exit 1
            ;;
    esac
done

# Default to both if neither specified
if [[ "$RUN_CPU" == false && "$RUN_TPU" == false ]]; then
    RUN_CPU=true
    RUN_TPU=true
fi

# Build test list
TESTS_TO_RUN=()
[[ "$RUN_CPU" == true ]] && TESTS_TO_RUN+=("${TESTS_CPU[@]}")
[[ "$RUN_TPU" == true ]] && TESTS_TO_RUN+=("${TESTS_TPU[@]}")

# Apply -k filter if specified
if [[ -n "$K_FILTER" ]]; then
    FILTERED_TESTS=()
    for TEST in "${TESTS_TO_RUN[@]}"; do
        if [[ "$TEST" == *"$K_FILTER"* ]]; then
            FILTERED_TESTS+=("$TEST")
        fi
    done
    TESTS_TO_RUN=("${FILTERED_TESTS[@]}")

    if [[ ${#TESTS_TO_RUN[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No tests match filter: ${K_FILTER}${NC}"
        exit 0
    fi
fi

# Environment setup (quiet)
export PALLAS_TARGET_TPU=1
export PJRT_DEVICE=TPU
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export ENABLE_AOT_AUTOGRAD_CACHE=0

# Quick TPU check
if ! python -c "import jax; assert len(jax.devices('tpu')) > 0" 2>/dev/null; then
    echo -e "${RED}ERROR: No TPU devices found${NC}"
    exit 1
fi

# Kill stale processes (quiet)
pkill -9 -f "python.*test_pallas" 2>/dev/null || true
pkill -9 -f "libtpu" 2>/dev/null || true
sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true
sleep 1

# Clear caches (quiet)
rm -rf "${HOME}/.cache/torch/inductor" 2>/dev/null || true
rm -rf "${HOME}/.cache/torch/_dynamo" 2>/dev/null || true
rm -rf "${HOME}/.cache/jax" 2>/dev/null || true
python -c "import torch._dynamo; torch._dynamo.reset()" 2>/dev/null || true

cd "${SCRIPT_DIR}"

# Run tests
PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=()
TOTAL=${#TESTS_TO_RUN[@]}

echo "Running ${TOTAL} tests..."
if [[ -n "$K_FILTER" ]]; then
    echo "Filter: -k '$K_FILTER'"
fi
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
