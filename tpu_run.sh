#!/bin/bash
#
# TPU Pallas Test Runner
#
# This script replicates the PyTorch CI environment for running test_pallas.py on TPU.
#
# CI Workflow Reference: .github/workflows/inductor-pallas.yml
# Build: linux-jammy-py3_12-inductor-pallas-tpu-build
# Test Config: inductor-pallas-tpu
# Runner: linux.google.tpuv7x.1
#
# What this script does:
# 1. Sets environment variables to target TPU (PALLAS_TARGET_TPU=1, PJRT_DEVICE=TPU)
# 2. Verifies JAX can detect TPU devices
# 3. Runs test_pallas.py using the same command as CI
#
# Usage:
#   ./tpu_run.sh                    # Run all tests
#   ./tpu_run.sh -k test_simple_add # Run specific test
#   ./tpu_run.sh --help             # Show pytest help
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# Environment Setup (matches CI)
# ============================================================

# Set TPU target - this is set in .ci/pytorch/test.sh test_inductor_pallas()
# when TEST_CONFIG contains "inductor-pallas-tpu"
export PALLAS_TARGET_TPU=1

# PJRT_DEVICE=TPU is set in the CI docker run command via TPU_DOCKER_FLAGS
# (see .github/workflows/_linux-test.yml step "Setup TPU docker flags")
export PJRT_DEVICE=TPU

echo_info "Environment variables set:"
echo "  PALLAS_TARGET_TPU=${PALLAS_TARGET_TPU}"
echo "  PJRT_DEVICE=${PJRT_DEVICE}"

# ============================================================
# TPU Detection (matches CI verification in test.sh)
# ============================================================

echo_info "Checking TPU availability..."

# This check mirrors the one in .ci/pytorch/test.sh test_inductor_pallas()
if ! python -c "import jax; devices = jax.devices('tpu'); print(f'Found {len(devices)} TPU device(s)'); assert len(devices) > 0, 'No TPU devices found'" 2>/dev/null; then
    echo_error "No TPU devices found!"
    echo ""
    echo "This script is designed to run on TPU hardware. Make sure:"
    echo "  1. You are running on a TPU VM (e.g., linux.google.tpuv7x.1)"
    echo "  2. JAX with TPU support is installed: pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    echo "  3. The TPU runtime is properly configured"
    echo ""
    echo "TPU environment variables that may need to be set (from CI):"
    echo "  TPU_SKIP_MDS_QUERY, TPU_TOPOLOGY, TPU_WORKER_ID, TPU_TOPOLOGY_WRAP,"
    echo "  TPU_CHIPS_PER_HOST_BOUNDS, TPU_ACCELERATOR_TYPE, TPU_RUNTIME_METRICS_PORTS,"
    echo "  TPU_TOPOLOGY_ALT, HOST_BOUNDS, TPU_HOST_BOUNDS, VBAR_CONTROL_SERVICE_URL,"
    echo "  CHIPS_PER_HOST_BOUNDS, TPU_WORKER_HOSTNAMES"
    echo ""
    echo "To check if the TPU is detected:"
    echo "  python -c \"import jax; print(jax.devices('tpu'))\""
    exit 1
fi

echo_info "TPU verification passed!"

# ============================================================
# Display JAX/Pallas Version Info
# ============================================================

echo_info "JAX/Pallas version info:"
python -c "
import jax
print(f'  JAX version: {jax.__version__}')
try:
    from jax.experimental import pallas as pl
    print(f'  Pallas available: Yes')
except ImportError:
    print(f'  Pallas available: No')
devices = jax.devices()
print(f'  JAX devices: {devices}')
"

# ============================================================
# Kill Stale TPU Processes
# ============================================================

echo ""
echo_info "Killing stale TPU processes..."

# Kill any Python processes that might be holding TPU resources
# This helps avoid "TPU already in use" errors from previous runs
pkill -9 -f "python.*test_pallas" 2>/dev/null || true
pkill -9 -f "python.*jax" 2>/dev/null || true

# Kill any libtpu processes that might be stuck
pkill -9 -f "libtpu" 2>/dev/null || true

# Remove stale libtpu lockfile (prevents "Internal error when accessing libtpu multi-process lockfile")
sudo rm -f /tmp/libtpu_lockfile 2>/dev/null || true

# Give processes time to clean up
sleep 2

echo_info "Stale processes killed (if any)"

# ============================================================
# Clear torch.compile / Inductor Cache
# ============================================================

echo ""
echo_info "Clearing torch.compile and Inductor caches..."

# Clear PyTorch Inductor cache
# Default cache location: ~/.cache/torch/inductor or TORCHINDUCTOR_CACHE_DIR
INDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torch/inductor}"
if [[ -d "${INDUCTOR_CACHE_DIR}" ]]; then
    rm -rf "${INDUCTOR_CACHE_DIR}"
    echo "  Cleared: ${INDUCTOR_CACHE_DIR}"
else
    echo "  No inductor cache found at: ${INDUCTOR_CACHE_DIR}"
fi

# Clear PyTorch dynamo cache
DYNAMO_CACHE_DIR="$HOME/.cache/torch/_dynamo"
if [[ -d "${DYNAMO_CACHE_DIR}" ]]; then
    rm -rf "${DYNAMO_CACHE_DIR}"
    echo "  Cleared: ${DYNAMO_CACHE_DIR}"
else
    echo "  No dynamo cache found at: ${DYNAMO_CACHE_DIR}"
fi

# Clear JAX compilation cache (if exists)
JAX_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax}"
if [[ -d "${JAX_CACHE_DIR}" ]]; then
    rm -rf "${JAX_CACHE_DIR}"
    echo "  Cleared: ${JAX_CACHE_DIR}"
else
    echo "  No JAX cache found at: ${JAX_CACHE_DIR}"
fi

# Also clear via Python to reset any in-memory state
python -c "
import torch
import torch._dynamo
import gc

# Reset dynamo state
torch._dynamo.reset()

# Clear any cached compilation artifacts
gc.collect()

print('  Python caches cleared (torch._dynamo.reset())')
" 2>/dev/null || echo_warn "Could not reset Python caches"

echo_info "Cache clearing complete"

# ============================================================
# Run Tests (matches CI command in .ci/pytorch/test.sh)
# ============================================================

echo ""
echo_info "Running test_pallas.py on TPU..."
echo ""

# Disable caches to ensure fresh compilation each run
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export ENABLE_AOT_AUTOGRAD_CACHE=0

echo_info "Cache-disabling environment variables set:"
echo "  TORCHINDUCTOR_FORCE_DISABLE_CACHES=${TORCHINDUCTOR_FORCE_DISABLE_CACHES}"
echo "  ENABLE_AOT_AUTOGRAD_CACHE=${ENABLE_AOT_AUTOGRAD_CACHE}"
echo ""

# CI uses: python test/run_test.py --include inductor/test_pallas.py --verbose
# For local development, we can also run pytest directly with additional args

cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
    # Run via pytest directly (run_test.py always adds -x internally which stops at first failure)
    python -m pytest test/inductor/test_pallas.py -v
else
    # With args: pass to pytest for flexibility (e.g., -k for specific tests)
    python -m pytest test/inductor/test_pallas.py -v "$@"
fi
