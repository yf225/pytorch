#!/bin/bash

# Parse arguments
K_FILTER=""
while getopts "k:" opt; do
    case $opt in
        k)
            K_FILTER="$OPTARG"
            ;;
        \?)
            echo "Usage: $0 [-k <pytest filter>]"
            exit 1
            ;;
    esac
done

POD_NAME="willfeng-test"
LOCAL_PALLAS_FILE="torch/_inductor/codegen/pallas.py"
LOCAL_TEST_SCRIPT="tpu_run_tests_of_interest.sh"
REMOTE_BASE_PATH="/root/local/pytorch-nightly"
REMOTE_PALLAS_FILE="$REMOTE_BASE_PATH/torch/_inductor/codegen/pallas.py"
REMOTE_TEST_SCRIPT="$REMOTE_BASE_PATH/tpu_run_tests_of_interest.sh"
LOG_FILE="/tmp/tpu_tests.log"

echo "=============================================="
echo "Step 1: Copying files to $POD_NAME..."
echo "=============================================="
kubectl cp "$LOCAL_PALLAS_FILE" "$POD_NAME:$REMOTE_PALLAS_FILE"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy pallas.py to pod"
    exit 1
fi
echo "  - pallas.py"

kubectl cp "$LOCAL_TEST_SCRIPT" "$POD_NAME:$REMOTE_TEST_SCRIPT"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy test script to pod"
    exit 1
fi
echo "  - tpu_run_tests_of_interest.sh"

# Make the script executable
kubectl exec "$POD_NAME" -- chmod +x "$REMOTE_TEST_SCRIPT"
echo "Successfully copied files to $POD_NAME"
echo ""

echo "=============================================="
echo "Step 2: Running tests on $POD_NAME..."
echo "=============================================="
echo "Log file: $POD_NAME:$LOG_FILE"
if [ -n "$K_FILTER" ]; then
    echo "Filter: -k '$K_FILTER'"
fi
echo ""

# Build the command with optional -k filter
if [ -n "$K_FILTER" ]; then
    TEST_CMD="$REMOTE_TEST_SCRIPT -k '$K_FILTER'"
else
    TEST_CMD="$REMOTE_TEST_SCRIPT"
fi

# Run the script in background, pipe output to log file
kubectl exec "$POD_NAME" -- bash -c "$TEST_CMD > $LOG_FILE 2>&1 &"

echo "=============================================="
echo "Step 3: Tests are running in background!"
echo "=============================================="
echo ""
echo "To periodically check the log file, run:"
echo ""
echo "  kubectl exec $POD_NAME -- tail -100 $LOG_FILE"
echo ""
echo "To continuously watch the log file:"
echo ""
echo "  kubectl exec $POD_NAME -- tail -f $LOG_FILE"
echo ""
echo "To check if the test is still running:"
echo ""
echo "  kubectl exec $POD_NAME -- pgrep -f tpu_run_tests_of_interest.sh"
echo ""
