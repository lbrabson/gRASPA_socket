#!/usr/bin/env bash
# gcmc_mace.bash — Launch gRASPA with MACE ML-potential server
#
# Usage:  bash gcmc_mace.bash [model_path]
#
# The server polls for socket_species.txt (written by gRASPA during init)
# before opening its listener, so no race condition.

set -euo pipefail

MODEL="${1:-/path/to/mace_model.pt}"
SOCKET_NAME="ase_ipi_socket"
SPECIES_FILE="socket_species.txt"

# Clean up from previous runs
rm -f "/tmp/ipi_${SOCKET_NAME}" "${SPECIES_FILE}"

echo "Starting MACE iPI server (model: ${MODEL})..."
python ase_ipi_server_mace.py \
    --socket "${SOCKET_NAME}" \
    --model "${MODEL}" \
    --species-file "${SPECIES_FILE}" \
    --device cpu &
SERVER_PID=$!

# Give the server a moment to start polling
sleep 1

echo "Starting gRASPA..."
./nvc_main.x   # writes socket_species.txt during init, then connects

echo "gRASPA finished. Stopping server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

# Cleanup
rm -f "/tmp/ipi_${SOCKET_NAME}" "${SPECIES_FILE}"
echo "Done."
