#!/usr/bin/env bash
# gcmc_mace.bash — Launch gRASPA with MACE ML-potential server
#
# Usage:  bash gcmc_mace.bash [model_path]
#
# The server polls for socket_species.txt (written by gRASPA during init)
# before opening its listener, so no race condition.

set -euo pipefail

MODEL="${1:-/storage/home/hcoda1/8/lbrabson3/r-amedford6-0/mace-mpa-0-medium.model}"
SOCKET_NAME="ase_ipi_socket"
SPECIES_FILE="socket_species.txt"

SOCKET_PATH="/tmp/${SOCKET_NAME}"
export GRASPA_SOCKET_PATH="${SOCKET_PATH}"

# Clean up from previous runs
rm -f "${SOCKET_PATH}" "${SPECIES_FILE}"

echo "Starting MACE iPI server (model: ${MODEL})..."
python ase_ipi_server_mace.py \
    --socket "${SOCKET_NAME}" \
    --model "${MODEL}" \
    --species-file "${SPECIES_FILE}" \
    --device cuda &
SERVER_PID=$!

# Wait until the server has created the socket file before launching gRASPA.
# Model loading (especially on GPU) can take much longer than a fixed sleep.
echo "Waiting for server socket: ${SOCKET_PATH} ..."
WAIT=0
until [ -S "${SOCKET_PATH}" ] || [ "${WAIT}" -ge 300 ]; do
    sleep 1
    WAIT=$((WAIT + 1))
done
if [ ! -S "${SOCKET_PATH}" ]; then
    echo "ERROR: server socket did not appear after ${WAIT}s — aborting"
    kill "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi
echo "Server socket ready after ${WAIT}s."

echo "Starting gRASPA..."
./patch_Socket/nvc_main.x   # writes socket_species.txt during init, then connects

echo "gRASPA finished. Stopping server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

# Cleanup
rm -f "${SOCKET_PATH}" "${SPECIES_FILE}"
echo "Done."
