#!/bin/sh

module purge
source ~/.bashrc

ml cuda
ml nvhpc

mamba activate range-mace

# gcmc_mace.bash — Launch gRASPA with MACE ML-potential server
#
# Usage:  bash gcmc_mace.bash [model_path]
#
# Socket name is auto-generated; GRASPA_SOCKET_PATH is exported for gRASPA.

set -euo pipefail

MODEL="${1:-/storage/home/hcoda1/9/ltimmerman3/r-amedford6-0/potentials/MACE/mace-mpa-0-medium.model}"
# Generate a unique socket name using the Python function (instant one-liner)
SOCKET_NAME=$(python3 -c "import random, string; print('graspa_' + ''.join(random.choices(string.hexdigits.lower(), k=6)))")
SOCKET_PATH="/tmp/${SOCKET_NAME}"
export GRASPA_SOCKET_PATH="${SOCKET_PATH}"

echo "Socket: ${SOCKET_PATH}"

# Clean up any leftover fd from a previous run
rm -f "${SOCKET_PATH}"

echo "Starting MACE iPI server (model: ${MODEL})..."
python ase_ipi_server_mace.py \
    --socket "${SOCKET_NAME}" \
    --model "${MODEL}" \
    --dtype float32 \
    --device cuda &> server.log &
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
/storage/home/hcoda1/9/ltimmerman3/scratch/gRASPA-container/gRASPA_socket/patch_Socket/nvc_main.x &> graspa.out  # writes socket_species.txt during init, then connects

echo "gRASPA finished. Stopping server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

# Cleanup
rm -f "${SOCKET_PATH}"
echo "Done."
