#!/usr/bin/env bash
# gcmc_odac.bash — Launch gRASPA with ODAC ML-potential server
#
# Usage:  bash gcmc_odac.bash [model_path]
#
# The server polls for socket_species.txt (written by gRASPA during init)
# before opening its listener, so no race condition.

#!/bin/sh
###################################################
#SBATCH -Jgcmc_odac
#SBATCH -A gts-amedford6-paid
#SBATCH -t12:00:00
#SBATCH --nodes=1 --gres=gpu:A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH -qinferno
#SBATCH -ogcmc_odac.out
###################################################

module load anaconda3
conda activate fairv2

MODEL="${1:-/storage/home/hcoda1/8/lbrabson3/r-amedford6-0/ODAC24/models/esen_sm_odac25_filtered.pt}"
SOCKET_NAME="ase_ipi_socket"
SPECIES_FILE="socket_species.txt"

SOCKET_PATH="/tmp/${SOCKET_NAME}"

# Clean up from previous runs
rm -f "${SOCKET_PATH}" "${SPECIES_FILE}"

echo "Starting ODAC iPI server (model: ${MODEL})..."
python ase_ipi_server_odac.py \
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
