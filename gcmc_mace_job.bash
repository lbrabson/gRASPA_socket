#!/bin/sh
###################################################
#SBATCH -Jgcmc
#SBATCH -A gts-amedford6-paid
#SBATCH -t5:00:00
#SBATCH --nodes=1 --gres=gpu:A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH -qinferno
#SBATCH -ogcmc_mace.out
###################################################

module purge
module load anaconda3
module load nvhpc
conda activate mlipmc_mace

# gcmc_mace.bash — Launch gRASPA with MACE ML-potential server
#
# Usage:  bash gcmc_mace.bash [model_path]
#
# Socket name is auto-generated; GRASPA_SOCKET_PATH is exported for gRASPA.


MODEL="${1:-/storage/home/hcoda1/8/lbrabson3/r-amedford6-0/mace-mpa-0-medium.model}"
# Generate a unique socket name (inline — avoids module import path issues)
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
/storage/home/hcoda1/8/lbrabson3/r-amedford6-0/gRASPA_dir/gRASPA_socket-manybody-ewald/patch_Socket/nvc_main.x > raspa.out 2>&1

echo "gRASPA finished. Stopping server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

# Cleanup
rm -f "${SOCKET_PATH}"
echo "Done."