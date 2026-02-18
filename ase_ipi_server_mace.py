#!/usr/bin/env python3
"""
Custom iPI-protocol server for gRASPA Monte Carlo.

Reads a species file written by gRASPA to know the chemical identity of atoms
in each configuration (total, framework-only, adsorbate-only).  Implements the
iPI wire protocol directly so that varying natoms between calls is handled
correctly.

Usage:
    python ase_ipi_server_mace.py \
        --socket ase_ipi_socket \
        --species-file socket_species.txt \
        --model /path/to/model.pt
"""

import argparse
import os
import socket
import struct
import sys
import time

import numpy as np
from ase import Atoms


# ---------------------------------------------------------------------------
# Calculator factory — edit this function to swap models
# ---------------------------------------------------------------------------
def get_calculator(args):
    """Return an ASE calculator.  Modify this to use CHGNet, etc."""
    from mace.calculators import mace_mp

    calc = mace_mp(
        model=args.model,
        device=args.device,
        default_dtype=args.dtype,
    )
    print(f"Loaded MACE model: {args.model}  device={args.device}")
    return calc


# ---------------------------------------------------------------------------
# Species-file parser
# ---------------------------------------------------------------------------
def parse_species_file(path):
    """Return (fw_symbols, ads_symbols) lists of element strings."""
    fw_symbols = []
    ads_symbols = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "FRAMEWORK":
                n_fw = int(parts[1])
                fw_symbols = next(f).split()
                assert len(fw_symbols) == n_fw, (
                    f"Expected {n_fw} framework symbols, got {len(fw_symbols)}"
                )
            elif parts[0] == "ADSORBATE":
                n_ads = int(parts[1])
                ads_symbols = next(f).split()
                assert len(ads_symbols) == n_ads, (
                    f"Expected {n_ads} adsorbate symbols, got {len(ads_symbols)}"
                )
    return fw_symbols, ads_symbols


def wait_for_species_file(path, poll=0.5, timeout=300):
    """Block until the species file exists and is non-empty."""
    print(f"Waiting for species file: {path}")
    t0 = time.time()
    while True:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            fw, ads = parse_species_file(path)
            print(f"  framework : {len(fw)} atoms  (e.g. {' '.join(fw[:5])} ...)")
            print(f"  adsorbate : {len(ads)} atoms  (e.g. {' '.join(ads[:5])} ...)")
            return fw, ads
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Species file {path} not found after {timeout}s")
        time.sleep(poll)


# ---------------------------------------------------------------------------
# Low-level iPI helpers (network byte order)
# ---------------------------------------------------------------------------
def send_header(conn, msg):
    hdr = msg.encode("ascii").ljust(12)[:12]
    conn.sendall(hdr)


def recv_header(conn):
    data = _recvall(conn, 12)
    return data.decode("ascii").strip()


def send_int32(conn, val):
    conn.sendall(struct.pack("!i", val))


def recv_int32(conn):
    return struct.unpack("!i", _recvall(conn, 4))[0]


def send_doubles(conn, arr):
    conn.sendall(arr.astype(np.float64).tobytes())


def recv_doubles(conn, n):
    return np.frombuffer(_recvall(conn, 8 * n), dtype=np.float64).copy()


def _recvall(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# iPI server loop
# ---------------------------------------------------------------------------
def do_handshake(conn, cell_hint):
    """Perform the iPI handshake on an accepted connection."""
    send_header(conn, "STATUS")
    hdr = recv_header(conn)
    assert hdr == "NEEDINIT", f"Expected NEEDINIT, got {hdr}"

    send_header(conn, "INIT")
    send_doubles(conn, cell_hint.flatten())               # 9 doubles
    send_doubles(conn, np.linalg.inv(cell_hint).flatten()) # 9 doubles
    print("Handshake complete")


def serve(conn, calc, fw_symbols, ads_symbols, cell_hint):
    """Run the iPI protocol main loop (handshake must be done already)."""

    n_fw = len(fw_symbols)
    n_ads = len(ads_symbols)
    n_total = n_fw + n_ads
    total_symbols = fw_symbols + ads_symbols

    symbol_map = {
        n_total: total_symbols,
        n_fw: fw_symbols,
        n_ads: ads_symbols,
    }
    print(f"Expecting natoms in {{{n_total}, {n_fw}, {n_ads}}}")

    # ---- Main loop ----
    step = 0
    while True:
        # STATUS -> expect READY
        send_header(conn, "STATUS")
        hdr = recv_header(conn)
        if hdr == "EXIT" or not hdr:
            print("Client sent EXIT — shutting down")
            break
        if hdr != "READY":
            print(f"Warning: expected READY, got '{hdr}'")
            break

        # POSDATA
        send_header(conn, "POSDATA")

        # Receive cell (9), inv_cell (9), natoms, positions
        cell = recv_doubles(conn, 9).reshape(3, 3)
        _inv = recv_doubles(conn, 9)  # discard client inv_cell
        natoms = recv_int32(conn)
        positions = recv_doubles(conn, 3 * natoms).reshape(natoms, 3)

        # Identify species from natoms
        if natoms not in symbol_map:
            print(f"ERROR: unexpected natoms={natoms}, expected one of {list(symbol_map.keys())}")
            break
        symbols = symbol_map[natoms]
        label = {n_total: "total", n_fw: "framework", n_ads: "adsorbate"}.get(natoms, "?")

        # Build ASE Atoms and compute
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        step += 1
        print(f"  step {step:4d}  {label:10s}  natoms={natoms:5d}  E={energy:14.6f} eV")

        # STATUS -> expect HAVEDATA
        send_header(conn, "STATUS")
        hdr = recv_header(conn)
        if hdr != "HAVEDATA":
            print(f"Warning: expected HAVEDATA, got '{hdr}'")
            break

        # GETFORCE
        send_header(conn, "GETFORCE")

        # FORCEREADY + energy + natoms + forces + virial + extras
        send_header(conn, "FORCEREADY")
        send_doubles(conn, np.array([energy]))       # 1 double
        send_int32(conn, natoms)                      # int32
        send_doubles(conn, forces.flatten())          # 3*natoms doubles
        send_doubles(conn, np.zeros(9))               # virial (9 doubles)
        send_int32(conn, 0)                           # extras length


def main():
    parser = argparse.ArgumentParser(
        description="Custom iPI server for gRASPA + ML potentials"
    )
    parser.add_argument("--socket", "-s", default="ase_ipi_socket",
                        help="UNIX socket name (creates /tmp/ipi_<name>)")
    parser.add_argument("--species-file", default="socket_species.txt",
                        help="Path to species file written by gRASPA")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to MACE model file (.pt)")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cuda", "cpu"],
                        help="Device for calculator (default: cpu)")
    parser.add_argument("--dtype", default="float64",
                        choices=["float32", "float64"],
                        help="Floating point precision (default: float64)")
    args = parser.parse_args()

    print("=" * 60)
    print("gRASPA iPI Server (custom protocol)")
    print("=" * 60)

    # 1. Create listener FIRST so gRASPA can connect while the model loads.
    #    The OS will queue the incoming connection (backlog=1) until we accept().
    sock_path = f"/tmp/{args.socket}"
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    print(f"Listening on {sock_path} (model loading may take a moment...)")

    # 2. Load calculator (may be slow on GPU; socket is already ready)
    calc = get_calculator(args)

    # Dummy cell for INIT (client sends real cell with POSDATA)
    cell_hint = np.eye(3) * 10.0

    fw_symbols = None
    ads_symbols = None

    try:
        while True:
            print("Waiting for client connection...")
            conn, _ = srv.accept()
            print("Client connected")

            # 3. Handshake first (no species info needed yet)
            do_handshake(conn, cell_hint)

            # 4. Wait for species file on first connection
            #    (gRASPA writes it after connecting + populating ReplicaAtoms)
            if fw_symbols is None:
                fw_symbols, ads_symbols = wait_for_species_file(args.species_file)

            try:
                serve(conn, calc, fw_symbols, ads_symbols, cell_hint)
            except (ConnectionError, BrokenPipeError) as e:
                print(f"Connection lost: {e}")
            finally:
                conn.close()
            print("Client disconnected — waiting for reconnect...")
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)


if __name__ == "__main__":
    main()
