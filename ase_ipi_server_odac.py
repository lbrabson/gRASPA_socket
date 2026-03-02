#!/usr/bin/env python3
"""
Custom iPI-protocol server for gRASPA Monte Carlo.

Receives a species map from gRASPA at connection time (replaces the old
static species file).  Each POSDATA message now carries per-atom type
indices and an n_mol field so the server can handle batched FXNMAIN calls
(all adsorbate molecules in one call) as well as single-molecule MC moves.

Wire protocol additions (C++ → Python):
  After handshake, before the STATUS loop:
    "SPECIESMAP" header (12-byte padded)
    int32  n_species
    int32  n_fw            atom count in ReplicaAtoms[0]
    for each species:
      int32 index
      int32 len
      len bytes symbol

  Each POSDATA payload now contains:
    int32   config_type    routing tag (0=fw, N=ads-only, 100+N=combined)
    9×f64   cell
    9×f64   inv_cell       (ignored, server uses PBC via cell)
    int32   n_mol          adsorbate molecule count
    int32   natoms
    N×int32 types          species index per atom
    3N×f64  xyz

  For config_type=100+N with n_mol>1 (FXNMAIN batch):
    Server loops per molecule, returns Σ_i E_total(fw + mol_i).

Usage:
    python ase_ipi_server_odac.py \
        --socket ase_ipi_socket \
        --model /path/to/model.pt
"""

import argparse
import os
import socket
import struct
import sys

import numpy as np
from ase import Atoms


# ---------------------------------------------------------------------------
# Calculator factory — edit this function to swap models
# ---------------------------------------------------------------------------
def get_calculator(args):
    """Return an ASE calculator.  Modify this to use CHGNet, etc."""
    from fairchem.core import FAIRChemCalculator

    calc = FAIRChemCalculator.from_model_checkpoint(
        name_or_path = args.model, 
        device = args.device, 
        task_name = 'odac', 
        seed=41
        )  

    print(f"Loaded ODAC model: {args.model}  device={args.device}")
    return calc


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
    send_doubles(conn, cell_hint.flatten())                # 9 doubles
    send_doubles(conn, np.linalg.inv(cell_hint).flatten()) # 9 doubles
    print("Handshake complete")


def recv_species_map(conn):
    """Receive the species map sent by gRASPA immediately after handshake.

    Returns:
        species_map : dict  {int index -> str symbol}
        n_fw        : int   framework atom count in ReplicaAtoms[0]
    """
    hdr = recv_header(conn)
    assert hdr == "SPECIESMAP", f"Expected SPECIESMAP, got '{hdr}'"

    n_species = recv_int32(conn)
    n_fw      = recv_int32(conn)

    species_map = {}
    for _ in range(n_species):
        idx = recv_int32(conn)
        ln  = recv_int32(conn)
        sym = _recvall(conn, ln).decode("ascii")
        species_map[idx] = sym

    print(f"Received species map: {n_species} species, n_fw={n_fw}")
    for k, v in sorted(species_map.items()):
        print(f"  {k}: {v}")

    return species_map, n_fw


def serve(conn, calc, species_map, n_fw):
    """Run the iPI protocol main loop (handshake + species map already done).

    Server-side HG decomposition:
      config_type=0          : startup prime — compute and cache E_fw
      config_type>=100       : combined fw+ads call — return HG directly:
                               HG = E(fw+ads) - cached_E_fw - E(ads alone)

    species_map : {int -> str}  mapping from type index to element symbol
    n_fw        : int           number of framework atoms in combined payloads
    """
    step = 0
    cached_E_fw = None   # primed by config_type=0 at startup

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

        # Receive routing fields
        config_type = recv_int32(conn)
        cell        = recv_doubles(conn, 9).reshape(3, 3)
        _inv        = recv_doubles(conn, 9)   # discard client inv_cell
        n_mol       = recv_int32(conn)
        natoms      = recv_int32(conn)

        # Types array (N × int32, network byte order)
        types_raw = _recvall(conn, 4 * natoms)
        types_arr = struct.unpack(f"!{natoms}i", types_raw)

        # Positions
        positions = recv_doubles(conn, 3 * natoms).reshape(natoms, 3)

        # Build element symbols from per-atom type indices
        try:
            symbols = [species_map[t] for t in types_arr]
        except KeyError as e:
            print(f"ERROR: type index {e} not in species map {species_map}")
            break

        # Human-readable label for logging
        if config_type == 0:
            label = "framework"
        elif config_type < 100:
            label = f"adsorbate[{config_type}]"
        else:
            label = f"total[{config_type - 100}] n_mol={n_mol}"

        # ---- Energy evaluation ----
        if config_type == 0:
            # Startup cache-priming call: framework atoms only.
            # Compute E_fw and cache it; return it to C++ for logging.
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            atoms.calc = calc
            cached_E_fw = atoms.get_potential_energy()
            energy = cached_E_fw
            forces = atoms.get_forces()
            print(f"  E_fw cached: {cached_E_fw:.6f} eV")

        elif config_type >= 100:
            # Combined fw + adsorbates call (n_mol >= 1).
            # Perform two ML evaluations and return HG directly.
            if cached_E_fw is None:
                raise RuntimeError(
                    "Framework cache not primed — send config_type=0 first")

            # E(fw + all ads together)
            atoms_combined = Atoms(
                symbols=symbols, positions=positions, cell=cell, pbc=True)
            atoms_combined.calc = calc
            E_combined = atoms_combined.get_potential_energy()

            # E(ads alone) — strip framework atoms
            ads_positions = positions[n_fw:]
            ads_symbols   = symbols[n_fw:]
            atoms_ads = Atoms(
                symbols=ads_symbols, positions=ads_positions, cell=cell, pbc=True)
            atoms_ads.calc = calc
            E_ads = atoms_ads.get_potential_energy()

            # HG = E(fw+ads) − E_fw − E(ads)
            energy = E_combined - cached_E_fw - E_ads
            forces = np.zeros((natoms, 3))   # forces not used by gRASPA
            print(f"  E_combined={E_combined:.6f}  E_fw={cached_E_fw:.6f}  "
                  f"E_ads={E_ads:.6f}  HG={energy:.6f} eV")

        else:
            # Fallback: standalone adsorbate-only or unrecognised config (not
            # used in normal flow but kept for completeness).
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()

        step += 1
        print(f"  step {step:4d}  {label:45s}  config_type={config_type:4d}  "
              f"natoms={natoms:5d}  n_mol={n_mol}  E={energy:14.6f} eV")

        # STATUS -> expect HAVEDATA
        send_header(conn, "STATUS")
        hdr = recv_header(conn)
        if hdr != "HAVEDATA":
            print(f"Warning: expected HAVEDATA, got '{hdr}'")
            break

        # GETFORCE → FORCEREADY + energy + natoms + forces + virial + extras
        send_header(conn, "GETFORCE")
        send_header(conn, "FORCEREADY")
        send_doubles(conn, np.array([energy]))    # 1 double
        send_int32(conn, natoms)                   # int32
        send_doubles(conn, forces.flatten())       # 3*natoms doubles
        send_doubles(conn, np.zeros(9))            # virial (9 doubles)
        send_int32(conn, 0)                        # extras length


def main():
    parser = argparse.ArgumentParser(
        description="Custom iPI server for gRASPA + ML potentials"
    )
    parser.add_argument("--socket", "-s", default="ase_ipi_socket",
                        help="UNIX socket name (creates /tmp/<name>)")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to ODAC model file (.pt)")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cuda", "cpu"],
                        help="Device for calculator (default: cpu)")
    parser.add_argument("--dtype", default="float64",
                        choices=["float32", "float64"],
                        help="Floating point precision (default: float64)")
    # --species-file kept for backwards-compatible CLI; no longer used
    parser.add_argument("--species-file", default=None,
                        help="(deprecated, ignored) species file path")
    args = parser.parse_args()

    if args.species_file is not None:
        print("Note: --species-file is deprecated; species map is now "
              "sent by gRASPA at connection time.")

    print("=" * 60)
    print("gRASPA iPI Server (species-map protocol)")
    print("=" * 60)

    # 1. Load calculator first so GPU memory is allocated before gRASPA connects.
    calc = get_calculator(args)

    # 2. Create listener.
    sock_path = f"/tmp/{args.socket}"
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    print(f"Listening on {sock_path}")

    # Dummy cell for INIT handshake (client sends real cell with POSDATA)
    cell_hint = np.eye(3) * 10.0

    try:
        while True:
            print("Waiting for client connection...")
            conn, _ = srv.accept()
            print("Client connected")

            # 3. iPI handshake
            do_handshake(conn, cell_hint)

            # 4. Receive species map (sent by gRASPA right after handshake)
            try:
                species_map, n_fw = recv_species_map(conn)
            except (AssertionError, ConnectionError) as e:
                print(f"Species map exchange failed: {e}")
                conn.close()
                continue

            try:
                serve(conn, calc, species_map, n_fw)
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
