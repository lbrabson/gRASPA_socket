#!/usr/bin/env python3
"""
gRASPA iPI-protocol server — N-body (many-body) socket scheme.

Wire protocol (C++ → Python):
  After handshake, gRASPA sends SPECIESMAP then enters the STATUS loop.

  Old format (config_type < 200):
    config_type → cell[9] → inv_cell[9] → n_mol → natoms → types → xyz
    Used only by PrimeFrameworkCache (config_type=0).

  New format (config_type >= 200):
    config_type → mol_idx → cell[9] → inv_cell[9] → n_mol → natoms → types → xyz

  Config type routing:
    0         FRAMEWORK_PRIME  : cache E_fw
    200+comp  DELTA_QUERY      : ΔE for trial move (mol_idx: -1=ins, ≥0=mov/del)
    300       COMMIT_INSERT    : accept insertion  (mol_idx=-1, n_mol=1)
    301       COMMIT_DELETE    : accept deletion   (mol_idx=deleted, n_mol=0)
    302       COMMIT_MOVE      : accept move       (mol_idx=moved, n_mol=1)
    303       SYNC_FULL        : full resync       (mol_idx=ads_comp, n_mol=N)
    500       QUERY_TOTAL      : return E_current - E_fw_cached

Usage:
    python ase_ipi_server_mace.py \\
        --socket ase_ipi_socket \\
        --model /path/to/model.pt
"""

import argparse
import json
import os
import socket
import struct
import sys
import time
import random
import string

import numpy as np
from ase import Atoms

def generate_random_socket_name(prefix="graspa_", length=6):
    """Generate a random socket name with the given prefix and a specified length of random hex characters."""
    random_chars = "".join(random.choices(string.hexdigits.lower(), k=length))
    return prefix + random_chars

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
    conn.sendall(np.asarray(arr, dtype=np.float64).tobytes())


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
# Atoms builders
# ---------------------------------------------------------------------------
def _build_atoms(fw_symbols, fw_positions, comp_molecules, comp_mol_syms, cell):
    """Combine framework + all stored adsorbate molecules into one ASE Atoms."""
    all_syms = list(fw_symbols)
    all_pos  = [fw_positions]
    for comp in sorted(comp_molecules.keys()):
        mol_syms = comp_mol_syms.get(comp, [])
        for mol_pos in comp_molecules[comp]:
            if mol_syms and mol_pos.shape[0] != len(mol_syms):
                raise ValueError(
                    f"comp_mol_syms[{comp}] has {len(mol_syms)} symbols "
                    f"but mol_pos has shape {mol_pos.shape} — "
                    f"mismatch between symbol count and position rows")
            all_syms.extend(mol_syms)
            all_pos.append(mol_pos)
    return Atoms(symbols=all_syms, positions=np.vstack(all_pos), cell=cell, pbc=True)


def _ads_energy(comp_molecules, comp_mol_syms, cell, calc):
    """Energy of all adsorbate molecules together, without the framework.

    Uses the same periodic cell as the full system so that MACE sees the
    same long-range environment.  Returns 0.0 when there are no adsorbates
    (avoids running MACE on a zero-atom system).
    """
    atoms_ads = _build_atoms([], np.zeros((0, 3)), comp_molecules, comp_mol_syms, cell)
    if len(atoms_ads) == 0:
        return 0.0
    atoms_ads.calc = calc
    return atoms_ads.get_potential_energy()


# ---------------------------------------------------------------------------
# iPI handshake + species map
# ---------------------------------------------------------------------------
def do_handshake(conn, cell_hint):
    send_header(conn, "STATUS")
    hdr = recv_header(conn)
    assert hdr == "NEEDINIT", f"Expected NEEDINIT, got {hdr}"
    send_header(conn, "INIT")
    send_doubles(conn, cell_hint.flatten())
    send_doubles(conn, np.linalg.inv(cell_hint).flatten())
    print("Handshake complete")


def recv_species_map(conn):
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


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------
def _print_timing_summary(t):
    print("\n=== SERVER TIMING PROFILE ===")
    hdr = (f"  {'Category':<16} {'calls':>7} {'t_recv':>8} {'t_build':>8} "
           f"{'t_mace_full':>11} {'t_mace_ads':>10} {'t_send':>8} {'total':>8} {'ms/call':>8}")
    print(hdr)
    for k, v in t.items():
        if v['count'] == 0:
            continue
        tot = sum(vv for kk, vv in v.items() if kk != 'count')
        avg = 1000.0 * tot / v['count']
        print(f"  {k:<16} {v['count']:>7} {v['t_recv']:>8.3f} {v['t_build']:>8.3f} "
              f"{v['t_mace_full']:>11.3f} {v['t_mace_ads']:>10.3f} {v['t_send']:>8.3f} "
              f"{tot:>8.3f} {avg:>8.1f}")
    print("=== END TIMING PROFILE ===\n")
    sys.stdout.flush()


def serve(conn, calc, species_map, n_fw, profile_output="./runs/profiles/server_profile.json"):
    """N-body iPI server loop.

    State maintained between calls:
      fw_symbols / fw_positions : framework atoms (set at FRAMEWORK_PRIME)
      cached_E_fw               : E(framework alone)
      comp_molecules[comp]      : list of np.ndarray (mol_size, 3) per component
      comp_mol_syms[comp]       : element symbols for one molecule of each component
      E_current                 : E(fw + all current adsorbates)
      E_ads_current             : E(all current adsorbates, no framework)
      pending_E                 : E(fw + trial adsorbates) from last DELTA_QUERY
      pending_ads_E             : E(trial adsorbates, no fw) from last DELTA_QUERY
      last_delta_comp           : ads_comp from most recent DELTA_QUERY

    ΔE returned to C++ for DELTA_QUERY:
      ΔE = [E(fw + trial_ads) - E(trial_ads)] - [E(fw + current_ads) - E(current_ads)]

    This is the change in framework-adsorbate interaction energy evaluated by
    the ML potential, with direct adsorbate-adsorbate energies subtracted out
    (those are handled classically as GG in gRASPA).  Many-body effects that
    involve the framework (fw-mediated ads-ads polarisation, etc.) are captured
    because both full-system calls see all atoms simultaneously.
    """
    _timing = {k: {'count': 0, 't_recv': 0.0, 't_build': 0.0,
                   't_mace_full': 0.0, 't_mace_ads': 0.0, 't_send': 0.0}
               for k in ('FRAMEWORK', 'DELTA_QUERY', 'COMMIT_INSERT',
                         'COMMIT_DELETE', 'COMMIT_MOVE', 'SYNC_FULL', 'QUERY_TOTAL')}
    DELTA_QUERY_BASE = 200
    COMMIT_INSERT    = 300
    COMMIT_DELETE    = 301
    COMMIT_MOVE      = 302
    SYNC_FULL        = 303
    QUERY_TOTAL      = 500

    step            = 0
    cached_E_fw     = None
    fw_symbols      = None
    fw_positions    = None
    comp_molecules  = {}     # comp -> list of np.ndarray (mol_size, 3)
    comp_mol_syms   = {}     # comp -> [symbol, ...] for one molecule
    E_current       = None   # E(fw + all current ads)
    E_ads_current   = 0.0    # E(all current ads, no fw); 0 when box is empty
    pending_E       = None   # E(fw + trial ads) from last DELTA_QUERY
    pending_ads_E   = 0.0    # E(trial ads, no fw) from last DELTA_QUERY
    pending_mol     = None   # trial positions from last DELTA_QUERY
    last_delta_comp = None

    while True:
        # ---- iPI STATUS / READY ----
        try:
            _t_recv_start = time.perf_counter()
            send_header(conn, "STATUS")
            hdr = recv_header(conn)
        except (ConnectionError, BrokenPipeError, OSError) as e:
            print(f"Connection lost during STATUS/READY: {e}")
            break
        if hdr in ("EXIT", ""):
            print("Client sent EXIT — shutting down")
            break
        if hdr != "READY":
            print(f"Warning: expected READY, got '{hdr}'")
            break
        send_header(conn, "POSDATA")

        # ---- Parse header ----
        config_type = recv_int32(conn)

        # New protocol: config_type >= 200 carries mol_idx before cell
        if config_type >= DELTA_QUERY_BASE:
            mol_idx = recv_int32(conn)
        else:
            mol_idx = None

        cell   = recv_doubles(conn, 9).reshape(3, 3)
        _inv   = recv_doubles(conn, 9)   # discarded
        n_mol  = recv_int32(conn)
        natoms = recv_int32(conn)

        if natoms > 0:
            types_raw = _recvall(conn, 4 * natoms)
            types_arr = struct.unpack(f"!{natoms}i", types_raw)
            print(f"[SERVER] types_arr sample: {types_arr[:10]}")
            print(f"[SERVER] species_map keys: {list(species_map.keys())}")
            sys.stdout.flush()
            positions = recv_doubles(conn, 3 * natoms).reshape(natoms, 3)
            symbols   = [species_map[t] for t in types_arr]
        else:
            types_arr = ()
            positions = np.zeros((0, 3), dtype=np.float64)
            symbols   = []
        _t_recv_end = time.perf_counter()

        # ---- Route ----
        energy = 0.0
        label  = f"config_type={config_type}"
        _t_build = 0.0; _t_mace_full = 0.0; _t_mace_ads = 0.0
        _timing_key = None

        if config_type == 0:
            # Framework prime: cache E_fw, initialise state
            _timing_key = 'FRAMEWORK'
            fw_symbols    = list(symbols)    # explicit copy — not an alias
            fw_positions  = positions.copy()
            _tb = time.perf_counter()
            atoms_fw = Atoms(symbols=fw_symbols, positions=fw_positions,
                             cell=cell, pbc=True)
            atoms_fw.calc = calc
            _t_build = time.perf_counter() - _tb
            _tm = time.perf_counter()
            print(atoms_fw)
            cached_E_fw   = atoms_fw.get_potential_energy()
            _t_mace_full = time.perf_counter() - _tm
            E_current     = cached_E_fw   # zero adsorbates → E_current = E_fw
            E_ads_current = 0.0           # no adsorbates yet
            energy = cached_E_fw
            label  = "framework"
            print(f"  E_fw cached: {cached_E_fw:.6f} eV")

        elif DELTA_QUERY_BASE <= config_type < COMMIT_INSERT:
            # DELTA_QUERY: compute ΔE = Δ(E_full - E_ads)
            #   = [E(fw+trial_ads) - E(trial_ads)] - [E(fw+cur_ads) - E(cur_ads)]
            _timing_key = 'DELTA_QUERY'
            ads_comp        = config_type - DELTA_QUERY_BASE
            last_delta_comp = ads_comp

            # Build trial molecule lists (shallow-copy to avoid mutating state)
            trial_mols = {c: list(mols) for c, mols in comp_molecules.items()}
            mol_list   = list(trial_mols.get(ads_comp, []))

            if mol_idx == -1:
                # Insertion
                pending_mol = positions.copy()
                mol_list.append(pending_mol)
                if ads_comp not in comp_mol_syms and symbols:
                    comp_mol_syms[ads_comp] = symbols
                move_str = "INS"
            elif natoms == 0:
                # Deletion: swap-with-last (mirrors C++ Update_deletion_data)
                pending_mol = None
                if 0 <= mol_idx < len(mol_list):
                    mol_list[mol_idx] = mol_list[-1]
                    mol_list.pop()
                else:
                    print(f"  WARNING: DELTA_QUERY DEL mol_idx={mol_idx} "
                          f"out of range (n={len(mol_list)})")
                move_str = "DEL"
            else:
                # Move / reinsertion
                pending_mol = positions.copy()
                if 0 <= mol_idx < len(mol_list):
                    mol_list[mol_idx] = pending_mol
                else:
                    print(f"  WARNING: DELTA_QUERY MOV mol_idx={mol_idx} "
                          f"out of range (n={len(mol_list)})")
                move_str = "MOV"

            trial_mols[ads_comp] = mol_list

            # Full system energy (fw + all trial adsorbates)
            _tb = time.perf_counter()
            atoms_trial = _build_atoms(fw_symbols, fw_positions,
                                       trial_mols, comp_mol_syms, cell)
            atoms_trial.calc = calc
            _t_build = time.perf_counter() - _tb
            _tm = time.perf_counter()
            E_full_trial = atoms_trial.get_potential_energy()
            _t_mace_full = time.perf_counter() - _tm

            # Adsorbate-only energy (same positions, no framework atoms)
            _ta = time.perf_counter()
            E_ads_trial = _ads_energy(trial_mols, comp_mol_syms, cell, calc)
            _t_mace_ads = time.perf_counter() - _ta

            pending_E     = E_full_trial
            pending_ads_E = E_ads_trial

            # ΔE = Δ(fw-ads interaction) — direct ads-ads energy cancels out
            energy = (E_full_trial - E_ads_trial) - (E_current - E_ads_current)

            label = (f"DELTA_QUERY comp={ads_comp} {move_str} "
                     f"mol_idx={mol_idx} natoms={natoms} "
                     f"n_stored={len(comp_molecules.get(ads_comp, []))}")
            print(f"  {label}")
            print(f"    E_full={E_full_trial:.6f}  E_ads={E_ads_trial:.6f}  "
                  f"E_cur={E_current:.6f}  E_ads_cur={E_ads_current:.6f}  "
                  f"dE={energy:.6f} eV")

        elif config_type == COMMIT_INSERT:
            _timing_key = 'COMMIT_INSERT'
            ads_comp = last_delta_comp
            if ads_comp not in comp_molecules:
                comp_molecules[ads_comp] = []
            comp_molecules[ads_comp].append(pending_mol)
            if ads_comp not in comp_mol_syms and symbols:
                comp_mol_syms[ads_comp] = symbols
            E_current     = pending_E
            E_ads_current = pending_ads_E
            label = (f"COMMIT_INSERT comp={ads_comp} "
                     f"n_stored={len(comp_molecules[ads_comp])}")
            print(f"  {label}  E_current={E_current:.6f}  "
                  f"E_ads_current={E_ads_current:.6f} eV")

        elif config_type == COMMIT_DELETE:
            _timing_key = 'COMMIT_DELETE'
            ads_comp = last_delta_comp
            mols = comp_molecules.get(ads_comp, [])
            if 0 <= mol_idx < len(mols):
                mols[mol_idx] = mols[-1]
                mols.pop()
            else:
                print(f"  WARNING: COMMIT_DELETE mol_idx={mol_idx} "
                      f"out of range (n={len(mols)})")
            E_current     = pending_E
            E_ads_current = pending_ads_E
            label = (f"COMMIT_DELETE comp={ads_comp} mol_idx={mol_idx} "
                     f"n_stored={len(mols)}")
            print(f"  {label}  E_current={E_current:.6f}  "
                  f"E_ads_current={E_ads_current:.6f} eV")

        elif config_type == COMMIT_MOVE:
            _timing_key = 'COMMIT_MOVE'
            ads_comp = last_delta_comp
            mols = comp_molecules.get(ads_comp, [])
            if 0 <= mol_idx < len(mols):
                mols[mol_idx] = pending_mol
            else:
                print(f"  WARNING: COMMIT_MOVE mol_idx={mol_idx} "
                      f"out of range (n={len(mols)})")
            E_current     = pending_E
            E_ads_current = pending_ads_E
            label = f"COMMIT_MOVE comp={ads_comp} mol_idx={mol_idx}"
            print(f"  {label}  E_current={E_current:.6f}  "
                  f"E_ads_current={E_ads_current:.6f} eV")

        elif config_type == SYNC_FULL:
            # mol_idx encodes ads_comp for SYNC_FULL
            _timing_key = 'SYNC_FULL'
            ads_comp = mol_idx
            mol_size = natoms // n_mol if n_mol > 0 else 0
            mols     = []
            mol_syms_one = symbols[:mol_size] if mol_size > 0 else []
            for i in range(n_mol):
                mols.append(positions[i * mol_size:(i + 1) * mol_size].copy())
            comp_molecules[ads_comp] = mols
            if mol_syms_one:
                comp_mol_syms[ads_comp] = mol_syms_one
            # Recompute E_current and E_ads_current from scratch
            _tb = time.perf_counter()
            atoms_full = _build_atoms(fw_symbols, fw_positions,
                                      comp_molecules, comp_mol_syms, cell)
            atoms_full.calc = calc
            _t_build = time.perf_counter() - _tb
            _tm = time.perf_counter()
            E_current     = atoms_full.get_potential_energy()
            _t_mace_full = time.perf_counter() - _tm
            _ta = time.perf_counter()
            E_ads_current = _ads_energy(comp_molecules, comp_mol_syms, cell, calc)
            _t_mace_ads = time.perf_counter() - _ta
            energy = (E_current - E_ads_current) - cached_E_fw
            label = (f"SYNC_FULL comp={ads_comp} n_mol={n_mol} "
                     f"total_atoms={len(atoms_full)}")
            print(f"  {label}")
            print(f"    E_current={E_current:.6f}  E_ads={E_ads_current:.6f}  "
                  f"HG_int={energy:.6f} eV")

        elif config_type == QUERY_TOTAL:
            # Return pure fw-ads interaction energy (ads-ads handled as GG in gRASPA)
            _timing_key = 'QUERY_TOTAL'
            energy = (E_current - E_ads_current) - cached_E_fw
            label  = "QUERY_TOTAL"
            print(f"  QUERY_TOTAL  E_current={E_current:.6f}  "
                  f"E_ads={E_ads_current:.6f}  E_fw={cached_E_fw:.6f}  "
                  f"HG_int={energy:.6f} eV")

        else:
            print(f"  Unknown config_type={config_type} — returning 0")

        step += 1
        # print(f"  step {step:5d}  {label}  config_type={config_type:4d}  "
        #      f"natoms={natoms:5d}  n_mol={n_mol}  E={energy:14.6f} eV")

        # ---- iPI FORCEREADY response ----
        _t_send_start = time.perf_counter()
        send_header(conn, "STATUS")
        hdr = recv_header(conn)
        if hdr != "HAVEDATA":
            print(f"Warning: expected HAVEDATA, got '{hdr}'")
            break

        send_header(conn, "GETFORCE")
        send_header(conn, "FORCEREADY")
        send_doubles(conn, np.array([energy]))  # energy scalar
        send_int32(conn, 1)                     # natoms=1 (avoids empty-read in C++)
        send_doubles(conn, np.zeros(3))         # forces for 1 dummy atom
        send_doubles(conn, np.zeros(9))         # virial
        send_int32(conn, 0)                     # extras length
        _t_send_end = time.perf_counter()

        # ---- Accumulate timing ----
        if _timing_key is not None:
            _timing[_timing_key]['count']      += 1
            _timing[_timing_key]['t_recv']     += _t_recv_end - _t_recv_start
            _timing[_timing_key]['t_build']    += _t_build
            _timing[_timing_key]['t_mace_full'] += _t_mace_full
            _timing[_timing_key]['t_mace_ads'] += _t_mace_ads
            _timing[_timing_key]['t_send']     += _t_send_end - _t_send_start

    # ---- End-of-session summary ----
    _print_timing_summary(_timing)
    os.makedirs(os.path.dirname(os.path.abspath(profile_output)), exist_ok=True)
    with open(profile_output, 'w') as f:
        json.dump(_timing, f, indent=2)
    print(f"Timing profile written to {profile_output}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="N-body iPI server for gRASPA + ML potentials"
    )
    parser.add_argument("--socket", "-s", default=None,
                        help="UNIX socket name (creates /tmp/<name>); "
                             "if omitted, caller should pre-generate via "
                             "generate_random_socket_name()")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to MACE model file (.pt)")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cuda", "cpu"],
                        help="Device for calculator (default: cpu)")
    parser.add_argument("--dtype", default="float64",
                        choices=["float32", "float64"],
                        help="Floating-point precision (default: float64)")
    parser.add_argument("--species-file", default=None,
                        help="(deprecated, ignored) species file path")
    parser.add_argument("--profile-output", default="./runs/profiles/server_profile.json",
                        help="Path to write JSON timing profile (default: ./runs/profiles/server_profile.json)")
    args = parser.parse_args()

    if args.socket is None:
        parser.error("--socket is required (use generate_random_socket_name() in the launch script)")

    # Force line-buffered stdout so prints appear immediately even when
    # stdout is redirected to a file or pipe (default is block-buffered).
    sys.stdout.reconfigure(line_buffering=True)

    if args.species_file is not None:
        print("Note: --species-file is deprecated; species map is sent by gRASPA.")

    print("=" * 60)
    print("gRASPA iPI Server (N-body protocol)")
    print("=" * 60)

    calc = get_calculator(args)

    sock_path = f"/tmp/{args.socket}"
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    print(f"Listening on {sock_path}")

    cell_hint = np.eye(3) * 10.0

    try:
        while True:
            print("Waiting for client connection...")
            conn, _ = srv.accept()
            print("Client connected")

            do_handshake(conn, cell_hint)

            try:
                species_map, n_fw = recv_species_map(conn)
            except (AssertionError, ConnectionError) as e:
                print(f"Species map exchange failed: {e}")
                conn.close()
                continue

            try:
                serve(conn, calc, species_map, n_fw, args.profile_output)
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
