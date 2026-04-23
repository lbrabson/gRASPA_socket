#!/usr/bin/env python3
"""
gRASPA iPI-protocol server — N-body (many-body) socket scheme.
TOTAL energy variant: energies are E(fw + all adsorbates) with the framework
self-energy subtracted.  GG interactions are handled by the MLIP (not classically).

Energies returned to C++:
  DELTA_QUERY (insertion):  [E(fw+trial) - E_isolated_mol] - E_current
  DELTA_QUERY (deletion):   [E(fw+trial) + E_isolated_mol] - E_current
  DELTA_QUERY (move):        E(fw+trial) - E_current   (self-energy cancels)
  QUERY_TOTAL:               E_current - cached_E_fw   (pure ads interaction energy)

Wire protocol (C++ -> Python):
  After handshake, gRASPA sends SPECIESMAP then enters the STATUS loop.

  Old format (config_type < 200):
    config_type -> cell[9] -> inv_cell[9] -> n_mol -> natoms -> types -> xyz
    Used only by PrimeFrameworkCache (config_type=0).

  New format (config_type >= 200):
    config_type -> mol_idx -> cell[9] -> inv_cell[9] -> n_mol -> natoms -> types -> xyz

  Config type routing:
    0         FRAMEWORK_PRIME  : cache E_fw
    200+comp  DELTA_QUERY      : dE for trial move (mol_idx: -1=ins, >=0=mov/del)
    300       COMMIT_INSERT    : accept insertion  (mol_idx=-1, n_mol=1)
    301       COMMIT_DELETE    : accept deletion   (mol_idx=deleted, n_mol=0)
    302       COMMIT_MOVE      : accept move       (mol_idx=moved, n_mol=1)
    303       SYNC_FULL        : full resync       (mol_idx=ads_comp, n_mol=N)
    500       QUERY_TOTAL      : return E_current - E_fw (pure ads interaction)

Usage:
    python ase_ipi_server_mace_total.py \\
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
    random_chars = "".join(random.choices(string.hexdigits.lower(), k=length))
    return prefix + random_chars


# ---------------------------------------------------------------------------
# Calculator factory
# ---------------------------------------------------------------------------
def get_calculator(args):
    from mace.calculators import mace_mp

    calc = mace_mp(
        model=args.model,
        device=args.device,
        default_dtype=args.dtype,
        enable_cueq=True,
    )
    print(f"Loaded MACE model: {args.model}  device={args.device}")
    return calc


# ---------------------------------------------------------------------------
# Low-level iPI helpers
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
                    f"but mol_pos has shape {mol_pos.shape}")
            all_syms.extend(mol_syms)
            all_pos.append(mol_pos)
    return Atoms(symbols=all_syms, positions=np.vstack(all_pos), cell=cell, pbc=True)


def _isolated_mol_energy(mol_positions, mol_symbols, cell, calc):
    """Energy of a single isolated molecule in the periodic cell."""
    if len(mol_positions) == 0:
        return 0.0
    atoms = Atoms(symbols=mol_symbols, positions=mol_positions, cell=cell, pbc=True)
    atoms.calc = calc
    return atoms.get_potential_energy()


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
# Timing summary
# ---------------------------------------------------------------------------
def _print_timing_summary(t):
    print("\n=== SERVER TIMING PROFILE ===")
    hdr = (f"  {'Category':<16} {'calls':>7} {'t_recv':>8} {'t_build':>8} "
           f"{'t_mace_full':>11} {'t_mace_iso':>10} {'t_send':>8} {'total':>8} {'ms/call':>8}")
    print(hdr)
    for k, v in t.items():
        if v['count'] == 0:
            continue
        tot = sum(vv for kk, vv in v.items() if kk != 'count')
        avg = 1000.0 * tot / v['count']
        print(f"  {k:<16} {v['count']:>7} {v['t_recv']:>8.3f} {v['t_build']:>8.3f} "
              f"{v['t_mace_full']:>11.3f} {v['t_mace_iso']:>10.3f} {v['t_send']:>8.3f} "
              f"{tot:>8.3f} {avg:>8.1f}")
    print("=== END TIMING PROFILE ===\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------
def serve(conn, calc, species_map, n_fw, profile_output="./runs/profiles/server_profile.json"):
    """N-body iPI server loop — total energy + isolated molecule correction.

    State maintained between calls:
      fw_symbols / fw_positions : framework atoms (set at FRAMEWORK_PRIME)
      cached_E_fw               : E(framework alone)
      comp_molecules[comp]      : list of np.ndarray (mol_size, 3) per component
      comp_mol_syms[comp]       : element symbols for one molecule of each component
      E_current                 : E(fw + all current adsorbates)
      pending_E                 : E(fw + trial adsorbates) from last DELTA_QUERY

    Energies returned to C++:
      DELTA_QUERY insertion:  (E_trial - E_isolated) - E_current
      DELTA_QUERY deletion:   (E_trial + E_isolated) - E_current
      DELTA_QUERY move:        E_trial - E_current   (self-energy cancels)
      QUERY_TOTAL:             E_current - cached_E_fw
    """
    _timing = {k: {'count': 0, 't_recv': 0.0, 't_build': 0.0,
                   't_mace_full': 0.0, 't_mace_iso': 0.0, 't_send': 0.0}
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
    comp_molecules  = {}
    comp_mol_syms   = {}
    E_current       = None
    pending_E       = None
    pending_mol     = None
    last_delta_comp = None

    while True:
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

        if config_type >= DELTA_QUERY_BASE:
            mol_idx = recv_int32(conn)
        else:
            mol_idx = None

        cell   = recv_doubles(conn, 9).reshape(3, 3)
        _inv   = recv_doubles(conn, 9)
        n_mol  = recv_int32(conn)
        natoms = recv_int32(conn)

        if natoms > 0:
            types_raw = _recvall(conn, 4 * natoms)
            types_arr = struct.unpack(f"!{natoms}i", types_raw)
            sys.stdout.flush()
            positions = recv_doubles(conn, 3 * natoms).reshape(natoms, 3)
            symbols   = [species_map[t] for t in types_arr]
        else:
            types_arr = ()
            positions = np.zeros((0, 3), dtype=np.float64)
            symbols   = []
        _t_recv_end = time.perf_counter()

        energy      = 0.0
        label       = f"config_type={config_type}"
        _t_build    = 0.0
        _t_mace_full = 0.0
        _t_mace_iso  = 0.0
        _timing_key  = None

        # ------------------------------------------------------------------ #
        if config_type == 0:
            # FRAMEWORK_PRIME
            _timing_key  = 'FRAMEWORK'
            fw_symbols   = list(symbols)
            fw_positions = positions.copy()

            _tb = time.perf_counter()
            atoms_fw = Atoms(symbols=fw_symbols, positions=fw_positions,
                             cell=cell, pbc=True)
            atoms_fw.calc = calc
            _t_build = time.perf_counter() - _tb

            _tm = time.perf_counter()
            print(atoms_fw)
            cached_E_fw  = atoms_fw.get_potential_energy()
            _t_mace_full = time.perf_counter() - _tm

            E_current = cached_E_fw
            energy    = cached_E_fw
            label     = "framework"
            print(f"  E_fw cached: {cached_E_fw:.6f} eV")

        # ------------------------------------------------------------------ #
        elif DELTA_QUERY_BASE <= config_type < COMMIT_INSERT:
            _timing_key     = 'DELTA_QUERY'
            ads_comp        = config_type - DELTA_QUERY_BASE
            last_delta_comp = ads_comp

            trial_mols = {c: list(mols) for c, mols in comp_molecules.items()}
            mol_list   = list(trial_mols.get(ads_comp, []))
            E_isolated = 0.0

            if mol_idx == -1:
                # ---- INSERTION: subtract isolated molecule self-energy ----
                pending_mol = positions.copy()
                mol_list.append(pending_mol)
                if ads_comp not in comp_mol_syms and symbols:
                    comp_mol_syms[ads_comp] = symbols
                trial_mols[ads_comp] = mol_list

                _tb = time.perf_counter()
                atoms_trial = _build_atoms(fw_symbols, fw_positions,
                                           trial_mols, comp_mol_syms, cell)
                atoms_trial.calc = calc
                _t_build = time.perf_counter() - _tb

                _tm = time.perf_counter()
                E_full_trial = atoms_trial.get_potential_energy()
                _t_mace_full = time.perf_counter() - _tm

                _ti = time.perf_counter()
                E_isolated = _isolated_mol_energy(
                    pending_mol, comp_mol_syms[ads_comp], cell, calc)
                _t_mace_iso = time.perf_counter() - _ti

                pending_E = E_full_trial
                energy    = (E_full_trial - E_isolated) - E_current
                move_str  = "INS"
                print(f"    E_trial={E_full_trial:.6f}  E_isolated={E_isolated:.6f}  "
                      f"E_current={E_current:.6f}  dE={energy:.6f} eV")

            elif natoms == 0:
                # ---- DELETION: add back isolated molecule self-energy ----
                pending_mol      = None
                mol_syms_deleted = comp_mol_syms.get(ads_comp, [])
                if 0 <= mol_idx < len(mol_list):
                    deleted_pos      = mol_list[mol_idx].copy()
                    mol_list[mol_idx] = mol_list[-1]
                    mol_list.pop()
                else:
                    print(f"  WARNING: DELTA_QUERY DEL mol_idx={mol_idx} "
                          f"out of range (n={len(mol_list)})")
                    deleted_pos = np.zeros((0, 3))
                trial_mols[ads_comp] = mol_list

                _tb = time.perf_counter()
                atoms_trial = _build_atoms(fw_symbols, fw_positions,
                                           trial_mols, comp_mol_syms, cell)
                atoms_trial.calc = calc
                _t_build = time.perf_counter() - _tb

                _tm = time.perf_counter()
                E_full_trial = atoms_trial.get_potential_energy()
                _t_mace_full = time.perf_counter() - _tm

                _ti = time.perf_counter()
                E_isolated = _isolated_mol_energy(
                    deleted_pos, mol_syms_deleted, cell, calc)
                _t_mace_iso = time.perf_counter() - _ti

                pending_E = E_full_trial
                energy    = (E_full_trial + E_isolated) - E_current
                move_str  = "DEL"
                print(f"    E_trial={E_full_trial:.6f}  E_isolated={E_isolated:.6f}  "
                      f"E_current={E_current:.6f}  dE={energy:.6f} eV")

            else:
                # ---- MOVE / REINSERTION: self-energy cancels, no correction ----
                pending_mol = positions.copy()
                if 0 <= mol_idx < len(mol_list):
                    mol_list[mol_idx] = pending_mol
                else:
                    print(f"  WARNING: DELTA_QUERY MOV mol_idx={mol_idx} "
                          f"out of range (n={len(mol_list)})")
                trial_mols[ads_comp] = mol_list

                _tb = time.perf_counter()
                atoms_trial = _build_atoms(fw_symbols, fw_positions,
                                           trial_mols, comp_mol_syms, cell)
                atoms_trial.calc = calc
                _t_build = time.perf_counter() - _tb

                _tm = time.perf_counter()
                E_full_trial = atoms_trial.get_potential_energy()
                _t_mace_full = time.perf_counter() - _tm

                pending_E = E_full_trial
                energy    = E_full_trial - E_current
                move_str  = "MOV"
                print(f"    E_trial={E_full_trial:.6f}  E_current={E_current:.6f}  "
                      f"dE={energy:.6f} eV")

            label = (f"DELTA_QUERY comp={ads_comp} {move_str} "
                     f"mol_idx={mol_idx} natoms={natoms} "
                     f"n_stored={len(comp_molecules.get(ads_comp, []))}")
            print(f"  {label}")

        # ------------------------------------------------------------------ #
        elif config_type == COMMIT_INSERT:
            _timing_key = 'COMMIT_INSERT'
            ads_comp = last_delta_comp
            if ads_comp not in comp_molecules:
                comp_molecules[ads_comp] = []
            comp_molecules[ads_comp].append(pending_mol)
            if ads_comp not in comp_mol_syms and symbols:
                comp_mol_syms[ads_comp] = symbols
            E_current = pending_E
            label = (f"COMMIT_INSERT comp={ads_comp} "
                     f"n_stored={len(comp_molecules[ads_comp])}")
            print(f"  {label}  E_current={E_current:.6f} eV")

        # ------------------------------------------------------------------ #
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
            E_current = pending_E
            label = (f"COMMIT_DELETE comp={ads_comp} mol_idx={mol_idx} "
                     f"n_stored={len(mols)}")
            print(f"  {label}  E_current={E_current:.6f} eV")

        # ------------------------------------------------------------------ #
        elif config_type == COMMIT_MOVE:
            _timing_key = 'COMMIT_MOVE'
            ads_comp = last_delta_comp
            mols = comp_molecules.get(ads_comp, [])
            if 0 <= mol_idx < len(mols):
                mols[mol_idx] = pending_mol
            else:
                print(f"  WARNING: COMMIT_MOVE mol_idx={mol_idx} "
                      f"out of range (n={len(mols)})")
            E_current = pending_E
            label = f"COMMIT_MOVE comp={ads_comp} mol_idx={mol_idx}"
            print(f"  {label}  E_current={E_current:.6f} eV")

        # ------------------------------------------------------------------ #
        elif config_type == SYNC_FULL:
            _timing_key  = 'SYNC_FULL'
            ads_comp     = mol_idx
            mol_size     = natoms // n_mol if n_mol > 0 else 0
            mols         = []
            mol_syms_one = symbols[:mol_size] if mol_size > 0 else []
            for i in range(n_mol):
                mols.append(positions[i * mol_size:(i + 1) * mol_size].copy())
            comp_molecules[ads_comp] = mols
            if mol_syms_one:
                comp_mol_syms[ads_comp] = mol_syms_one

            _tb = time.perf_counter()
            atoms_full = _build_atoms(fw_symbols, fw_positions,
                                      comp_molecules, comp_mol_syms, cell)
            atoms_full.calc = calc
            _t_build = time.perf_counter() - _tb

            _tm = time.perf_counter()
            E_current    = atoms_full.get_potential_energy()
            _t_mace_full = time.perf_counter() - _tm

            energy = E_current - cached_E_fw
            label  = (f"SYNC_FULL comp={ads_comp} n_mol={n_mol} "
                      f"total_atoms={len(atoms_full)}")
            print(f"  {label}")
            print(f"    E_current={E_current:.6f}  E_fw={cached_E_fw:.6f}  "
                  f"ads_interaction={energy:.6f} eV")

        # ------------------------------------------------------------------ #
        elif config_type == QUERY_TOTAL:
            # Pure adsorbate interaction energy (HG + GG) from MLIP
            _timing_key = 'QUERY_TOTAL'
            energy = E_current
            label  = "QUERY_TOTAL"
            print(f"  QUERY_TOTAL  E_current={E_current:.6f}  "
                  f"E_fw={cached_E_fw:.6f}  ads_interaction={energy:.6f} eV")

        else:
            print(f"  Unknown config_type={config_type} — returning 0")

        step += 1

        # ---- iPI FORCEREADY response ----
        _t_send_start = time.perf_counter()
        send_header(conn, "STATUS")
        hdr = recv_header(conn)
        if hdr != "HAVEDATA":
            print(f"Warning: expected HAVEDATA, got '{hdr}'")
            break

        send_header(conn, "GETFORCE")
        send_header(conn, "FORCEREADY")
        send_doubles(conn, np.array([energy]))
        send_int32(conn, 1)
        send_doubles(conn, np.zeros(3))
        send_doubles(conn, np.zeros(9))
        send_int32(conn, 0)
        _t_send_end = time.perf_counter()

        if _timing_key is not None:
            _timing[_timing_key]['count']        += 1
            _timing[_timing_key]['t_recv']       += _t_recv_end - _t_recv_start
            _timing[_timing_key]['t_build']      += _t_build
            _timing[_timing_key]['t_mace_full']  += _t_mace_full
            _timing[_timing_key]['t_mace_iso']   += _t_mace_iso
            _timing[_timing_key]['t_send']       += _t_send_end - _t_send_start

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
        description="N-body iPI server for gRASPA + ML potentials (total energy + iso correction)"
    )
    parser.add_argument("--socket", "-s", default=None,
                        help="UNIX socket name (creates /tmp/<name>)")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to MACE model file (.pt)")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cuda", "cpu"],
                        help="Device for calculator (default: cpu)")
    parser.add_argument("--dtype", default="float64",
                        choices=["float32", "float64"],
                        help="Floating-point precision (default: float64)")
    parser.add_argument("--species-file", default=None,
                        help="(deprecated, ignored)")
    parser.add_argument("--profile-output",
                        default="./runs/profiles/server_profile.json",
                        help="Path to write JSON timing profile")
    args = parser.parse_args()

    if args.socket is None:
        parser.error("--socket is required")

    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 60)
    print("gRASPA iPI Server (N-body, total energy + isolated mol correction)")
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