#!/usr/bin/env python3
"""
validate_energies.py — re-run MACE on structures dumped by gRASPA validation mode
and compare the independently computed energies to those gRASPA recorded.

Usage
-----
    python validate_energies.py \\
        --xyz  validation_dump_box0.xyz \\
        --model /path/to/model.pt \\
        [--device cpu|cuda] \\
        [--dtype float32|float64] \\
        [--tol 1e-4] \\
        [--max-frames N]

The XYZ file is the extended-XYZ file written by gRASPA when ValidationMode=yes.
Each frame's comment line contains key=value metadata::

    MOVE=<label> E_total_ev=<v> E_fw_ev=<v> E_ads_ev=<v> E_int_ev=<v>
    N_FW=<n_fw> Lattice="<9 floats>" Properties=species:S:1:pos:R:3 pbc="T T T"

What this script checks
-----------------------
1. Framework self-energy (frame 0 only):
   MACE(framework atoms) == gRASPA cached E_fw
2. Adsorbate self-energy (frame 0 only):
   MACE(adsorbate atoms) == gRASPA cached E_ads
3. Per-frame interaction energy:
   MACE(all atoms) - E_fw_mace - E_ads_mace == gRASPA E_int_ev

A discrepancy in (1) or (2) means the self-energy cache is wrong (e.g. wrong
atom positions or species sent to MACE on initialisation).  A discrepancy in
(3) means gRASPA is sending different positions during MC than what gRASPA
believes it is sending.
"""

import argparse
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Calculator factory
# ---------------------------------------------------------------------------

def get_calculator(model, device, dtype):
    from mace.calculators import mace_mp
    calc = mace_mp(model=model, device=device, default_dtype=dtype)
    print(f"Loaded MACE model : {model}")
    print(f"  device={device}  dtype={dtype}")
    return calc


# ---------------------------------------------------------------------------
# Extended-XYZ parser
# ---------------------------------------------------------------------------

def parse_comment(line):
    """Return a dict extracted from an extended-XYZ comment line.

    Handles:
      - MOVE=<word>               (string label)
      - KEY=<number>              (int or float)
      - Lattice="f f f ..."       (9 floats as list)
      - pbc="T T T"               (list of strings)
    """
    result = {}

    # MOVE label (string value)
    m = re.search(r'MOVE=(\S+)', line)
    if m:
        result['MOVE'] = m.group(1)

    # Lattice="..." — may contain spaces, must be handled before generic KEY=VAL
    m = re.search(r'Lattice="([^"]+)"', line)
    if m:
        result['Lattice'] = list(map(float, m.group(1).split()))

    # pbc="..."
    m = re.search(r'pbc="([^"]+)"', line)
    if m:
        result['pbc'] = m.group(1).split()

    # Generic KEY=<numeric> pairs (int or float, including scientific notation)
    for m in re.finditer(r'(\w+)=([-+]?[\d][\d\.eE+\-]*)', line):
        key, val = m.group(1), m.group(2)
        if key in ('Lattice', 'pbc', 'MOVE', 'Properties'):
            continue  # handled separately
        try:
            result[key] = int(val)
        except ValueError:
            try:
                result[key] = float(val)
            except ValueError:
                result[key] = val

    return result


def parse_xyz_frames(path):
    """Yield (natoms, comment, symbols, positions) for each frame in an XYZ file."""
    with open(path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            header = header.strip()
            if not header:
                continue
            try:
                natoms = int(header)
            except ValueError:
                continue
            comment = f.readline().rstrip('\n')
            symbols = []
            positions = []
            for _ in range(natoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Atom line too short: {parts!r}")
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
            yield natoms, comment, symbols, np.array(positions)


# ---------------------------------------------------------------------------
# ASE helpers
# ---------------------------------------------------------------------------

def make_atoms(symbols, positions, lattice_flat, pbc=True):
    from ase import Atoms
    cell = np.array(lattice_flat).reshape(3, 3)
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)


def compute_energy(calc, atoms):
    atoms.calc = calc
    return atoms.get_potential_energy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate gRASPA validation-dump energies by re-running MACE"
    )
    parser.add_argument("--xyz", "-x", required=True,
                        help="validation_dump_box0.xyz from gRASPA")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to MACE model file (.pt)")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for MACE (default: cpu)")
    parser.add_argument("--dtype", default="float64",
                        choices=["float32", "float64"],
                        help="Floating-point precision (default: float64)")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="E_int tolerance in eV to flag as FAIL (default: 1e-4)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames (default: all)")
    args = parser.parse_args()

    # Load MACE
    calc = get_calculator(args.model, args.device, args.dtype)

    print(f"\nParsing: {args.xyz}")
    print("-" * 90)

    # State that is fixed after frame 0
    E_fw_mace   = None
    E_ads_mace  = None
    E_fw_graspa_ref  = None
    E_ads_graspa_ref = None

    results = []

    for frame_idx, (natoms, comment, symbols, positions) in enumerate(
            parse_xyz_frames(args.xyz)):

        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        # --- parse metadata ---
        meta = parse_comment(comment)
        move          = meta.get('MOVE', 'UNKNOWN')
        E_total_ref   = meta.get('E_total_ev')
        E_fw_ref      = meta.get('E_fw_ev')
        E_ads_ref     = meta.get('E_ads_ev')
        E_int_ref     = meta.get('E_int_ev')
        n_fw          = meta.get('N_FW')
        lattice       = meta.get('Lattice')

        missing = [k for k, v in dict(E_total_ev=E_total_ref, E_fw_ev=E_fw_ref,
                                       E_ads_ev=E_ads_ref, E_int_ev=E_int_ref,
                                       N_FW=n_fw, Lattice=lattice).items()
                   if v is None]
        if missing:
            print(f"Frame {frame_idx}: WARNING — missing fields {missing}, skipping")
            continue

        n_fw = int(n_fw)
        n_ads = natoms - n_fw

        fw_symbols  = symbols[:n_fw]
        ads_symbols = symbols[n_fw:]
        fw_pos      = positions[:n_fw]
        ads_pos     = positions[n_fw:]

        # --- on the first frame: verify self-energies ---
        if frame_idx == 0:
            E_fw_graspa_ref  = E_fw_ref
            E_ads_graspa_ref = E_ads_ref

            print(f"Frame 0 — self-energy verification (computed once, should match gRASPA cache)")
            print(f"  n_fw={n_fw}  n_ads={n_ads}  n_total={natoms}")

            atoms_fw = make_atoms(fw_symbols, fw_pos, lattice)
            E_fw_mace = compute_energy(calc, atoms_fw)

            atoms_ads = make_atoms(ads_symbols, ads_pos, lattice)
            E_ads_mace = compute_energy(calc, atoms_ads)

            dfw  = E_fw_mace  - E_fw_ref
            dads = E_ads_mace - E_ads_ref
            fw_ok  = "OK  " if abs(dfw)  < args.tol else "FAIL"
            ads_ok = "OK  " if abs(dads) < args.tol else "FAIL"
            print(f"  [{fw_ok}]  E_fw  : MACE={E_fw_mace:+16.8f} eV  "
                  f"gRASPA={E_fw_ref:+16.8f} eV  diff={dfw:+.4e} eV")
            print(f"  [{ads_ok}] E_ads : MACE={E_ads_mace:+16.8f} eV  "
                  f"gRASPA={E_ads_ref:+16.8f} eV  diff={dads:+.4e} eV")
            print()

        else:
            # Verify gRASPA is still using the same cached self-energies
            if (abs(E_fw_ref - E_fw_graspa_ref) > 1e-9 or
                    abs(E_ads_ref - E_ads_graspa_ref) > 1e-9):
                print(f"Frame {frame_idx}: WARNING — gRASPA E_fw or E_ads changed; "
                      f"cache invalidation may have occurred")

        # --- compute E_total for this frame ---
        atoms_total = make_atoms(symbols, positions, lattice)
        E_total_mace = compute_energy(calc, atoms_total)

        # Interaction energy: same decomposition as gRASPA
        E_int_mace = E_total_mace - E_fw_mace - E_ads_mace

        err_total = E_total_mace - E_total_ref
        err_int   = E_int_mace   - E_int_ref

        status = "OK  " if abs(err_int) < args.tol else "FAIL"
        print(f"[{status}] frame={frame_idx:4d}  {move:22s} | "
              f"E_total: MACE={E_total_mace:+12.5f}  gRASPA={E_total_ref:+12.5f}  "
              f"Δ={err_total:+.3e} | "
              f"E_int: MACE={E_int_mace:+10.5f}  gRASPA={E_int_ref:+10.5f}  "
              f"Δ={err_int:+.3e} eV")

        results.append(dict(
            frame=frame_idx, move=move,
            E_total_ref=E_total_ref, E_total_mace=E_total_mace,
            E_int_ref=E_int_ref,   E_int_mace=E_int_mace,
            err_total=err_total,   err_int=err_int,
        ))

    if not results:
        print("No frames processed.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print(f"SUMMARY  ({len(results)} frames, tolerance={args.tol} eV)")
    print("=" * 90)

    errs_total = np.array([r['err_total'] for r in results])
    errs_int   = np.array([r['err_int']   for r in results])

    def _stats(label, errs):
        idx_max = int(np.argmax(np.abs(errs)))
        print(f"{label}:")
        print(f"  mean ± std = {np.mean(errs):+.4e} ± {np.std(errs):.4e} eV")
        print(f"  max |err|  = {np.max(np.abs(errs)):.4e} eV  (frame {results[idx_max]['frame']}, "
              f"move={results[idx_max]['move']})")

    _stats("E_total errors (MACE − gRASPA)", errs_total)
    print()
    _stats("E_int   errors (MACE − gRASPA)", errs_int)

    failures = [r for r in results if abs(r['err_int']) >= args.tol]
    print()
    if failures:
        print(f"FAILURES (|ΔE_int| ≥ {args.tol} eV): {len(failures)}/{len(results)}")
        for r in failures[:20]:
            print(f"  frame={r['frame']:4d}  {r['move']:22s}  err_int={r['err_int']:+.4e} eV")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        sys.exit(1)
    else:
        print(f"All {len(results)} frames PASSED (|ΔE_int| < {args.tol} eV). "
              f"Structures and energies are consistent.")


if __name__ == "__main__":
    main()
