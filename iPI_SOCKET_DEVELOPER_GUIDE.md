# iPI Socket Communication in gRASPA - Developer Guide

## Table of Contents
1. [Overview](#overview)
2. [Directory Roles](#directory-roles)
3. [Architecture](#architecture)
4. [iPI Protocol Specification](#ipi-protocol-specification)
5. [Core Implementation: `ase_energy_client.h`](#core-implementation)
6. [Integration Points Across the Codebase](#integration-points)
7. [Patching System](#patching-system)
8. [Configuration](#configuration)
9. [Energy Calculation Strategy](#energy-calculation-strategy)
10. [Supercell Replication](#supercell-replication)
11. [Python Server](#python-server)
12. [Runtime Workflow](#runtime-workflow)
13. [Error Handling](#error-handling)
14. [Known Limitations](#known-limitations)
15. [Known Issues and Future Work](#known-issues-and-future-work)
16. [Changelog](#changelog)
17. [Supercell, PBC, and Energy Decomposition Correctness](#supercell-pbc-and-energy-decomposition-correctness)
18. [Allegro vs Socket/ASE: Energy Calculation Differences](#allegro-vs-socketase-energy-calculation-differences)
19. [Per-Move Energy Breakdown and Atoms Involved](#per-move-energy-breakdown-and-atoms-involved)
20. [Data Structures for DNN Atomic Positions](#data-structures-for-dnn-atomic-positions)
21. [MC → Energy Call Flow and the N-body Scheme](#mc--energy-call-flow-and-the-n-body-scheme)

---

## 1. Overview

gRASPA uses an **i-PI socket protocol** to offload host-guest interaction energy calculations to an external machine-learning server (e.g. MACE via ASE). The C++ simulation engine acts as a **socket client**, sending atomic configurations over a UNIX domain socket and receiving predicted energies. This replaces (or supplements) classical force-field host-guest energy during Monte Carlo moves.

**Key design choice — N-body delta-query scheme:** The server maintains the current
accepted configuration of all adsorbates and the corresponding total energy `E_current`.
Per MC move, C++ sends one trial molecule's coordinates; the server returns:
```
ΔE = E(fw + all_ads + trial_mol) − E_current
```
On acceptance, C++ sends a COMMIT message; the server updates its stored configuration
and recomputes `E_current`. On rejection, no message is needed — server state already
mirrors C++ state.

This design correctly captures many-body contributions (adsorbate–adsorbate interactions
mediated by the ML model, framework polarization) that the previous 1-body scheme
`E_HG = E(fw + 1 mol) − E_fw − E_ads` missed entirely.

**Framework cache:** At startup, C++ sends the framework-only configuration
(`config_type = 0`). The server evaluates and caches `E_fw`. `E_current` is initialized
to `E_fw` (zero adsorbates). The framework cache is never invalidated during the run.

---

## 2. Directory Roles

| Directory | Role |
|-----------|------|
| `src_clean/` | **Baseline gRASPA source** with unfilled `###PATCH_SOCKET_*###` placeholders. Contains `ase_energy_client.h` (the full socket client) and `UseSocket` flag wiring in `data_struct.h`, `read_data.cpp`, and `main.cpp`. |
| `patch_Socket/` | **Patched gRASPA source** where all `###PATCH_SOCKET_*###` placeholders have been filled in with working socket code. This is the buildable version. |
| `socket-patch/Socket/` | **Patch snippets** as standalone `.txt` files. Each file contains the code block that replaces a corresponding `###PATCH_SOCKET_*###` marker. Useful as a reference for what was changed. |
| `gRASPA_files_for_Lucas/` | **Runtime artifacts**: simulation input files, molecule definitions, run scripts, example output, and a copy of the Python server + patched source for distribution. |

### Relationship between `src_clean` and `patch_Socket`

`src_clean` has comment markers like:
```cpp
//###PATCH_SOCKET_INSERTION###//
```

`patch_Socket` has those same locations filled in with actual code (marked `_PATCHED`):
```cpp
//###PATCH_SOCKET_INSERTION_PATCHED###//
if(SystemComponents.UseSocket)
{
    // ... actual socket energy code ...
}
```

The `socket-patch/Socket/PATCH_SOCKET_*.txt` files contain just the snippet code for each patch point.

---

## 3. Architecture

```
+--------------------------------------------------+
|  gRASPA MC Engine (C++/CUDA)                     |
|  - Generates trial configs on GPU                |
|  - cudaMemcpy positions to host                  |
|  - Calls DNN_Prediction_Move/Reinsertion/Total   |
+------------------+-------------------------------+
                   |
                   v
+--------------------------------------------------+
|  Socket Client (ase_energy_client.h)             |
|  - Wraps atoms into UCBox (MACE handles PBC)    |
|  - Sends positions via iPI protocol              |
|  - Receives energy (eV), converts to 10J/mol    |
+------------------+-------------------------------+
                   |  UNIX Domain Socket
                   |  /tmp/ase_ipi_socket
                   v
+--------------------------------------------------+
|  Python Server (ase_ipi_server_mace.py)          |
|  - Custom iPI protocol handler                   |
|  - MACE neural network potential                 |
|  - GPU-accelerated inference                     |
+--------------------------------------------------+
```

---

## 4. iPI Protocol Specification

### Transport
- **Socket type:** UNIX Domain (`AF_UNIX`, `SOCK_STREAM`) or INET TCP
- **Default path:** `/tmp/ase_ipi_socket` (hardcoded in `Socket` struct)
- **Max path length:** 108 bytes

### Message Format
- **Headers:** 12-byte ASCII strings, space-padded (e.g. `"STATUS      "`)
- **Integers:** 4-byte `int32_t` in **network byte order** (`htonl`/`ntohl`)
- **Floats:** Native `double` (IEEE 754, 8 bytes each)

### Handshake (once at startup)

```
Client                              Server
------                              ------
                    <-- STATUS (12 bytes)
NEEDINIT (12 bytes) -->
                    <-- INIT (12 bytes)
                    <-- cell[9 doubles]
                    <-- inv_cell[9 doubles]
[handshake_done = true]
```

### Species map exchange (once, immediately after handshake)

After `do_ipi_handshake()` completes, the C++ client sends a species map over the
wire so the server can build its `config_map` without relying on a side-channel file:

```
Client                              Server
------                              ------
SPECIESMAP (12-byte header) -->
n_species [int32, network order] -->
n_fw      [int32, network order] -->
for each species i:
  index [int32] -->
  len   [int32] -->
  symbol[len bytes] -->
```

The server uses `n_fw` plus the per-config `natoms` to infer molecule counts at
runtime (e.g. `n_mol = (natoms - n_fw) / DNN_Molsize`).

### Per-evaluation data exchange

> **Note:** The POSDATA payload extends the standard iPI format. Fields added by
> gRASPA are `config_type` (sent before the cell matrix), `n_mol`, `types`, and
> (for `config_type ≥ 200`) `mol_idx`.

### Standard format (`config_type < 200`)

```
Client                              Server
------                              ------
                    <-- STATUS
READY -->
                    <-- POSDATA
config_type [int32, network order] -->
cell[9 doubles] -->
inv_cell[9 doubles, zeroed] -->
n_mol [int32, network order] -->
natoms [int32, network order] -->
types[natoms × int32, network order] -->   (absent if natoms=0)
positions[3*natoms doubles] -->             (absent if natoms=0)
                    <-- STATUS
HAVEDATA -->
                    <-- GETFORCE
                    <-- FORCEREADY
                    <-- energy [1 double, eV]
                    <-- natoms [int32]
                    <-- forces[3*natoms doubles]
                    <-- virial[9 doubles]
                    <-- extras_len [int32]
                    <-- extras[extras_len bytes]
```

### Extended format (`config_type ≥ 200`) — N-body delta-query and commit

For all N-body messages, `mol_idx [int32]` is inserted immediately after `config_type`:

```
config_type [int32, network order] -->
mol_idx     [int32, network order] -->   ← NEW: −1 for insertions; mol index otherwise
cell[9 doubles] -->
inv_cell[9 doubles, zeroed] -->
n_mol [int32, network order] -->
natoms [int32, network order] -->
types[natoms × int32, network order] -->
positions[3*natoms doubles] -->
```

### `config_type` values

| Value | Name | Meaning |
|-------|------|---------|
| `0` | `CONFIG_FRAMEWORK` | Framework only; server caches `E_fw` |
| `1..N` | — | Adsorbate-only for component N (startup validation only) |
| `100+N` | — | Total (framework + adsorbate N); old 1-body path (startup validation) |
| `200+comp` | `DELTA_QUERY_BASE` | Return `ΔE = E(trial_config) − E_current` |
| `300` | `COMMIT_INSERT` | Append new mol; recompute `E_current` |
| `301` | `COMMIT_DELETE` | Swap-with-last delete `mol_idx`; recompute `E_current` |
| `302` | `COMMIT_MOVE` | Update `mol_idx` to new position; recompute `E_current` |
| `303` | `SYNC_FULL` | Replace entire component config; recompute `E_current` |
| `500` | `QUERY_TOTAL` | Return `E_current − E_fw_cached` |

**DELTA_QUERY** (200+comp): For insertion/move, `n_mol=1`, `natoms=DNN_Molsize` with trial
positions. For deletion, `n_mol=0`, `natoms=0` (no position data). Server returns ΔE as
the energy scalar in FORCEREADY.

**COMMIT messages** (300–303): Server returns dummy `0.0` energy via FORCEREADY; return
value is discarded by C++.

**SYNC_FULL** (303): `mol_idx` = adsorbate component index; `n_mol` = total molecule
count for this component; `natoms` = `n_mol × DNN_Molsize`; all molecules concatenated
in flat order.

Forces and virial are **read but not used** by gRASPA (protocol compliance only).

---

## 5. Core Implementation

**File:** `ase_energy_client.h` (present in both `src_clean/` and `patch_Socket/`)

### Socket Struct

```cpp
struct Socket {
    // Geometry
    Boxsize UCBox;                              // Unit cell box (sent directly to MACE)
    std::vector<Atoms> UCAtoms;                 // Atoms per component (unit cell)

    // Element mapping
    std::vector<std::string> ElementSymbolUsed;
    std::vector<int> Match_Element_PseudoAtom_order;

    // Connection state
    char socket_path[108] = "/tmp/ase_ipi_socket";
    int fd = -1;                                // File descriptor
    size_t natoms = 0;
    size_t DNN_Molsize = 0;                     // Atoms per adsorbate (excluding fictional)
    size_t nstep = 0;
    bool handshake_done = false;
};
```

> **Removed:** `ReplicaBox`, `ReplicaAtoms`, and `NReplicacell` fields are gone.
> MACE handles PBC natively; gRASPA sends `UCBox.Cell` directly. Energy caching
> (`cached_E_framework_ev`, `cache_valid`) has moved server-side — the Python server
> caches `E_fw` after the startup `PrimeFrameworkCache()` call.

### Key Methods

**Connection and setup:**

| Method | Purpose |
|--------|---------|
| `init(path, n_atoms)` | Set socket path and initial atom count |
| `connect_socket()` | Open UNIX socket, handshake, send species map, prime server cache, call SyncFull (if molecules present) |
| `send_positions(xyz, types, n, config_type, n_mol)` | Standard POSDATA exchange (config_type < 200) |
| `send_positions_extended(xyz, types, n, config_type, n_mol, mol_idx)` | Extended POSDATA exchange with `mol_idx` field (config_type ≥ 200) |
| `receive_energy(energy_ev)` | Read FORCEREADY response |
| `PredictFromSocket(xyz, types, n, config_type, n_mol)` | Send coords via standard format, get energy back (eV) |
| `PredictFromSocketExtended(xyz, types, n, config_type, n_mol, mol_idx)` | Send coords via extended format, get energy back (eV) |
| `PrimeFrameworkCache()` | Send framework-only config (`config_type=0`); server computes and caches E_fw, initializes E_current = E_fw |
| `send_species_map()` | Send element symbols over the wire immediately after handshake |
| `close_socket()` | Send EXIT command, close fd (call explicitly if needed; no destructor) |
| `Match_Element_PseudoAtom_with_model(PA)` | Map pseudo-atoms to element symbols |
| `CopyAtomsFromFirstUnitcell(...)` | Extract UC atoms, filter fictional sites |
| `WrapSuperCellAtomIntoUCBox(comp)` | PBC wrapping via fractional coords |

**N-body delta-query and commit (per MC move):**

| Method | Purpose |
|--------|---------|
| `QueryDelta(move_type, ads_comp, mol_idx, DNNEnergyConversion)` | Send trial config; receive `ΔE = E(trial) − E_current` in internal units. For deletion: `mol_idx` = mol to remove, no positions sent. For insertion: `mol_idx = −1`. For move: `mol_idx` = moving molecule. |
| `CommitInsert(ads_comp)` | Commit accepted insertion to server; server appends mol, recomputes E_current |
| `CommitDelete(ads_comp, mol_idx)` | Commit accepted deletion; server swap-with-last removes mol_idx, recomputes E_current |
| `CommitMove(ads_comp, mol_idx)` | Commit accepted translation/rotation/reinsertion; server updates mol_idx position, recomputes E_current |
| `SyncFull(ads_comp, n_mol, full_molsize, host_positions, consider_atom)` | Full configuration resync: send all N molecules for a component; server replaces stored config, recomputes E_current |
| `QueryTotal(DNNEnergyConversion)` | Return `E_current − E_fw_cached` in internal units (used for FxnMain total-energy check) |

**Legacy methods (kept for startup validation):**

| Method | Purpose |
|--------|---------|
| `Predict(ads_comp)` | Build combined fw+ads xyz array, call `PredictFromSocket(config_type=100+ads_comp)`; server returns 1-body HG energy. Used at startup for validation tests only. |
| `MCEnergyWrapper(comp, init, conv)` | Wrap UCAtoms into UCBox + Predict() + unit convert. Used at startup for validation tests only. |

### Robust I/O

`write_all()` and `read_all()` loop until all bytes are transferred:
```cpp
while (left > 0) {
    ssize_t r = read(fd, ptr, left);
    if (r == 0) { fprintf(stderr, "SOCKET CLOSED BY ASE\n"); exit(1); }
    if (r < 0)  { perror("read"); exit(1); }
    left -= r; ptr += r;
}
```

---

## 6. Integration Points Across the Codebase

### `data_struct.h`
- **Line ~887:** `#include "ase_energy_client.h"` (via patch)
- **Line ~1123:** `bool UseSocket = false;` in `Components` struct
- **Line ~1131:** `Socket DNN;` member in `Components`

### `read_data.cpp`
- **Lines 3100-3103:** Parses `DNN_Method Socket` from `simulation.input` → sets `UseSocket = true`
- **Lines 3107-3122:** Parses `DNNEnergyUnit` → sets conversion factor
- **Lines 3137+:** `ReadSocketModelParameters()` reads `DNNModelName` and `MaxDNNDrift`

### `main.cpp`
- **Lines 119-130:** Calls `ReadDNNModelSetup()` early in init
- **Lines 193-197:** Copies `UseSocket` flag into runtime `Components`
- **Lines 332-392 (patched):** Full socket setup:
  - Match elements to pseudo-atoms
  - Copy framework + adsorbate atoms into `UCAtoms`
  - `WrapSuperCellAtomIntoUCBox(0)` — wrap framework atoms into UCBox (explicit step)
  - `connect_socket()` — handshake, send species map over wire, prime server E_fw cache
  - First test `Predict()` call (1-body, startup validation only)
  - Second test with displaced atoms via `MCEnergyWrapper(1, false, ...)` (startup validation only)
- **After `CreateMolecule_InOneBox` (patched):** `SyncFull()` for each adsorbate component
  — primes server `ads_config` and recomputes `E_current = E(fw + initial_ads)` before
  the MC loop begins.

### `DNN_HostGuest_Energy_Functions.h`
All MC move types are handled via N-body delta queries:

| Move Type | Patch Name | What it does |
|-----------|-----------|--------------|
| INSERTION (CBMC) | `PATCH_SOCKET_INSERTION` | `cudaMemcpy` trial pos → `Check_DNNAtom_and_copy_pos_to_UCAtoms` → `QueryDelta(INSERTION, comp, -1)`. Result = DNN_New. |
| DELETION (CBMC) | `PATCH_SOCKET_DELETION` | `QueryDelta(DELETION, comp, mol_idx)`. **Negated**: `DNN_New = -QueryDelta(...)` (see sign convention). |
| TRANSLATION/ROTATION | `PATCH_SOCKET_SINGLE` | `QueryDelta(TRANSLATION, comp, mol_idx, new_pos)`. DNN_Old = 0. Returned ΔE = `E(new)−E_current` with old pos already in server state. |
| SINGLE_INSERTION | `PATCH_SOCKET_SINGLE` | `QueryDelta(INSERTION, comp, -1)`. DNN_Old = 0. |
| SINGLE_DELETION | `PATCH_SOCKET_SINGLE` | `QueryDelta(DELETION, comp, mol_idx)`. DNN_Old = 0. (No negation — SINGLE convention.) |
| REINSERTION | `PATCH_SOCKET_REINSERTION` | `QueryDelta(TRANSLATION, comp, mol_idx, new_pos)`. DNN_Old = 0. |
| Total energy | `PATCH_SOCKET_FXNMAIN` | Single `QueryTotal()` call; returns `E_current − E_fw_cached` summed over all adsorbates. |

### `mc_utilities.h`
Commit patches fire on acceptance, after GPU arrays are updated:

| Accept Function | Patch Name | What it does |
|-----------------|-----------|--------------|
| `AcceptTranslation` | `PATCH_SOCKET_COMMIT_MOVE` | `cudaMemcpy` accepted pos → `CommitMove(comp, mol_idx)` |
| `AcceptInsertion` | `PATCH_SOCKET_COMMIT_INSERT` | `cudaMemcpy` last mol slot → `CommitInsert(comp)` |
| `AcceptDeletion` | `PATCH_SOCKET_COMMIT_DELETE` | `CommitDelete(comp, mol_idx)` |

### `move_struct.h`
| Accept Function | Patch Name | What it does |
|-----------------|-----------|--------------|
| `ReinsertionMove::Acceptance` | `PATCH_SOCKET_COMMIT_REINSERTION` | `cudaMemcpy` accepted pos → `CommitMove(comp, mol_idx)` |

### `fxn_main.h`
- **Lines ~341-346:** `DNN_Prediction_Total()` called during `Check_Simulation_Energy()`
- **Lines ~491-500:** Reports DNN energy, stored classical HG energy, and the correction

### Helper: `Check_DNNAtom_and_copy_pos_to_UCAtoms()`
Filters atoms using `ConsiderThisAdsorbateAtom[]` boolean array (excludes fictional charge sites like TIP4P M-site) and copies positions from GPU-transferred buffer into `UCAtoms`.

---

## 7. Patching System

The codebase uses a text-marker patching approach. Each marker in `src_clean` corresponds to a patch file in `socket-patch/Socket/`:

**Energy evaluation patches (DNN_HostGuest_Energy_Functions.h):**

| Marker in `src_clean` | Patch File | Target File |
|-----------------------|-----------|-------------|
| `###PATCH_SOCKET_DATA_STRUCT_H###` | `PATCH_SOCKET_data_struct.h.txt` | `data_struct.h` |
| `###PATCH_SOCKET_READDATA###` | `PATCH_SOCKET_read_data.cpp.txt` | `read_data.cpp` |
| `###PATCH_SOCKET_READDATA_H###` | `PATCH_SOCKET_read_data.h.txt` | `read_data.h` |
| `###PATCH_SOCKET_MAIN_READMODEL###` | `PATCH_SOCKET_main.cpp.txt` | `main.cpp` |
| `###PATCH_SOCKET_MAIN_PREP###` | `PATCH_SOCKET_main.cpp.txt` | `main.cpp` |
| `###PATCH_SOCKET_POST_CREATEMOL###` | `PATCH_SOCKET_main.cpp.txt` | `main.cpp` |
| `###PATCH_SOCKET_CONSIDER_DNN_ATOMS###` | `PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt` | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_INSERTION###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_DELETION###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_SINGLE###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_REINSERTION###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_FXNMAIN###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |

**Commit patches (acceptance callbacks — N-body scheme):**

| Marker in `src_clean` | Patch File | Target File |
|-----------------------|-----------|-------------|
| `###PATCH_SOCKET_COMMIT_MOVE###` | `PATCH_SOCKET_mc_utilities.h.txt` | `mc_utilities.h` |
| `###PATCH_SOCKET_COMMIT_INSERT###` | `PATCH_SOCKET_mc_utilities.h.txt` | `mc_utilities.h` |
| `###PATCH_SOCKET_COMMIT_DELETE###` | `PATCH_SOCKET_mc_utilities.h.txt` | `mc_utilities.h` |
| `###PATCH_SOCKET_COMMIT_REINSERTION###` | `PATCH_SOCKET_move_struct.h.txt` | `move_struct.h` |

**Patch file naming convention:** `PATCH_SOCKET_<source_filename>.txt`. `patch.py` strips
the `PATCH_SOCKET_` prefix to determine the target source file. Multiple patch sections
can live in a single `.txt` file (each section headed by the marker name).

**To apply patches:** Replace each `//###PATCH_SOCKET_*###//` line with the corresponding snippet. The `patch_Socket/` directory contains the already-patched result.

---

## 8. Configuration

### `simulation.input` Required Settings

```
UseDNNforHostGuest    yes
DNN_Method            Socket
DNNEnergyUnit         eV          # or kJ_mol
MaxDNNDrift           100000      # threshold in internal units
DNNModelName          Socket      # identifier (not a file path for socket mode)
```

### Energy Unit Conversion

| Source Unit | Conversion Factor | Target Unit |
|-------------|-------------------|-------------|
| eV | 9648.53074992579265 | 10 J/mol (gRASPA internal) |
| kJ/mol | 100.0 | 10 J/mol (gRASPA internal) |

### Socket Path

Hardcoded to `/tmp/ase_ipi_socket` in the `Socket` struct default. The Python server uses `/tmp/<socket_name>` where `<socket_name>` is passed via `--socket` flag. **These must match.** Pass `--socket ase_ipi_socket` to the server to use the default path.

---

## 9. Energy Calculation Strategy

### N-body delta-query scheme (current)

The energy returned to gRASPA for each MC move is the **many-body marginal energy**:

```
ΔE = E(fw + all_N_ads + trial_mol) − E(fw + all_N_ads)   [insertion/move]
ΔE = E(fw + all_N_ads − mol_del)   − E(fw + all_N_ads)   [deletion]
```

This is physically exact: it accounts for all adsorbate–adsorbate interactions and
framework polarization captured by the ML model.

### Server-side state machine

The server maintains:
```
ads_config[comp]  = list of accepted molecule position arrays for component comp
E_current         = E(fw + all ads in ads_config), updated after every COMMIT
E_fw_cached       = E(fw alone), computed once at startup and never changed
```

**Per move (C++ → server):**
```
1. Prepare trial positions for 1 molecule in UCAtoms[comp]
2. QueryDelta(move_type, comp, mol_idx, DNNEnergyConversion)
   → send_positions_extended(config_type = 200+comp, mol_idx, trial_xyz, ...)
   → server computes E(trial_config), returns ΔE = E(trial_config) − E_current
3. Convert eV → 10 J/mol → DNN_E
```

**On acceptance (C++ → server, 1 additional socket call):**
```
CommitInsert(comp)        → config_type=300: server appends mol, recomputes E_current
CommitDelete(comp, idx)   → config_type=301: server swap-deletes idx, recomputes E_current
CommitMove(comp, idx)     → config_type=302: server updates idx, recomputes E_current
```

**On rejection:** No message sent. Server state unchanged = consistent with C++ state.

### Framework cache priming (at startup)

`connect_socket()` calls `PrimeFrameworkCache()` after sending the species map. This
sends `config_type=0` (framework only); the server computes and stores `E_fw_cached` and
initializes `E_current = E_fw_cached` (zero adsorbates).

After `CreateMolecule_InOneBox`, `SyncFull()` is called for each adsorbate component to
prime `ads_config` and bring `E_current` up to `E(fw + initial_ads)`.

### Per-move socket call count

| Move type | DELTA_QUERY calls | COMMIT calls (on accept) | Total (accept / reject) |
|-----------|-------------------|--------------------------|-------------------------|
| Translation/Rotation | 1 | 1 (COMMIT_MOVE) | 2 / 1 |
| Insertion (CBMC/single) | 1 | 1 (COMMIT_INSERT) | 2 / 1 |
| Deletion (CBMC/single) | 1 | 1 (COMMIT_DELETE) | 2 / 1 |
| Reinsertion | 1 | 1 (COMMIT_MOVE) | 2 / 1 |
| Total energy (FxnMain) | 1 (QUERY_TOTAL) | — | 1 |

Each DELTA_QUERY triggers one MACE forward pass on the server (trial_config). Each
COMMIT triggers one MACE forward pass to recompute `E_current`. The framework cache
is always reused.

### Mol-index ordering invariant

Server mol ordering for each component mirrors C++ `d_a[comp]`:
- **Insert:** both append new molecule at index N (new last slot)
- **Delete mol_idx:** both use swap-with-last then decrement N, matching
  `Update_deletion_data_Parallel` in gRASPA's GPU kernel

Violation of this invariant would cause the server to compute energies for wrong
molecule configurations on subsequent moves.

### Previous scheme (1-body, now startup-validation only)

The old scheme evaluated the interaction energy for one adsorbate in isolation:
```
E_HG = E(fw + 1 mol) − E_fw_cached − E(1 mol alone)
```
This is computed by `Predict(ads_comp)` which makes 2 socket calls per evaluation
(config_type=100+comp for the combined system, config_type=comp for the adsorbate alone).
It is still called at startup for validation tests but is no longer used in the MC loop.

---

## 10. Supercell Replication

> **Supercell replication has been removed.** The socket client now sends `UCBox.Cell`
> directly to MACE with `pbc=True`. MACE handles periodic boundary conditions natively
> using its own neighbor-list construction with full lattice periodicity.

**User constraint:** Each UCBox dimension must exceed `2 × r_cutoff(MACE)`. This is
the same constraint already enforced by gRASPA for the classical force field; no
separate `SocketReplicaCell` input keyword is needed.

At startup, `WrapSuperCellAtomIntoUCBox(0)` is called explicitly for the framework
(comp=0) to ensure all framework atom coordinates are within the unit cell before the
first socket call. Previously this wrap happened implicitly inside
`ReplicateAtomsPerComponent`; it is now an explicit step in the PREP block.

---

## 11. Python Server

**File:** `ase_ipi_server_mace.py`

```bash
python ase_ipi_server_mace.py --socket ase_ipi_socket --model /path/to/model.pt
```

- Loads a MACE ML potential (or other ASE-compatible calculator)
- Implements the iPI wire protocol directly (not via ASE's SocketIOCalculator)
- Routes configurations by `natoms` to assign correct element symbols
- Supports CUDA GPU or CPU (`--device`)
- Socket path becomes `/tmp/ase_ipi_socket` (matching the C++ client default)

### Server Startup Sequence

The server startup is carefully sequenced to avoid deadlocks with gRASPA:

```
1. Load ML model
2. Bind + listen on UNIX socket          (socket file now exists)
3. Accept connection from gRASPA
4. do_handshake()                        (STATUS → NEEDINIT → INIT)
5. recv_species_map()                    (species info sent over wire by gRASPA)
6. recv_prime_framework()               (config_type=0 call; compute and cache E_fw;
                                          initialize E_current = E_fw, ads_config = {})
7. serve() main loop                     (STATUS → READY → POSDATA → ...)
   - config_type=303: SyncFull → set ads_config[comp], recompute E_current
   - config_type=200+comp: delta query → return E(trial_config) − E_current
   - config_type=300: commit insert → append mol, recompute E_current
   - config_type=301: commit delete → swap-with-last remove, recompute E_current
   - config_type=302: commit move → update position, recompute E_current
   - config_type=500: query total → return E_current − E_fw_cached
```

Species information arrives over the wire immediately after the handshake
(`recv_species_map()`). There is no species file to poll. On reconnection, the
species data, framework cache, and adsorbate config are re-established via the same
startup sequence (gRASPA calls `SyncFull` before the MC loop begins).

### Key Functions

| Function | Purpose |
|----------|---------|
| `do_handshake(conn, cell)` | Perform iPI handshake (STATUS/NEEDINIT/INIT) |
| `recv_species_map(conn)` | Receive element symbols; build `config_map` |
| `recv_prime_framework(conn, calc, config_map)` | Handle config_type=0; compute and cache E_fw; set E_current = E_fw |
| `serve(conn, calc, config_map, cell)` | Main protocol loop; dispatch by config_type; manage ads_config and E_current |
| `handle_delta_query(comp, mol_idx, trial_xyz)` | Compute E(trial_config) − E_current; return ΔE |
| `handle_commit_insert(comp, xyz)` | Append mol to ads_config[comp]; recompute E_current |
| `handle_commit_delete(comp, mol_idx)` | Swap-with-last remove from ads_config[comp]; recompute E_current |
| `handle_commit_move(comp, mol_idx, xyz)` | Update ads_config[comp][mol_idx]; recompute E_current |
| `handle_sync_full(comp, n_mol, xyz)` | Replace ads_config[comp] entirely; recompute E_current |
| `handle_query_total()` | Return E_current − E_fw_cached |

### Typical Launch Script (`gcmc_mace.bash`)

```bash
SOCKET_NAME="ase_ipi_socket"
SOCKET_PATH="/tmp/${SOCKET_NAME}"

rm -f "${SOCKET_PATH}"                                # Clean old socket
python ase_ipi_server_mace.py \
    --socket "${SOCKET_NAME}" \
    --model /path/to/model.pt &                       # Start server

# Wait for model to load (socket appears only after torch.load completes)
until [ -S "${SOCKET_PATH}" ]; do sleep 1; done

./nvc_main.x                                          # Start gRASPA
kill %1                                               # Stop server
rm -f "${SOCKET_PATH}"                                # Cleanup
```

---

## 12. Runtime Workflow

```
Server side:
  S1. Load ML model
  S2. Bind + listen on /tmp/ase_ipi_socket
  S3. Accept connection (blocks until gRASPA connects at step C7)
  S4. do_handshake() — STATUS → NEEDINIT → INIT
  S5. recv_species_map() — species info arrives over wire
  S6. recv_prime_framework() — config_type=0; compute and cache E_fw;
                               initialize E_current = E_fw, ads_config = {}
  S7. serve() main loop — handles all POSDATA requests

gRASPA side:
  C1.  Parse simulation.input -> UseSocket = true, DNNEnergyConversion = 9648.53...
  C2.  ReadSocketModelParameters() -> DNNModelName, MaxDNNDrift
  C3.  Match pseudo-atoms to element symbols
  C4.  Copy framework atoms into DNN.UCAtoms[0]
  C5.  Copy adsorbate template into DNN.UCAtoms[1]
  C6.  WrapSuperCellAtomIntoUCBox(0) -> wrap framework into UCBox (explicit step)
  C7.  connect_socket():
         a. do_ipi_handshake() (server at S4)
         b. send_species_map() over wire (server at S5)
         c. PrimeFrameworkCache() -> config_type=0; server caches E_fw (server at S6)
  C8.  Startup validation: first test Predict() call (1-body, logs to stdout)
  C9.  Startup validation: second test via MCEnergyWrapper(1, false, ...)
  C10. CreateMolecule_InOneBox() -> places initial adsorbate molecules
  C11. SyncFull() for each adsorbate component:
         - config_type=303; sends all initial mol positions
         - Server sets ads_config[comp], recomputes E_current = E(fw + initial_ads)
  C12. Check_Simulation_Energy() -> calls QueryTotal() -> server returns E_current - E_fw
  C13. Begin MC simulation loop (per move):
         a. Select MC move type and molecule
         b. Generate trial configuration (GPU)
         c. cudaMemcpy trial positions to host
         d. Check_DNNAtom_and_copy_pos_to_UCAtoms() -> filter + pack into UCAtoms[comp]
         e. WrapSuperCellAtomIntoUCBox(comp) -> PBC-wrap into UCBox
         f. DNN_Prediction_Move() ->
              QueryDelta(move_type, comp, mol_idx, DNNEnergyConversion)
                → config_type=200+comp with mol_idx and trial positions
                → server returns ΔE = E(trial_config) − E_current
              DNN_Replace_Energy(): store classical HG terms, zero them, set DNN_E = ΔE
              Check_DNN_Drift(): if |DNN_E - classical_HG_sum| > MaxDNNDrift → reject
         g. Metropolis acceptance/rejection
         h. If ACCEPTED:
              Update GPU arrays (Update_deletion_data_Parallel, etc.)
              CommitXxx() -> config_type=300/301/302 with accepted position
                → server updates ads_config[comp], recomputes E_current
            If REJECTED:
              No socket message; server state unchanged
  C14. Check_Simulation_Energy() calls QueryTotal() -> server returns E_current - E_fw
       (used for periodic energy drift monitoring)
  C15. Process exits -> OS reclaims socket fd
```

---

## 13. Error Handling

| Scenario | Behavior |
|----------|----------|
| Socket connection fails | Returns -1, prints `perror()` |
| `PredictFromSocket` called with `fd < 0` | Auto-connects via `connect_socket()`; exits fatally if that fails |
| Server closes socket mid-transfer | Prints `"SOCKET CLOSED BY ASE"`, calls `exit(EXIT_FAILURE)` |
| Read/write returns partial data | Retries in loop until all bytes transferred |
| Wrong header received | Prints unexpected header to stderr |
| DNN drift exceeds threshold | Move rejected, event logged to `DNN/Outliers_*.data` |
| Missing config parameters | Throws `std::runtime_error` with descriptive message |
| Normal shutdown | OS reclaims fd on process exit; call `close_socket()` explicitly for a clean EXIT to the server |

**Design philosophy:** Fail loudly on communication errors rather than silently producing wrong energies.

---

## 14. Known Limitations

1. **Two socket calls per energy evaluation** (E_total + E_adsorbate) after framework cache is warm — the first `Predict()` still requires 3 calls (adds E_framework)
2. **Synchronous/blocking I/O** — no pipelining or async evaluation
3. **Socket path hardcoded** — changing requires source modification and recompilation
4. **Single socket connection** — one ML server per simulation
5. **Rigid framework only** — `main.cpp` throws if framework is not rigid or has multiple components
6. **`natoms`-based config identification** — the server routes each call by `natoms` via `symbol_map`; if two config types happen to have the same atom count (e.g., `n_fw == n_ads`), the routing is ambiguous
7. **Forces/virial unused** — read from server for protocol compliance but discarded
8. **Server reconnection** — the server accepts new connections after a client disconnects, but the C++ client has no reconnect logic (if the server dies mid-simulation, gRASPA terminates)

### Species File Format (current)

```
FRAMEWORK <n_fw>
<fw_symbols...>
ADSORBATE <n_ads_comp1>
<ads_comp1_symbols...>
ADSORBATE <n_ads_comp2>      # one block per adsorbate component (mixture support)
<ads_comp2_symbols...>
```

The server builds `symbol_map` entries for each adsorbate block: `n_ads_i`, `n_fw + n_ads_i`, and `n_fw`.

---

## 15. Known Issues and Future Work

The following issues were identified in code review. Items marked [FIXED] have been addressed; all others remain open.

### Recently Fixed

These P0 correctness bugs have been resolved in the current patched source (`patch_Socket/`):

- **[FIXED] `DNN_Molsize` accumulated across adsorbate components.** `CopyAtomsFromFirstUnitcell()` was adding to `DNN_Molsize` rather than assigning it, so simulating multiple adsorbate components would silently corrupt the stored molecule size. (`src_clean/ase_energy_client.h`, ~line 607.)
- **[FIXED] `PATCH_SOCKET_FXNMAIN` hardcoded component index.** The total-energy loop called `MCEnergyWrapper` with a literal `comp=1` rather than the loop variable, so only the first adsorbate component was ever evaluated in `Check_Simulation_Energy()`. (`socket-patch/Socket/PATCH_SOCKET_FXNMAIN block`.)
- **[FIXED] `update_i` incremented unconditionally in `Check_DNNAtom_and_copy_pos_to_UCAtoms`.** The counter was incremented for every atom in the loop, including atoms excluded by the `ConsiderThisAdsorbateAtom` filter, producing gaps in the packed output array and incorrect atom positions sent to the ML server. (`socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt`, `Check_DNNAtom_and_copy_pos_to_UCAtoms` function.)

---

### Memory Safety

**M1. No destructor on `Socket` struct — all heap members are leaked.**
The `Socket` struct has no destructor. All `malloc`'d members — `UCAtoms[*].pos`, `UCAtoms[*].Type`, `UCBox.Cell`, `UCBox.InverseCell` — are never freed. Currently harmless because the process exits immediately after the simulation, but makes sanitizer-based debugging impossible and will become a real leak if `Socket` is ever reused within a longer-lived process.
- File: `src_clean/ase_energy_client.h`, `struct Socket` (line 25); allocation sites at lines 572–573, 578–579.
- Note: `ReplicaAtoms[*].pos`, `ReplicaAtoms[*].Type`, `ReplicaBox.Cell`, `ReplicaBox.InverseCell` are no longer present (removed with supercell replication machinery).

**M2. Double-`malloc` of `UCBox.InverseCell` in `GenerateUCBox()`.**
`GenerateUCBox()` allocates `UCBox.InverseCell` via `malloc` (line 579), then immediately passes `&UCBox.InverseCell` to `inverse_matrix()` (line 584), which overwrites the pointer with a fresh `malloc` allocation (line 493) — leaking the first 72 bytes on every call.
- File: `src_clean/ase_energy_client.h`, lines 576–585 (`GenerateUCBox`) and lines 487–503 (`inverse_matrix`).

**[FIXED / REMOVED] M3. Double-`malloc` of `ReplicaBox.Cell`/`ReplicaBox.InverseCell` in `GenerateReplicaCells(Allocate=true)`.**
`GenerateReplicaCells()` and `ReplicaBox` have been removed entirely. This issue no longer exists.

**[FIXED / REMOVED] M4. `GenerateReplicaCells(Allocate=false)` may write out-of-bounds if `DNN_Molsize` increases.**
`GenerateReplicaCells()` and `ReplicateAtomsPerComponent()` have been removed entirely. This issue no longer exists.

**M5. No `malloc` failure checks anywhere.**
Every `malloc` call in `ase_energy_client.h` (lines 493, 572, 573, 658, 659, 701, 702, 708, 709) is used without checking for a `NULL` return. On allocation failure the code will dereference a null pointer and segfault rather than printing a diagnostic.
- File: `src_clean/ase_energy_client.h` (multiple sites listed above).

---

### Performance

**P1. Three `std::vector<double>` heap allocations per `Predict()` call.**
`Predict()` allocates `xyz_total` (line 397), `xyz_framework` (line 419, first call only), and `xyz_adsorbate` (line 432) as local `std::vector<double>` objects on every invocation. In a typical GCMC run `Predict()` is called millions of times. These buffers should be pre-allocated as members of `Socket` and reused.
- File: `src_clean/ase_energy_client.h`, lines 390–445.

**[FIXED / REMOVED] P2. Framework coordinates are re-copied into `xyz_total` on every call.**
`ReplicaAtoms` has been removed. `Predict()` now builds `xyz_total` from `UCAtoms[0]`
on every call; the framework copy overhead is the same (P2 still exists in principle but
applies to `UCAtoms[0]` instead of the removed `ReplicaAtoms[0]`). The underlying
performance opportunity (caching `xyz_framework`) remains open as a future optimization.

**P3. `send_positions()` issues multiple separate `write()` syscalls per send.**
`send_positions()` makes separate `write_all()` calls for each field: `send_int32(config_type)`, `write_all(UCBox.Cell, ...)`, `write_all(inv_cell, ...)`, `send_int32(n_mol)`, `send_int32(n_atoms)`, `write_all(types_net, ...)`, and `write_all(xyz, ...)`. Batching these into a single `writev()` or a pre-assembled contiguous buffer would reduce context-switch overhead at high call rates.
- File: `src_clean/ase_energy_client.h`, `send_positions()` function.

**[FIXED / REMOVED] P4. `ReplicateAtomsPerComponent()` recomputes all 27 replica images for the framework on every MC move.**
`ReplicateAtomsPerComponent()` and `GenerateReplicaCells()` have been removed entirely. This issue no longer exists.

**P5. Python server reconstructs an ASE `Atoms` object on every evaluation.**
Inside the `serve()` loop, `Atoms(symbols=..., positions=..., cell=..., pbc=True)` is called on every server-side evaluation (line 218 of `ase_ipi_server_mace.py`). The `Atoms` object and its calculator attachment should be created once during setup and updated with `atoms.set_positions()` and `atoms.set_cell()` on subsequent calls to avoid repeated Python object allocation.
- File: `ase_ipi_server_mace.py`, line 218.

**P6. Per-move `malloc`/`free` of `temp_pos` in all four patch blocks.**
Each of the INSERTION, DELETION, SINGLE, and REINSERTION patch blocks allocates `temp_pos` via `malloc` at the start and `free`s it at the end of the block (e.g., `PATCH_SOCKET_INSERTION` lines 19 and 33). This per-move allocation occurs millions of times during a simulation. A single pre-allocated buffer stored in `SystemComponents` (or as a `Socket` member) and reused across moves would eliminate this overhead.
- File: `socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt`, INSERTION block (lines 19, 33), DELETION block (lines 40, 44), SINGLE block (lines 54, 69), REINSERTION block (lines 76, 85).

**P7. Unconditional debug `printf`/`std::cout` with `fflush` inside the hot path.**
`PredictFromSocket()` prints every atom position for the first five atoms and flushes stderr on every call (lines 332–337 of `ase_energy_client.h`). `Predict()` prints three energy values to stdout on every call (lines 441–443). Both functions are called on every MC move; at millions of moves per simulation these prints are a significant I/O bottleneck. All such output should be gated behind a verbosity flag or `#ifdef DEBUG`.
- File: `src_clean/ase_energy_client.h`, lines 332–337 (`PredictFromSocket`), lines 428 and 441–443 (`Predict`).

---

### Correctness and Flexibility

**C1. `ConsiderThisAdsorbateAtom` is a single shared array applied to all adsorbate components.**
`CopyAtomsFromFirstUnitcell()` and all patch blocks use a single `ConsiderThisAdsorbateAtom` boolean array (length = `Moleculesize` of whichever component was set up first). If a second adsorbate component has a different molecule size, the same array is applied to it, risking out-of-bounds reads or incorrect atom filtering. Each adsorbate component should have its own filter array.
- File: `src_clean/ase_energy_client.h`, line 594 (`CopyAtomsFromFirstUnitcell` signature); `socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt`, all patch blocks.

**C2. `WrapSuperCellAtomIntoUCBox()` computes bond distances into a vector that is never used.**
The function populates the local `Bonds` vector with inter-atom distances (lines 741–745) but `Bonds` is never read, returned, or checked. The bond-distance code is dead. It should either be completed (the comment suggests it was intended to detect molecules split across the periodic boundary) or removed to avoid confusion.
- File: `src_clean/ase_energy_client.h`, lines 725–752, particularly lines 727 and 741–745.

**C3. `NComponents.y` meaning is undocumented in the rigid-framework guard.**
The initialization check `Vars.SystemComponents[a].NComponents.y != 1` at `PATCH_SOCKET_main.cpp.txt` line 31 gates on `.y` without any comment explaining what `.y` represents. It is unclear whether `.y` encodes "number of semi-flexible framework copies", "number of framework molecule types", or something else. The intent of the check should be documented inline.
- File: `socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`, line 31.

**C4. `Predict(ads_comp = 1)` default parameter silently evaluates component 1.**
The `Predict()` method signature `double Predict(size_t ads_comp = 1)` allows callers to omit the component index, which silently defaults to component 1. In a multi-component mixture, a caller that forgets to pass the component index will always evaluate the first adsorbate component regardless of which molecule is actually being moved. Removing the default would force explicit specification at every call site.
- File: `src_clean/ase_energy_client.h`, line 390.

**[FIXED / REMOVED] C5. Species file writes replica atom counts rather than unit-cell atom counts.**
`WriteSpeciesFile()` and the species file mechanism have been removed. Species information
is now sent directly over the wire via `send_species_map()` using `UCAtoms[0].size`
(unit-cell count). This issue no longer exists.

**C6. `cache_valid` framework-energy caching assumes a permanently rigid framework.**
The `cache_valid` flag in `Socket` is set to `true` after the first `E_framework` evaluation and never cleared. If flexible framework support is added in the future, the cached energy will be stale on every move where the framework geometry changes. The rigid-framework assertion at initialization (enforced by the `NComponents.y` check in `PATCH_SOCKET_main.cpp.txt` line 31) guards against this for now, but the dependency between the cache logic and the rigidity assumption should be documented explicitly.
- File: `src_clean/ase_energy_client.h`, lines 46–48 (cache members), lines 416–427 (caching logic).

---

## 16. Changelog

All changes are listed in reverse chronological order (most recent first). Items marked **[PENDING]** are included in this release but may require further testing or validation.

---

### N-body socket HG energy scheme

**Date:** 2026-02-25
**Files:** `src_clean/ase_energy_client.h`, `src_clean/mc_utilities.h`,
`src_clean/move_struct.h`, `src_clean/main.cpp`,
`socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt`,
`socket-patch/Socket/PATCH_SOCKET_mc_utilities.h.txt` *(new)*,
`socket-patch/Socket/PATCH_SOCKET_move_struct.h.txt` *(new)*,
`socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`,
`socket-patch/Socket/PATCH_SOCKET_data_struct.h.txt`

Upgraded the socket energy scheme from 1-body (single adsorbate vs bare framework) to
N-body (true many-body marginal energy accounting for all adsorbates and polarization
effects). See Section 9 for full design and CHANGELOG.md for details.

---

### Remove SocketReplicaCell: send UCBox directly to MACE

**Date:** 2026-02-24
**Files:** `src_clean/ase_energy_client.h`, `socket-patch/Socket/PATCH_SOCKET_read_data.cpp.txt`, `socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`

The supercell replication mechanism (`ReplicaAtoms`, `ReplicaBox`, `NReplicacell`,
`GenerateReplicaCells`, `ReplicateAtomsPerComponent`) has been removed from the socket
path. gRASPA now sends unit cell atoms directly with `UCBox.Cell` and `pbc=True`. MACE
handles periodic boundary conditions natively.

**Root cause / motivation:** The replica machinery introduced a coordinate-system
mismatch (framework atoms in ReplicaBox coords, adsorbate in UCBox coords). The
`SocketReplicaCell` keyword created a user-configurable parameter whose correct value
depends on ML model cutoff — but gRASPA already enforces the same constraint for the
classical force field, so a separate keyword is redundant.

**New user constraint:** Each UCBox dimension must exceed `2 × r_cutoff(MACE)`. This
is identical to the classical force-field cutoff constraint; no separate input keyword
is needed.

1. **`src_clean/ase_energy_client.h`** — Removed `ReplicaBox`, `ReplicaAtoms`,
   `NReplicacell` fields from `Socket` struct. Removed `GenerateReplicaCells()` and
   `ReplicateAtomsPerComponent()`. Updated `send_positions()` to use `UCBox.Cell`.
   Updated `PrimeFrameworkCache()`, `Predict()`, and `MCEnergyWrapper()` to use
   `UCAtoms` instead of `ReplicaAtoms`.
2. **`PATCH_SOCKET_read_data.cpp.txt`** — Removed `SocketReplicaCell` keyword parsing,
   `found_replica` boolean, and the `runtime_error` guard.
3. **`PATCH_SOCKET_main.cpp.txt`** — PREP section no longer calls
   `GenerateReplicaCells(true)`. Now explicitly calls `WrapSuperCellAtomIntoUCBox(0)`
   for the framework before connecting.

---

### Fix endianness bug in types array

**Date:** 2026-02-24
**Files:** `src_clean/ase_energy_client.h`

`send_positions()` was writing the per-atom types array as native little-endian integers.
The Python server reads them with `struct.unpack(f"!{natoms}i", ...)` (network byte order
= big-endian). This caused type indices to be byte-swapped: type index 4 (Zn) was
received as 0x04000000 = 67108864, triggering an "ERROR: type index 67108864 not in
species map" crash.

All other int32 fields in the protocol already used `send_int32()` with `htonl` — the
types array was the only field missing the byte-swap.

Fixed by replacing `write_all(types, ...)` with a byte-swap loop into a temporary
`types_net` vector before `write_all`.

---

### Documentation: Added Section 17 — Supercell, PBC, and energy decomposition correctness

**Date:** 2026-02-24
**Files:** `iPI_SOCKET_DEVELOPER_GUIDE.md`

Added a new section covering the physical correctness of the pipeline:
- How MACE receives atoms (pbc=True with supercell cell matrix)
- The SocketReplicaCell size correctness criterion and common MACE cutoff values
- Why the three-call energy decomposition (E_total - E_framework - E_adsorbate) is exact under PBC
- Cell matrix row-major / lower-triangular storage convention and how ReplicaBox.Cell is formed
- How to validate the pipeline using `validate_socket_energy.py`

---

### [PENDING] Dynamic adsorbate count in Python server

**Date:** 2026-02-24
**Files:** `ase_ipi_server_mace.py`

The Python server now builds the adsorbate symbol list dynamically from the `natoms` count received in each POSDATA message rather than using a fixed-length list determined at startup. For total-configuration evaluations (config_type >= 100), the number of adsorbate molecules is computed at runtime as:

```
n_mols = (natoms - n_fw) / n_ads_per_mol
```

This enables correct handling of variable adsorbate occupancy as the GCMC simulation progresses (molecule count changes on every accepted insertion or deletion).

---

### **[SUPERSEDED]** NReplicacell now read from simulation.input

**Date:** 2026-02-24
**Files:** `socket-patch/Socket/PATCH_SOCKET_read_data.cpp.txt`, `socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`

> **This change was superseded by the "Remove SocketReplicaCell" entry above.**
> The configurable `SocketReplicaCell` keyword was implemented but then removed in the
> same session when the entire supercell replication mechanism was eliminated. The
> `SocketReplicaCell` keyword should not appear in any current simulation input files.

~~The supercell replication factor is now a configurable simulation input parameter rather
than a hardcoded constant. A new keyword `SocketReplicaCell nx ny nz` is parsed from
`simulation.input`.~~

---

### Protocol: config_type routing replaces natoms routing

**Date:** 2026-02-24
**Files:** `src_clean/ase_energy_client.h`, `ase_ipi_server_mace.py`

Configuration identification between the C++ client and Python server has been redesigned. Previously the server used the `natoms` count received in the POSDATA message to determine which element symbol list to apply. This is ambiguous when two configuration types have the same atom count (e.g., when the framework and a single adsorbate molecule contain the same number of atoms, or when two adsorbate species have equal molecule sizes).

**New design:** A `ConfigType` enum is defined in `ase_energy_client.h`:

```cpp
// CONFIG_FRAMEWORK = 0
// Adsorbate-only for component i = i  (1, 2, ...)
// Total (framework + adsorbate i) = 100 + i  (101, 102, ...)
```

`send_positions()` now transmits a 4-byte `int32_t` `config_type` value immediately before the cell matrix in every POSDATA exchange. `PredictFromSocket()` and `Predict()` pass the correct `config_type` for each of the three evaluations performed per `Predict()` call.

On the Python server side, `config_map` (keyed by `config_type` integer) replaces `symbol_map` (keyed by `natoms`). At startup, the server prints a validation table mapping each expected `config_type` to its atom count and symbol list. Natoms is retained as a secondary sanity check and triggers a warning if the received count does not match the expected value for the given `config_type`.

**Rationale:** Unambiguous routing is required for correct multi-component mixture simulations and for any framework where `n_fw == n_ads`.

---

### Fix: DNN_Molsize accumulation bug in CopyAtomsFromFirstUnitcell

**Date:** 2026-02-24
**Files:** `src_clean/ase_energy_client.h`

`CopyAtomsFromFirstUnitcell()` was using `DNN_Molsize += 1` to count atoms as it iterated over each adsorbate component. Because `DNN_Molsize` was a persistent member variable and was never reset between component calls, the accumulated total for component 2 incorrectly included the atom count from component 1. Subsequent components therefore had inflated sizes stored in `UCAtoms[comp].size`.

**Fix:** A local variable `dnn_size = 0` is used to count atoms within the loop body. After the loop for each component, `DNN_Molsize` is assigned (not incremented) from `dnn_size`. Each component's `UCAtoms[comp].size` is now set correctly and independently.

---

### Fix: PATCH_SOCKET_FXNMAIN hardcoded comp=1

**Date:** 2026-02-24
**Files:** `socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`

The total-energy patch block (`PATCH_SOCKET_FXNMAIN`) used `size_t comp = 1` as a fixed assignment rather than a loop variable. As a result, `MCEnergyWrapper` was called with `comp=1` for every molecule regardless of which adsorbate component was being summed, and all adsorbate components beyond the first were silently ignored in `Check_Simulation_Energy()`.

**Fix:** An outer `for(comp = 1; comp < NComponents.x; comp++)` loop now covers all adsorbate components. The `MCEnergyWrapper` call uses the loop variable.

---

### Fix: update_i bug in Check_DNNAtom_and_copy_pos_to_UCAtoms

**Date:** 2026-02-24
**Files:** `socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt`

In the atom-filtering and copy loop inside `Check_DNNAtom_and_copy_pos_to_UCAtoms`, the dense-packing write index `update_i` was incremented unconditionally for every atom in the loop — including atoms excluded by the `ConsiderThisAdsorbateAtom[i]` predicate. This caused gaps in the packed output array: excluded atoms left `update_i` advanced, so the next real atom was written past the correct position, and the positions array sent to the ML server contained uninitialized memory at the skipped slots.

**Fix:** `update_i` is incremented only inside the `if(ConsiderThisAdsorbateAtom[i])` block, ensuring contiguous dense packing of accepted atoms.

---

### New caching strategy: framework cached, adsorbate always recomputed

**Date:** 2026-02-24
**Files:** `src_clean/ase_energy_client.h`

Previously, `Predict()` cached both `E_framework` and `E_adsorbate` after the first call. Caching `E_adsorbate` is incorrect: the adsorbate configuration changes on every accepted MC move (translation, rotation, insertion, deletion). Stale cached adsorbate energies would cause incorrect acceptance probabilities for all subsequent moves.

**New behavior:** `Predict(ads_comp)` makes up to 3 socket calls:

1. `E_total` — always computed fresh (framework + adsorbate combined)
2. `E_framework` — computed once and cached; reused on all subsequent calls (`cached_E_framework_ev`, `cache_valid` flag)
3. `E_adsorbate` — always computed fresh

The returned interaction energy is `E_total - cached_E_framework_ev - E_adsorbate_ev`.

The `Socket` struct now documents this intent explicitly in the inline comment for the cache members.

---

### New: WriteSpeciesFile writes one ADSORBATE block per component

**Date:** 2026-02-24
**Files:** `src_clean/ase_energy_client.h`

`WriteSpeciesFile()` previously wrote a single `ADSORBATE` block combining element symbols for all adsorbate components into one section. This prevented the Python server from correctly distinguishing between adsorbate types in mixture simulations.

**Fix:** `WriteSpeciesFile()` now iterates over all adsorbate components (indices 1 through `NComponents - 1`) and writes a separate `ADSORBATE <n> <symbols...>` block for each. The species file format is described in the Known Limitations section (Section 14).

---

### New: Python server multi-component adsorbate support

**Date:** 2026-02-24
**Files:** `ase_ipi_server_mace.py`

`parse_species_file()` previously returned a flat `(fw_symbols, ads_symbols)` pair assuming a single adsorbate type. It now returns `(fw_symbols, ads_groups)` where `ads_groups` is a list of symbol lists, one entry per adsorbate component parsed from the species file.

`serve()` and `main()` have been updated to accept and handle the list of adsorbate symbol groups, constructing separate symbol-list entries in `config_map` for each adsorbate component index.

---

### Confirmed: INSERTION and DELETION patches are correct as written

**Date:** 2026-02-24
**Files:** `socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt`

An initial review flagged two potential bugs in the INSERTION and DELETION patch blocks. Investigation of the surrounding MC machinery confirmed these were not bugs:

- **INSERTION uses `Sims.Old.pos`:** `Prepare_DNN_InitialPositions()` copies the trial (new) positions into `Sims.Old` before the patch block executes. `Sims.Old.pos` therefore holds the correct trial configuration at the point the socket call is made.
- **DELETION uses `DNN_New`:** Only `DNN_New` is declared in the deletion block. The energy sign convention is handled through Rosenbluth weight division in the outer MC acceptance criterion rather than by negation at the socket layer. `DNN_New` is the correct variable.

Any changes to these two patch blocks made during this session were reverted.

---

## 17. Supercell, PBC, and Energy Decomposition Correctness

This section explains the physical and numerical basis for the correctness of the socket energy pipeline, covering how the ML model sees the system, what supercell size is required, why the three-call energy decomposition is exact, and how the cell matrix is stored.

---

### 17.1 How MACE Receives Atoms: pbc=True with UCBox Cell

The Python server creates the ASE `Atoms` object for each evaluation as:

```python
atoms = Atoms(symbols=symbols, positions=positions, cell=UCBox.Cell, pbc=True)
```

The consequences of this construction are:

- The system is treated as **periodic at the unit-cell level**, not as a finite cluster. There are no Dirichlet (open) boundaries.
- MACE builds its neighbor list using the **unit cell matrix** (`UCBox.Cell`) as the periodic repeat unit. Every atom in the unit cell sees periodic images of the full unit cell tiling all of space.
- This is physically equivalent to an **infinite crystal of unit cells**: the (0,0,0) cell is surrounded by image cells in all directions, each a translated copy of the full framework+adsorbate configuration.
- The framework atoms at the edges of the unit cell are therefore treated as connected to the framework atoms at the opposite edge (through the periodic image), which is the correct physical boundary condition for a crystalline host.

There are **no isolated-cluster or vacuum-boundary artifacts** in the energies produced by this setup, provided the unit cell is large enough (see Section 17.2).

---

### 17.2 Why UCBox Must Be Large Enough

The correctness criterion for the unit cell size is:

```
For each spatial dimension i:
  UCBox_length[i]  >  2 × r_cutoff(MACE)
```

**Why this criterion is necessary:**

With `pbc=True`, the adsorbate positioned in the (0,0,0) unit cell interacts not only with the framework in the same cell but also with periodic images of **itself** located in adjacent cell repetitions. These adsorbate self-images are unphysical in a dilute-loading GCMC simulation: in reality only one (or a few) adsorbate molecules are present, not an infinite lattice of them.

When the unit cell is large enough that the nearest adsorbate self-image is beyond the ML cutoff radius, its contribution to the energy is negligibly small (identically zero in cutoff-based models). The self-image interactions then cancel exactly in the decomposition (see Section 17.3).

If the unit cell is **too small**, the adsorbate interacts with its own periodic images, introducing a spurious self-interaction energy that does not cancel in the decomposition and corrupts the host-guest interaction energy.

**Common MACE cutoff radii:**

| Model | Cutoff radius |
|-------|--------------|
| MACE-MP-0 | ~6 Å |
| MACE-OFF23 | ~12 Å |

This constraint is **identical** to the classical force-field cutoff constraint already enforced by gRASPA (each simulation cell dimension must exceed twice the interaction cutoff). No separate `SocketReplicaCell` input keyword is needed; if gRASPA's classical constraint is satisfied, the MACE constraint is satisfied for typical cutoff radii.

---

### 17.3 Why the Energy Decomposition Is Correct

The host-guest interaction energy is computed as:

```
E_interaction = E_total(fw + ads) - E_framework(fw) - E_adsorbate(ads)
```

All three evaluations use **the same unit cell matrix** (`UCBox.Cell`) and `pbc=True`. As a result:

- **Adsorbate-adsorbate PBC interactions cancel exactly.** In both `E_total` and `E_adsorbate`, the adsorbate atoms appear at **identical positions** in identical periodic environments. Any energy contribution from adsorbate-adsorbate interactions (including self-image interactions under PBC) is therefore the same in both calls. The subtraction eliminates this contribution: `Delta_E_ads-ads = 0`.

- **Framework boundary effects cancel exactly.** Framework atoms at the edges of the supercell experience the same periodic images of the framework in both `E_total` and `E_framework` (since the cell and framework positions are identical). Framework-framework interaction energies therefore cancel identically in the subtraction, leaving no boundary artifact.

- **What remains is purely the host-guest interaction energy.** After cancellation, the only surviving contribution is from cross-terms between framework atoms and adsorbate atoms — the physical quantity of interest for GCMC.

- **Framework energy caching is valid for rigid frameworks.** Because the framework geometry is fixed throughout the simulation, `E_framework` is constant. It is computed once on the first `Predict()` call and stored in `cached_E_framework_ev`. All subsequent `Predict()` calls reuse this cached value. The cached value is exact (not an approximation) because the framework positions, cell, and periodic environment are unchanged.

This decomposition is numerically stable as long as the supercell size criterion in Section 17.2 is satisfied. If the criterion is violated, adsorbate self-image interactions appear in `E_adsorbate` but not (at the same magnitude) in `E_total`, breaking the cancellation and introducing a systematic error in `E_interaction`.

---

### 17.4 Cell Matrix Convention

The cell matrix is stored in **row-major format**: each row is a lattice vector.

```
Cell = [ a[0]  a[1]  a[2] ]   <- lattice vector a (row 0)
       [ b[0]  b[1]  b[2] ]   <- lattice vector b (row 1)
       [ c[0]  c[1]  c[2] ]   <- lattice vector c (row 2)
```

gRASPA uses the standard **lower-triangular (or upper-triangular by convention) reduced cell** form:

- **a** is aligned along the x-axis: `a = (a_x, 0, 0)` so `a[1] = a[2] = 0`
- **b** lies in the xy-plane: `b = (b_x, b_y, 0)` so `b[2] = 0`
- **c** is general: `c = (c_x, c_y, c_z)`

`UCBox.Cell` is sent directly to the Python server. The three diagonal unit cell lengths
(`UCBox.Cell[0]`, `UCBox.Cell[4]`, `UCBox.Cell[8]`) are printed to stdout at
initialization for verification.

In the flat 9-element array layout used throughout the code, the index mapping is:

```
Index:  0      1      2      3      4      5      6      7      8
        a[0]   a[1]   a[2]   b[0]   b[1]   b[2]   c[0]   c[1]   c[2]
```

`UCBox.Cell[0]`, `UCBox.Cell[4]`, and `UCBox.Cell[8]` are therefore the three diagonal unit cell lengths printed at initialization.

---

### 17.5 How to Validate the Pipeline

The script `validate_socket_energy.py` (located in the repository root) provides an automated check of the energy decomposition correctness across all MC steps logged in a gRASPA output file.

**What the script does:**

1. **Parses gRASPA stdout** for lines reporting the ML energy components. Each `Predict()` call prints `E_total`, `E_framework` (computed or cached), and `E_adsorbate` to stdout.
2. **Parses the `MCEnergyWrapper` return value** (the converted host-guest interaction energy reported by gRASPA in the MC move log).
3. **Checks the decomposition identity:**
   ```
   |E_total - E_framework - E_adsorbate - MCEnergyWrapper_in_eV| < 1e-6 eV
   ```
   A failure here indicates a unit conversion error, a sign error, or a mismatch between what was logged and what was used in the acceptance criterion.
4. **Checks framework energy stability:** Verifies that `E_framework` is identical (to floating-point precision) across all MC steps after the first. Any drift indicates a bug in the caching logic or an unintended framework geometry change between steps.

**Running the validator:**

```bash
python validate_socket_energy.py graspa_run.log
```

The script exits with a non-zero status and prints the first failing step if any check fails. A clean run prints a summary of the number of steps checked and confirms all decomposition identities hold.

---

## 18. Allegro vs Socket/ASE: Energy Calculation Differences

gRASPA supports two ML backends for host-guest energy evaluation: the **Allegro** libtorch backend and the **Socket/ASE** backend (supporting any ASE calculator such as MACE, CHGNet, SevenNet, etc.). They differ fundamentally in what the model outputs and how host-guest interaction energy is derived.

### 18.1 Allegro (libtorch/TorchScript)

Allegro is an equivariant neural network interatomic potential that natively outputs a **per-atom energy decomposition**. The `Predict()` function in `torch_allegro.h` queries the model output dictionary for the key `"atomic_energy"`, which is a tensor of shape `[N_atoms, 1]` (float32):

```cpp
torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
auto atomic_energies = atomic_energy_tensor.accessor<float, 2>();

float nAtomSum = 0.0;
for (size_t i = 0; i < nAtoms; i++) {
    size_t AtomIndex = i;
    if (i >= NFrameworkAtoms) {
        AtomIndex -= NFrameworkAtoms;
        AtomIndex += N_Replica_FrameworkAtoms;  // remap to replica tensor index
    }
    nAtomSum += atomic_energies[AtomIndex][0];
}
return static_cast<double>(nAtomSum);
```

The framework contribution is **always included in the sum**. This is intentional:

- For **delta-energy moves** (translation, rotation, reinsertion): the framework is identical in old and new states, so it cancels exactly in `DNN_New − DNN_Old`. The result is a pure interaction-energy change.
- For **absolute-energy moves** (insertion, deletion): the framework self-energy appears in both the insertion and deletion evaluations and cancels in the GCMC acceptance ratio, preserving thermodynamic correctness.

The model is called **1–2 times per MC step** (once for old state, once for new state). There is no framework energy cache at the model level — the framework atoms are included in the input on every call and contribute to the sum every time.

### 18.2 Socket/ASE (MACE, CHGNet, etc.)

ASE calculators expose energy via `get_potential_energy()`, which returns a **single total energy scalar** with no per-atom decomposition. The socket path uses an **N-body delta-query scheme** where the server maintains the full accepted configuration and returns the marginal energy for each trial move.

**Per MC move (production path):**
```
C++ calls QueryDelta(comp, mol_idx, trial_xyz)
  → send_positions_extended(config_type=200+comp, mol_idx, trial_xyz, ...)
  → server computes E(trial_config) = MACE.get_potential_energy(fw + all_ads + trial_mol)
  → server returns ΔE = E(trial_config) − E_current
```

**On acceptance:**
```
C++ calls CommitXxx(comp, mol_idx)
  → server updates ads_config[comp], recomputes E_current (1 MACE call)
```

The `config_type` int32 tag routes each call server-side:

| `config_type` | Meaning | mol_idx field |
|---|---|---|
| `0` | Framework only (startup cache) | — |
| `N` (1, 2, …) | Adsorbate N only (startup validation) | — |
| `100 + N` | Combined: fw + ads N (startup validation) | — |
| `200 + N` | Delta query for adsorbate N (production MC) | yes |
| `300` | Commit insert | yes |
| `301` | Commit delete | yes |
| `302` | Commit move | yes |
| `303` | Sync full (resync entire component) | yes (= comp) |
| `500` | Query total energy | — |

This design requires **2 MACE forward passes per accepted MC move** (1 for the delta query + 1 for the commit to recompute E_current) and **1 MACE pass per rejected move** (delta query only). The old 1-body scheme required 2 MACE passes per delta-energy move type regardless of acceptance.

### 18.3 Comparison

| Feature | Allegro | Socket/ASE |
|---|---|---|
| Model output | `atomic_energy[N, 1]` per atom | `get_potential_energy()` scalar |
| HG energy derivation | Implicit: fw cancels in delta; framework offset in absolute | Explicit: `ΔE = E(trial) − E_current` (N-body marginal) |
| Many-body contributions | Captured implicitly via per-atom sum | Captured via server-maintained full-system state |
| Framework energy | Re-evaluated every call (cancels in delta) | Cached at startup; never re-evaluated |
| MACE calls per accepted move | 1–2 | 2 (delta query + commit) |
| MACE calls per rejected move | 1 | 1 (delta query only) |
| Suitable models | Allegro, NequIP (per-atom output required) | Any ASE calculator (MACE, CHGNet, SevenNet, M3GNet, …) |
| Unit of raw output | eV (from model metadata) | eV (from ASE) |
| Conversion applied | `× DNNEnergyConversion` | `× DNNEnergyConversion` |

---

## 19. Per-Move Energy Breakdown and Atoms Involved

### 19.1 What DNN replaces

The DNN **only replaces host-guest (HG) interaction terms**. Framework-framework (HH) and guest-guest (GG) interactions remain fully classical. After every DNN evaluation, `DNN_Replace_Energy()` is called:

1. Stores classical HG values: `storedHGVDW`, `storedHGReal`, `storedHGEwaldE`
2. Zeros the classical terms: `HGVDW = HGReal = HGEwaldE = 0`
3. Places the DNN result in `DNN_E`

The total energy used in the Metropolis acceptance criterion is then:

```
E_total = HHVDW + GGVDW + HHReal + GGReal + HHEwaldE + GGEwaldE + TailE + DNN_E
```

### 19.2 Drift checking

After `DNN_Replace_Energy()`, `Check_DNN_Drift()` computes:

```
DNN_Correction() = DNN_E − (storedHGVDW + storedHGReal + storedHGEwaldE)
```

This measures the discrepancy between the DNN prediction and the classical host-guest energy. If `|DNN_Correction()| > DNNDrift` (set by `MaxDNNDrift` in `simulation.input`, default 100000.0 internal units), the move is **unconditionally rejected** and the outlier configuration is written to `DNN/Outliers_*.data` for diagnostics.

### 19.3 Per-move table (socket N-body scheme)

In all cases, only **one adsorbate molecule at a time** is sent as the trial position.
`ConsiderThisAdsorbateAtom` filtering applies before the positions are packed into
`UCAtoms[comp]`. The server's stored `ads_config` provides the N-body context implicitly.

| Move type | Trial pos source | mol_idx | DNN_New | DNN_Old | DNN_E used in MC |
|---|---|---|---|---|---|
| Translation | `Sims.New.pos` | moving mol | QueryDelta result | 0 | DNN_New (ΔE from server) |
| Rotation | `Sims.New.pos` | moving mol | QueryDelta result | 0 | DNN_New |
| Reinsertion | `d_a[comp].pos` after kernel | moving mol | QueryDelta result | 0 | DNN_New |
| Insertion (CBMC) | `Sims.Old.pos`* | −1 | QueryDelta result | 0 | DNN_New |
| Deletion (CBMC) | — (no positions) | mol to delete | **−QueryDelta** | 0 | DNN_New (negated — see below) |
| Insertion (SINGLE) | `Sims.New.pos` | −1 | QueryDelta result | 0 | DNN_New − DNN_Old |
| Deletion (SINGLE) | — (no positions) | mol to delete | QueryDelta result | 0 | DNN_New − DNN_Old |
| Total energy (FxnMain) | — | — | QueryTotal result | — | direct assignment |

\* For CBMC insertion, trial positions are staged into `Sims.Old.pos` by the
`Initialize_DNN_Positions` GPU kernel before the DNN call. The naming reflects the
unified scratch buffer used by the kernel, not the semantic "old state".

**CBMC Deletion sign convention:** `QueryDelta(DELETION)` returns
`E(N-1) − E(N)` = positive for a bound molecule. CBMC MC acceptance code interprets
`DNN_New` as a binding energy (negative = favorable). The patch negates:
```cpp
DNN_New = -SystemComponents.DNN.QueryDelta(DELETION, SelectedComponent, mol_idx, conv);
```
For `SINGLE_DELETION`, the delta convention already uses unsigned change, so no negation.

**Why DNN_Old = 0 for all moves:** In the N-body scheme, the server's `E_current`
already encodes the energy with the current (old) position of the moving molecule. The
delta query `ΔE = E(trial) − E_current` is therefore relative to the old state. Setting
`DNN_Old = 0` and using `DNN_New = ΔE` gives the correct net change `DNN_New − DNN_Old = ΔE`.

### 19.4 Identity Swap limitation

`IdentitySwapMove()` (in `mc_swap_moves.h`) throws a `std::runtime_error` if `UseDNNforHostGuest` is enabled:

```cpp
if(SystemComponents.UseDNNforHostGuest)
    throw std::runtime_error("NO DEEP POTENTIAL FOR IDENTITY SWAP!");
```

DNN support for identity swap moves is not implemented.

---

## 20. Data Structures for DNN Atomic Positions

### 20.1 `Atoms` struct (`data_struct.h`)

The primary position container used throughout gRASPA, for both classical and DNN pathways:

```cpp
struct Atoms {
    double3* pos;           // xyz coordinates, one entry per atom
    double*  scale;         // VDW scaling factor
    double*  charge;        // partial charge
    double*  scaleCoul;     // Coulomb scaling factor
    size_t*  Type;          // pseudo-atom type index (into PseudoAtoms table)
    size_t*  MolID;         // molecule ID for each atom
    size_t   Molsize;       // atoms per molecule (constant for this component)
    size_t   size;          // current number of atoms stored
    size_t   Allocate_size; // allocated capacity (2× size for adsorbates)
};
```

**Flat index formula:** atom `j` of molecule `M` in component `C` maps to:

```
d_a[C].pos[M * Moleculesize[C] + j]
```

For adsorbate components, `Allocate_size = 2 × max_atoms`. The second half of the allocation is used as a trial-position buffer during volume moves (`UseOffset=true`).

### 20.2 `Allegro` struct (`torch_allegro.h`) — DNN-specific containers

| Field | Type | Purpose |
|---|---|---|
| `UCAtoms` | `std::vector<Atoms>` | Unit cell atoms, one entry per component. Adsorbate entries contain only atoms selected by `ConsiderThisAdsorbateAtom`. |
| `ReplicaAtoms` | `std::vector<Atoms>` | Supercell tiling of `UCAtoms`. Default 3×3×3 = 27 replicas per component. |
| `UCBox` | `Boxsize` | Unit cell box matrix and inverse. Derived from the simulation supercell divided by `NumberofUnitCells`. |
| `ReplicaBox` | `Boxsize` | Replicated supercell box used for neighbor list calculations. |
| `Model` | `torch::jit::script::Module` | Frozen TorchScript model, loaded on CUDA at startup. |
| `ElementSymbolUsed` | `std::vector<std::string>` | Element symbols extracted from model metadata `type_names` (e.g. `["C", "H", "O"]`). |
| `Match_AllegroElement_PseudoAtom_order` | `std::vector<int>` | Maps gRASPA pseudo-atom type index → Allegro element index. `-1` for unrecognized types. Built by string-matching `PseudoAtoms.Symbol` against `ElementSymbolUsed`. |
| `DNN_Molsize` | `size_t` | Number of adsorbate atoms actually passed to the DNN per molecule (≤ `Moleculesize`). Excludes fictional sites filtered by `ConsiderThisAdsorbateAtom`. |
| `Cutoff` | `double` | Neighbor list cutoff in Å (default 6.0). |
| `NReplicacell` | `int3` | Replica cell dimensions, e.g. `{3, 3, 3}`. Must be odd. |
| `NL` | `NeighList` | Neighbor list: atom-pair edges within `Cutoff` distance across the replica supercell. |

### 20.3 Position flow per MC step

**Socket path (N-body delta-query):**

```
GPU: Sims.Old.pos / Sims.New.pos
     (Molsize atoms, double3)
         |
         | cudaMemcpy(DeviceToHost)
         v
CPU: temp_pos[]
     (full Moleculesize atoms, all sites including fictional)
         |
         | Check_DNNAtom_and_copy_pos_to_UCAtoms()
         | filters by ConsiderThisAdsorbateAtom[j]
         | packs surviving atoms densely (no gaps)
         v
CPU: UCAtoms[comp].pos[0..DNN_Molsize-1]
         |
         | QueryDelta(move_type, comp, mol_idx, DNNEnergyConversion)
         |   WrapSuperCellAtomIntoUCBox(comp)  — PBC-wrap into UCBox
         |   send_positions_extended(config_type=200+comp, mol_idx, ...)
         |     → server: E(trial_config) = MACE(fw + ads_config + trial_mol)
         |     → server returns ΔE = E(trial_config) − E_current
         |   × DNNEnergyConversion
         v
double DNN_E = ΔE  (internal units: 10 J/mol)

On acceptance:
         |
         | CommitXxx(comp, mol_idx)
         |   send_positions_extended(config_type=300/301/302, mol_idx, accepted_pos)
         |     → server updates ads_config[comp]
         |     → server recomputes E_current = MACE(fw + all_updated_ads)
```

**Allegro path (for reference):**

```
GPU: all atoms in UCAtoms[comp] + framework
         |
         | MCEnergyWrapper(comp, Initialize, DNNEnergyConversion)
         |   WrapSuperCellAtomIntoUCBox()   — PBC-wrap into unit cell
         |   GenerateReplicaCells()          — tile to NReplicacell supercell
         |   Get_Neighbor_List_Replica()     — build edges within Cutoff
         |   Predict()
         |     build pos[N,3], atom_types[N], edge_index[2,E]  (CPU)
         |     .to(torch::kCUDA)
         |     Model.forward() → "atomic_energy"[N,1]
         |     sum selected atomic energies → double
         |   × DNNEnergyConversion
         v
double DNN_E  (internal units: 10 J/mol)
```

### 20.4 `ConsiderThisAdsorbateAtom` filter

A `bool*` array of length `Moleculesize[adsorbate_comp]`, stored as CUDA managed memory (accessible from both host and device). Entry `j` is `true` if atom site `j` of the adsorbate molecule template should be evaluated by the DNN, `false` for fictional sites (e.g. TIP4P virtual M-site, lone pairs).

**Population:** Set at input-parsing time from the `DNNPseudoAtoms` keyword. For each listed name, gRASPA resolves the pseudo-atom type index and marks the corresponding molecule sites `true`. Sites not listed remain `false`.

**Effect:** `UCAtoms[comp].size = DNN_Molsize` is the count of `true` entries, which may be less than `Moleculesize`. The `UCAtoms` and `ReplicaAtoms` arrays are sized and populated accordingly — the model never sees the fictional sites.

### 20.5 Tensor layout fed to the Allegro model

All tensors are built on CPU then transferred to CUDA before `Model.forward()`:

| Tensor | Shape | dtype | Contents |
|---|---|---|---|
| `pos` | `[N_total, 3]` | float32 | Positions of all replica atoms: all framework replicas first, then adsorbate replica atoms |
| `atom_types` | `[N_total]` | int64 | Allegro element indices from `Match_AllegroElement_PseudoAtom_order` |
| `edge_index` | `[2, N_edges]` | int64 | Neighbor list: row 0 = source atom indices, row 1 = target atom indices |

`N_total = N_replica_framework + N_replica_adsorbate`

where `N_replica_framework = NReplicacell.x × NReplicacell.y × NReplicacell.z × UCAtoms[0].size` and analogously for the adsorbate component.

The atom-energy output `"atomic_energy"[N_total, 1]` is summed over one unit-cell equivalent of framework atoms plus the adsorbate atoms, with adsorbate indices remapped from the UC ordering to the replica tensor ordering before summing.

---

## 21. MC → Energy Call Flow and the N-body Scheme

This section traces how a Monte Carlo move becomes a DNN energy evaluation, explains the
role of `DNN_Replace_Energy` and `Check_DNN_Drift`, describes the CBMC vs. SINGLE energy
conventions, and explains how the N-body scheme fits into this machinery.

---

### 21.1 General Structure: How gRASPA Handles DNN Energies

gRASPA's MC acceptance machinery uses an **energy-correction model**: the DNN does not
fully replace the force field; instead it replaces only the **host-guest (HG) interaction
terms** while framework–framework (HH) and guest–guest (GG) remain classical. The handover
happens via two functions called after every DNN evaluation:

**`DNN_Replace_Energy()`** (called from `DNN_Prediction_Move` after the socket/Allegro call):
```
1. Save classical HG terms:
     storedHGVDW   = running_energy.HGVDW
     storedHGReal  = running_energy.HGReal
     storedHGEwaldE = running_energy.HGEwaldE
2. Zero the classical HG terms:
     HGVDW = HGReal = HGEwaldE = 0
3. Set DNN result:
     DNN_E = (value returned by socket/Allegro call)
```

After this, the total energy in the acceptance criterion is:
```
E_total = HHVDW + GGVDW + HHReal + GGReal + HHEwaldE + GGEwaldE + TailE + DNN_E
```
The classical HG is gone from the total; DNN_E has taken its place.

**`Check_DNN_Drift()`** (called immediately after `DNN_Replace_Energy`):
```
correction = DNN_E − (storedHGVDW + storedHGReal + storedHGEwaldE)
if |correction| > MaxDNNDrift → unconditional reject, write outlier to DNN/Outliers_*.data
```
This is a sanity filter: if the DNN disagrees with the classical HG by more than
`MaxDNNDrift` internal units, the configuration is assumed to be outside the training
distribution and is rejected. The threshold must be set large enough to allow legitimate
ML corrections through.

---

### 21.2 Two MC Energy Conventions

gRASPA uses **two different conventions** for what `DNN_E` means depending on the move
type. Understanding this is critical to interpreting socket patch code.

#### CBMC moves (Insertion / Deletion via Rosenbluth sampling)

CBMC builds a Rosenbluth chain by growing the molecule atom-by-atom across multiple trial
orientations. At each growth step the **Boltzmann weight** is evaluated:

```
w_i = exp(−β × E_HG_i)
```

For CBMC, `DNN_E = DNN_New` is interpreted as an **absolute interaction energy**
(binding energy convention): **negative = favorable binding**. The Rosenbluth weight is
`exp(−β × DNN_New)`.

- **Insertion:** Rosenbluth weight for new config = `exp(−β × DNN_New)`. DNN_New should be negative for favorable binding.
- **Deletion:** Rosenbluth weight for reference config = `exp(−β × DNN_New)`. DNN_New should again be negative (binding energy of the molecule being deleted).

#### SINGLE moves (Translation / Rotation / Single-step Insertion / Deletion)

For SINGLE moves, the acceptance probability is based on the **energy change** `ΔE`:

```
P_accept = min(1, exp(−β × ΔE))   [standard Metropolis]
```

Here `DNN_E = DNN_New − DNN_Old` is the **signed energy change**: negative = move is
energetically favorable (downhill).

---

### 21.3 Why the CBMC Deletion Sign Requires Special Handling

With the N-body scheme, `QueryDelta(DELETION, comp, mol_idx)` returns:

```
ΔE = E(fw + N_ads − mol_del) − E(fw + N_ads) = E(N-1 config) − E(N config)
```

For a bound molecule this is **positive** (removing a molecule from its binding site costs
energy). But CBMC convention expects `DNN_New` to be **negative** (binding energy =
favorable). Therefore the CBMC deletion patch negates:

```cpp
DNN_New = -SystemComponents.DNN.QueryDelta(DELETION, SelectedComponent, mol_idx, conv);
```

This sign flip converts the "cost to remove" into the "binding energy of the molecule",
which is what the Rosenbluth weight computation expects.

For `SINGLE_DELETION`, no negation is needed: the single-move acceptance criterion uses
`DNN_New − DNN_Old` directly as a signed energy change, and `QueryDelta(DELETION)` already
gives the correct positive ΔE for that convention.

---

### 21.4 The 1-Body Scheme (Predecessor)

The original socket patch evaluated the host-guest interaction energy for a **single
adsorbate in isolation**, ignoring all other adsorbates:

```
E_HG(1-body) = E(fw + 1 mol) − E_fw − E(1 mol alone)
```

This is a 2-body term only: it captures the direct interaction between the adsorbate and
the framework, but it does **not** include:

- Adsorbate–adsorbate interactions mediated by the ML model
- Framework polarization changes due to multiple adsorbates
- Any collective many-body effects in the system

In practice, for high loading GCMC simulations or systems where the ML model captures
significant adsorbate–adsorbate correlations, this scheme introduces a systematic bias.

The call pattern was:

```
MCEnergyWrapper(comp, Initialize, DNNEnergyConversion)
  → WrapSuperCellAtomIntoUCBox(comp)
  → Predict(ads_comp):
      PredictFromSocket(fw_atoms + ads_atoms, config_type=100+comp)  [E_total]
      PredictFromSocket(fw_atoms,             config_type=0)         [E_fw, cached]
      PredictFromSocket(ads_atoms,            config_type=comp)      [E_ads, fresh]
      return E_total − E_fw − E_ads
```

This required **2 MACE forward passes per move** (E_total + E_ads; E_fw cached). For
SINGLE moves (translation, rotation), it required 4 passes total (new + old states each
needing 2).

---

### 21.5 The N-body Scheme: What Changes

The N-body scheme replaces the `MCEnergyWrapper` / `Predict` call with `QueryDelta`, and
adds `CommitXxx` on acceptance. The structural change is minimal — `DNN_Replace_Energy`
and `Check_DNN_Drift` are called in exactly the same way. Only the source and meaning
of `DNN_E` changes.

**What's the same:**
- `DNN_Replace_Energy()` is still called after every move evaluation
- `Check_DNN_Drift()` is still called; the `correction` threshold and reject logic are unchanged
- `DNN_E` is still placed in the acceptance criterion in place of classical HG
- `MaxDNNDrift` still guards against out-of-distribution configurations

**What's different:**

| Aspect | 1-body scheme | N-body scheme |
|--------|--------------|---------------|
| `DNN_E` represents | `E(fw + 1 mol) − E_fw − E_ads` (2-body HG) | `E(fw + N_ads + trial) − E_current` (many-body marginal) |
| Server state | Stateless: each call is independent | Stateful: server stores all accepted adsorbate positions |
| MACE passes per query | 2 (total + ads) | 1 (trial config) |
| MACE passes on acceptance | 0 | 1 (recompute E_current) |
| Many-body contributions | Missing | Fully captured |
| `DNN_Old` for SINGLE moves | Non-zero: re-evaluate old config | Zero: old state already in E_current |
| Translation ΔE source | `E_new_1body − E_old_1body` | `E(new_config) − E_current` (E_current includes old pos) |
| Drift correction meaning | DNN vs classical HG (2-body) | DNN many-body marginal vs classical HG (2-body) |

**Drift check with N-body:** The correction `DNN_E − storedHGVDW` now measures the
discrepancy between the many-body marginal (which includes adsorbate–adsorbate cross terms
captured by the ML model) and the classical 1-body HG. This will naturally be larger for
systems with significant many-body effects. `MaxDNNDrift` must be set large enough that
legitimate corrections are not rejected — the drift check is intended to catch numerical
instabilities and out-of-distribution geometries, not to penalize physically meaningful
many-body corrections.

---

### 21.6 Acceptance → Commit → Server State Update

After an accepted move, the server must be notified so its stored configuration mirrors
the C++ state. This is handled by the commit patches in `mc_utilities.h` and
`move_struct.h`, which fire immediately after the GPU arrays are updated:

```
C++ acceptance path (AcceptInsertion, for example):
  1. GPU kernel: Update_NumberOfMolecules(...)      ← GPU d_a[comp] updated
  2. CommitInsert(comp):
       cudaMemcpy last mol slot → host temp_pos
       Check_DNNAtom_and_copy_pos_to_UCAtoms(temp_pos, UCAtoms[comp], ...)
       send_positions_extended(config_type=300, mol_idx=-1, accepted_xyz, ...)
         → server: ads_config[comp].append(xyz)
         → server: E_current = MACE(fw + all_updated_ads)
```

The **mol-index ordering invariant** must be maintained:
- `CommitInsert`: both C++ and server append new molecule at index `N` (new last slot)
- `CommitDelete(mol_idx)`: C++ calls `Update_deletion_data_Parallel` which swaps
  `mol_idx ← last mol` then decrements count. Server's `handle_commit_delete` must
  perform the identical swap-with-last operation.

Any break in this invariant causes the server's `ads_config[comp]` ordering to diverge
from C++, after which all subsequent delta queries compute the wrong ΔE.

---

### 21.7 FxnMain Total Energy Check

`Check_Simulation_Energy()` in `fxn_main.h` calls `DNN_Prediction_Total()` periodically
to verify that accumulated `DNN_E` contributions have not drifted from a fresh evaluation.

With the N-body scheme, `PATCH_SOCKET_FXNMAIN` replaces the old per-molecule loop
(which called `MCEnergyWrapper` for each molecule individually) with a single call:

```cpp
DNN_E = SystemComponents.DNN.QueryTotal(SystemComponents.DNNEnergyConversion);
// → config_type=500; server returns E_current − E_fw_cached
```

`E_current` already encodes the full system energy `E(fw + all N_ads)`. Subtracting
`E_fw_cached` gives the total host-guest contribution, directly analogous to what the
sum-over-molecules gave in the 1-body scheme — but now including all many-body
contributions.

This also eliminates the ordering issue of the old scheme (where `MCEnergyWrapper` was
called for each molecule independently and the per-molecule energies were summed, losing
any cross-molecule interaction terms the ML model might capture).
