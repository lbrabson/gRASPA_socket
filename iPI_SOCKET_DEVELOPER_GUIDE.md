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

---

## 1. Overview

gRASPA uses an **i-PI socket protocol** to offload host-guest interaction energy calculations to an external machine-learning server (CHGNet via ASE). The C++ simulation engine acts as a **socket client**, sending atomic configurations over a UNIX domain socket and receiving predicted energies. This replaces (or supplements) classical force-field host-guest energy during Monte Carlo moves.

**Key design choice:** The energy is decomposed as:
```
E_interaction = E_total(framework+adsorbate) - E_framework - E_adsorbate
```
E_framework and E_adsorbate are **cached after the first evaluation** (both are constant for a rigid framework + replicated rigid molecule), so subsequent calls require only **1 socket round-trip** for E_total.

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
> gRASPA are `config_type` (sent before the cell matrix), `n_mol`, and `types`.

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
types[natoms × int32, network order] -->
positions[3*natoms doubles] -->
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

`config_type` values: `0` = framework only, `1..N` = adsorbate-only for component N,
`100+N` = total (framework + adsorbate N). The server returns the **host-guest
interaction energy** `E(fw+ads) - E_fw_cached - E(ads)` directly; gRASPA makes
exactly **one socket call per MC move**.

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

| Method | Purpose |
|--------|---------|
| `init(path, n_atoms)` | Set socket path and initial atom count |
| `connect_socket()` | Open UNIX socket, handshake, send species map, prime server cache |
| `send_positions(xyz, types, n, config_type, n_mol)` | Full POSDATA exchange (config_type + cell + types + atoms, all network byte order) |
| `receive_energy(energy_ev)` | Read FORCEREADY response |
| `PredictFromSocket(xyz, types, n, config_type, n_mol)` | Send coords, get energy back (eV); auto-connects if `fd < 0` |
| `PrimeFrameworkCache()` | Send framework-only config (`config_type=0`); server computes and caches E_fw |
| `send_species_map()` | Send element symbols over the wire immediately after handshake |
| `Predict(ads_comp)` | Build combined fw+ads xyz array, call `PredictFromSocket(config_type=100+ads_comp)`; server returns HG energy directly |
| `MCEnergyWrapper(comp, init, conv)` | Wrap UCAtoms into UCBox + Predict() + unit convert |
| `CopyAtomsFromFirstUnitcell(...)` | Extract UC atoms, filter fictional sites |
| `WrapSuperCellAtomIntoUCBox(comp)` | PBC wrapping via fractional coords |
| `close_socket()` | Send EXIT command, close fd (call explicitly if needed; no destructor) |
| `Match_Element_PseudoAtom_with_model(PA)` | Map pseudo-atoms to element symbols |

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
  - First test `Predict()` call
  - Second test with displaced atoms via `MCEnergyWrapper(1, false, ...)`

### `DNN_HostGuest_Energy_Functions.h`
All MC move types are handled:

| Move Type | Patch Name | What it does |
|-----------|-----------|--------------|
| INSERTION | `PATCH_SOCKET_INSERTION` | `cudaMemcpy` trial pos → filter → `MCEnergyWrapper()` |
| DELETION | `PATCH_SOCKET_DELETION` | Same flow for deletion candidate |
| TRANSLATION/ROTATION | `PATCH_SOCKET_SINGLE` | Evaluate both old & new pos, return `E_new - E_old` |
| SINGLE_INSERTION | `PATCH_SOCKET_SINGLE` | Only evaluates new (skips old) |
| SINGLE_DELETION | `PATCH_SOCKET_SINGLE` | Only evaluates old (skips new) |
| REINSERTION | `PATCH_SOCKET_REINSERTION` | Evaluate trial + original, return delta |
| Total energy | `PATCH_SOCKET_FXNMAIN` | Loop over all molecules in component 1, sum energies |

### `fxn_main.h`
- **Lines ~341-346:** `DNN_Prediction_Total()` called during `Check_Simulation_Energy()`
- **Lines ~491-500:** Reports DNN energy, stored classical HG energy, and the correction

### Helper: `Check_DNNAtom_and_copy_pos_to_UCAtoms()`
Filters atoms using `ConsiderThisAdsorbateAtom[]` boolean array (excludes fictional charge sites like TIP4P M-site) and copies positions from GPU-transferred buffer into `UCAtoms`.

---

## 7. Patching System

The codebase uses a text-marker patching approach. Each marker in `src_clean` corresponds to a patch file in `socket-patch/Socket/`:

| Marker in `src_clean` | Patch File | Target File |
|-----------------------|-----------|-------------|
| `###PATCH_SOCKET_DATA_STRUCT_H###` | `PATCH_SOCKET_data_struct.h.txt` | `data_struct.h` |
| `###PATCH_SOCKET_READDATA###` | `PATCH_SOCKET_read_data.cpp.txt` | `read_data.cpp` |
| `###PATCH_SOCKET_READDATA_H###` | `PATCH_SOCKET_read_data.h.txt` | `read_data.h` |
| `###PATCH_SOCKET_MAIN_READMODEL###` | `PATCH_SOCKET_main.cpp.txt` | `main.cpp` |
| `###PATCH_SOCKET_MAIN_PREP###` | `PATCH_SOCKET_main.cpp.txt` | `main.cpp` |
| `###PATCH_SOCKET_CONSIDER_DNN_ATOMS###` | `PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt` | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_INSERTION###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_DELETION###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_SINGLE###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_REINSERTION###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |
| `###PATCH_SOCKET_FXNMAIN###` | (same file) | `DNN_HostGuest_Energy_Functions.h` |

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

The host-guest interaction energy is computed as:

```
E_host_guest = E_total(fw + ads) - E_framework(fw) - E_adsorbate(ads)
```

### Design: Server-side decomposition, 1 call per MC move

The three-call decomposition is handled **entirely on the Python server side**.
The C++ client makes **exactly one socket call per energy evaluation**:

```
C++ client (per MC move):
  1. Build xyz_total = framework atoms + adsorbate atoms (UCAtoms, unit cell only)
  2. PredictFromSocket(xyz_total, config_type = 100 + ads_comp, n_mol = 1)
     --> Python server returns HG = E_total - E_fw_cached - E_ads  (1 round-trip)
  3. Convert eV -> 10J/mol

Python server (per evaluation with config_type >= 100):
  A. Compute E_total(fw + ads)           (fresh MACE evaluation)
  B. Retrieve E_fw from server-side cache (set at startup by PrimeFrameworkCache())
  C. Compute E_ads(ads only)             (fresh MACE evaluation)
  D. Return HG = E_total - E_fw - E_ads
```

### Framework cache priming (at startup)

`connect_socket()` calls `PrimeFrameworkCache()` immediately after sending the
species map. This sends `config_type=0` (framework only) so the server can compute
and cache `E_fw` once before the MC loop begins. The cache is never invalidated
(framework is rigid throughout the simulation).

### Per-move socket call count

After the startup prime:

| Move type | Socket calls (C++ side) |
|-----------|------------------------|
| Translation/Rotation | 2 (one for old config, one for new config) |
| Insertion | 1 (new config only) |
| Deletion | 1 (old config only) |
| Reinsertion | 2 (old + new) |

Each of those "1 call" figures triggers **2 internal MACE evaluations** on the
server (E_total + E_ads). The framework cache (E_fw) is always reused.

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
6. recv_prime_framework()               (config_type=0 call; compute and cache E_fw)
7. serve() main loop                     (STATUS → READY → POSDATA → ...)
```

Species information now arrives over the wire immediately after the handshake
(`recv_species_map()`). There is no species file to poll. On reconnection, the
species data and framework cache are re-established via the same startup sequence.

### Key Functions

| Function | Purpose |
|----------|---------|
| `do_handshake(conn, cell)` | Perform iPI handshake (STATUS/NEEDINIT/INIT) |
| `recv_species_map(conn)` | Receive element symbols sent by C++ client after handshake; build `config_map` |
| `serve(conn, calc, config_map, cell)` | Main protocol loop; routes each eval by `config_type`; performs 2-call decomposition server-side |

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
  S3. Accept connection (blocks until gRASPA connects at step 7)
  S4. do_handshake() — STATUS → NEEDINIT → INIT
  S5. recv_species_map() — species info arrives over wire (sent by gRASPA at step 8)
  S6. serve() main loop — handles POSDATA requests

gRASPA side:
  1. Parse simulation.input -> UseSocket = true, DNNEnergyConversion = 9648.53...
  2. ReadSocketModelParameters() -> DNNModelName, MaxDNNDrift
  3. Match pseudo-atoms to element symbols
  4. Copy framework atoms into DNN.UCAtoms[0]
  5. Copy adsorbate template into DNN.UCAtoms[1]
  6. WrapSuperCellAtomIntoUCBox(0) -> wrap framework into UCBox (explicit step)
  7. connect_socket():
       a. do_ipi_handshake() (server at step S4)
       b. send_species_map() over wire (server at step S5)
       c. PrimeFrameworkCache() -> config_type=0 call; server caches E_fw
  8. First test Predict() call
  9. Second test with displaced atoms via MCEnergyWrapper(1, false, ...)
 10. Begin MC simulation loop:
     a. Select MC move type
     b. Generate trial configuration (GPU)
     c. cudaMemcpy trial positions to host
     d. Filter atoms via ConsiderThisAdsorbateAtom[]
     e. WrapSuperCellAtomIntoUCBox(comp) -> wrap adsorbate into UCBox
     f. Copy to DNN.UCAtoms
     g. MCEnergyWrapper() -> Predict() -> 1 socket call (server does 2-call decomp)
     h. Convert eV -> 10J/mol
     i. Check drift against MaxDNNDrift
     j. Metropolis acceptance/rejection
     k. Update system state
 11. Process exits -> OS reclaims socket fd (no destructor; Socket is copied during init)
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

ASE calculators expose energy via `get_potential_energy()`, which returns a **single total energy scalar** with no per-atom decomposition. To recover the host-guest interaction energy, the socket client in `ase_energy_client.h::Predict()` performs an explicit three-call decomposition:

```
E_total = PredictFromSocket(fw_atoms + ads_atoms,  config_type = 100 + comp)
E_fw    = PredictFromSocket(fw_atoms,               config_type = 0)          ← cached
E_ads   = PredictFromSocket(ads_atoms,              config_type = comp)
return E_total − E_fw − E_ads
```

The `config_type` int32 tag (sent as a 4-byte prefix before the position data in the i-PI protocol) routes each call server-side:

| `config_type` | Meaning |
|---|---|
| `0` | Framework atoms only |
| `N` (1, 2, …) | Adsorbate component N only |
| `100 + N` | Combined: framework + adsorbate component N |

The framework energy `E_fw` is computed once and stored in `cached_E_framework_ev`. Subsequent calls reuse the cached value as long as `cache_valid` is true. This reduces the number of socket round-trips per MC step to **2 (translation/rotation)** or **3 (insertion/deletion, where the adsorbate-only energy is also needed)**.

### 18.3 Comparison

| Feature | Allegro | Socket/ASE |
|---|---|---|
| Model output | `atomic_energy[N, 1]` per atom | `get_potential_energy()` scalar |
| HG energy derivation | Implicit: fw cancels in delta; constant offset in absolute | Explicit: `E_total − E_fw − E_ads` |
| Framework energy | Re-evaluated every call | Computed once, cached |
| Model calls per MC step | 1–2 | 2–3 |
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

### 19.3 Per-move table

In all cases, only **one adsorbate molecule at a time** is passed to the DNN, together with the **full framework** (via the UCAtoms/ReplicaAtoms supercell). Only atoms selected by `ConsiderThisAdsorbateAtom` are included.

| Move type | `DNN_New` source | `DNN_Old` source | Returned value | Notes |
|---|---|---|---|---|
| Translation | `Sims.New.pos` | `Sims.Old.pos` | `DNN_New − DNN_Old` | Framework cancels in difference |
| Rotation | `Sims.New.pos` | `Sims.Old.pos` | `DNN_New − DNN_Old` | Framework cancels in difference |
| Reinsertion | `temp` GPU buffer | `Sims.Old.pos` | `DNN_New − DNN_Old` | `temp` holds trial reinserted positions |
| Insertion (CBMC/single) | `Sims.Old.pos`* | — (0) | `DNN_New` | Framework self-energy cancels in acceptance ratio |
| Deletion | `Sims.Old.pos` | — (0) | `DNN_New` | Symmetric with insertion |
| Total energy (all molecules) | `HostSystem[comp].pos` per mol | — | Σ over all molecules | Called once per molecule; used for energy reporting |

\* For insertion moves, the trial positions are staged into `Sims.Old.pos` by the `Initialize_DNN_Positions` GPU kernel before the DNN call. The naming reflects the unified scratch buffer used by the kernel, not the semantic "old state".

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

```
GPU: Sims.Old.pos / Sims.New.pos
     (Molsize atoms, local indices 0..Molsize-1, double3)
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
