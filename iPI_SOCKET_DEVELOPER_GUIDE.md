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
15. [Energy Validation](#energy-validation)

---

## 1. Overview

gRASPA uses an **i-PI socket protocol** to offload host-guest interaction energy calculations to an external machine-learning server (MACE via ASE). The C++ simulation engine acts as a **socket client**, sending atomic configurations over a UNIX domain socket and receiving predicted energies. This replaces (or supplements) classical force-field host-guest energy during Monte Carlo moves.

**Key design choice:** The energy is decomposed as:
```
E_interaction = E_total(framework+adsorbate) - E_framework - E_adsorbate
```
E_framework and E_adsorbate are **cached after the first evaluation** (both are constant for a rigid framework + replicated rigid molecule), so subsequent calls require only **1 socket round-trip** for E_total.

**Energy validation:** An optional validation dump writes every MACE-call configuration to an extended-XYZ file so that `validate_energies.py` can re-run MACE independently and confirm that gRASPA is passing the correct structures and energies to the ML potential. See [Section 15](#energy-validation).

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
|  - Wraps atoms into unit cell                    |
|  - Replicates to 3x3x3 supercell                |
|  - Sends positions via iPI protocol              |
|  - Receives energy (eV), converts to 10J/mol    |
+------------------+-------------------------------+
                   |  UNIX Domain Socket
                   |  /tmp/ipi_ase_ipi_socket
                   v
+--------------------------------------------------+
|  Python Server (ase_ipi_server_claude.py)        |
|  - ASE SocketIOCalculator (iPI protocol)         |
|  - CHGNet v0.3.0 neural network potential        |
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

### Per-evaluation data exchange

```
Client                              Server
------                              ------
                    <-- STATUS
READY -->
                    <-- POSDATA
cell[9 doubles] -->
inv_cell[9 doubles] -->
natoms [int32, network order] -->
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

Forces and virial are **read but not used** by gRASPA (protocol compliance only).

---

## 5. Core Implementation

**File:** `ase_energy_client.h` (present in both `src_clean/` and `patch_Socket/`)

### Socket Struct

```cpp
struct Socket {
    // Geometry
    Boxsize UCBox;                              // Unit cell box
    Boxsize ReplicaBox;                         // Supercell box
    std::vector<Atoms> UCAtoms;                 // Atoms per component (unit cell)
    std::vector<Atoms> ReplicaAtoms;            // Atoms per component (supercell)

    // Element mapping
    std::vector<std::string> ElementSymbolUsed;
    std::vector<int> Match_Element_PseudoAtom_order;

    // Supercell control
    int3 NReplicacell = {1,1,1};                // Default; set to {3,3,3} at init

    // Connection state
    char socket_path[108] = "/tmp/ase_ipi_socket";
    int fd = -1;                                // File descriptor
    size_t natoms = 0;
    size_t DNN_Molsize = 0;                     // Atoms per adsorbate (excluding fictional)
    size_t nstep = 0;                           // Total Predict() calls (for periodic log)
    bool handshake_done = false;

    // Energy caching (framework & adsorbate are constant across MC moves)
    double cached_E_framework_ev = 0.0;
    double cached_E_adsorbate_ev = 0.0;
    bool   cache_valid = false;

    // Cached xyz arrays for the framework and isolated adsorbate (set on first Predict())
    // Kept as members so DumpValidationFrame() can access them without recomputing.
    std::vector<double> cached_xyz_framework;
    std::vector<double> cached_xyz_adsorbate;

    // Validation dump: records every MACE call to an extended-XYZ file
    bool   validation_mode  = false;            // Enabled via ValidationMode=yes
    size_t validation_max   = 100;              // Max frames (ValidationMaxFrames)
    size_t validation_count = 0;               // Frames written so far
    FILE*  validation_fp    = nullptr;          // Output file handle
    int    current_move_type = -1;              // Set by caller before MCEnergyWrapper()
};
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `init(path, n_atoms)` | Set socket path and initial atom count |
| `connect_socket()` | Open UNIX socket, call `do_ipi_handshake()` |
| `send_positions(xyz)` | Full POSDATA exchange (cell + atoms) |
| `receive_energy(energy_ev)` | Read FORCEREADY response |
| `PredictFromSocket(xyz, n)` | Send coords, get energy back (eV); auto-connects if `fd < 0` |
| `Predict()` | Energy decomposition with caching: 3 calls on first invocation, 1 call thereafter. Fires unit validation log on first call; per-1000-step summary every 1000 calls; calls `DumpValidationFrame()` when validation is active |
| `MCEnergyWrapper(comp, init, conv)` | Wrap + replicate + predict + unit convert |
| `CopyAtomsFromFirstUnitcell(...)` | Extract UC atoms, filter fictional sites |
| `GenerateReplicaCells(alloc)` | Build 3x3x3 supercell |
| `WrapSuperCellAtomIntoUCBox(comp)` | PBC wrapping via fractional coords |
| `WriteSpeciesFile(path)` | Write framework/adsorbate element symbols for the Python server |
| `OpenValidationFile(path)` | Open the extended-XYZ validation dump file; call after `connect_socket()` when `validation_mode=true` |
| `DumpValidationFrame(xyz, n, E_total_ev, move_type)` | Append one extended-XYZ frame to the validation dump, including all energy components and `N_FW` |
| `close_socket()` | Send EXIT command, close fd and validation file (call explicitly; no destructor) |
| `Match_Element_PseudoAtom_with_model(PA)` | Map pseudo-atoms to MACE element symbols |

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
- Parses two optional validation keywords:
  - `ValidationMode yes|true` → sets `DNN.validation_mode = true`
  - `ValidationMaxFrames <N>` → sets `DNN.validation_max = N` (default 100)

### `main.cpp`
- **Lines 119-130:** Calls `ReadDNNModelSetup()` early in init
- **Lines 193-197:** Copies `UseSocket` flag into runtime `Components`
- **Lines 332-392 (patched):** Full socket setup:
  - Match elements to pseudo-atoms
  - Copy framework + adsorbate atoms into `UCAtoms`
  - Set `NReplicacell = {3,3,3}`
  - `WrapSuperCellAtomIntoUCBox()` + `GenerateReplicaCells(true)` — populate ReplicaAtoms (no socket needed)
  - `WriteSpeciesFile("socket_species.txt")` — server reads this after handshake
  - `connect_socket()` — connect and complete iPI handshake
  - If `validation_mode`: `OpenValidationFile("validation_dump_box<N>.xyz")`
  - First test `Predict()` call (ReplicaAtoms already populated)
  - Second test with displaced atoms via `MCEnergyWrapper(1, false, ...)`

### `DNN_HostGuest_Energy_Functions.h`
All MC move types are handled. **Before each `MCEnergyWrapper()` call,
`SystemComponents.DNN.current_move_type` is set** so that the validation dump
records the correct MC move label for every configuration.

| Move Type | Patch Name | What it does |
|-----------|-----------|--------------|
| INSERTION | `PATCH_SOCKET_INSERTION` | Sets `current_move_type=INSERTION`, `cudaMemcpy` trial pos → filter → `MCEnergyWrapper()` |
| DELETION | `PATCH_SOCKET_DELETION` | Sets `current_move_type=DELETION`, same flow for deletion candidate |
| TRANSLATION/ROTATION | `PATCH_SOCKET_SINGLE` | Sets `current_move_type` for new and old calls separately; returns `E_new - E_old` |
| SINGLE_INSERTION | `PATCH_SOCKET_SINGLE` | Sets `current_move_type=SINGLE_INSERTION`; evaluates new only |
| SINGLE_DELETION | `PATCH_SOCKET_SINGLE` | Sets `current_move_type=SINGLE_DELETION`; evaluates old only |
| REINSERTION | `PATCH_SOCKET_REINSERTION` | Sets `current_move_type=REINSERTION` for both new and old; returns delta |
| Total energy | `PATCH_SOCKET_FXNMAIN` | Loop over all molecules in component 1, sum energies |

### `fxn_main.h`
- **Lines ~341-346:** `DNN_Prediction_Total()` called during `Check_Simulation_Energy()`
- **Lines ~491-500:** Reports DNN energy, stored classical HG energy, and the correction

### Helper: `Check_DNNAtom_and_copy_pos_to_UCAtoms()`
Filters atoms using `ConsiderThisAdsorbateAtom[]` boolean array (excludes fictional charge sites like TIP4P M-site) and copies positions from GPU-transferred buffer into `UCAtoms`.

---

## 7. Patching System

The codebase uses a text-marker patching approach. `patch.py` reads every
`socket-patch/Socket/PATCH_SOCKET_*.txt` file, locates the matching
`###PATCH_SOCKET_*###` marker in the corresponding `src_clean/` file, and
inserts the snippet code immediately after it, producing the buildable
`patch_Socket/` directory.

### Special case: `ase_energy_client.h`

**`ase_energy_client.h` has no patch markers.** `patch.py` copies it verbatim
from `src_clean/` to `patch_Socket/`. The authoritative source is therefore
`src_clean/ase_energy_client.h` — edit it there and `patch_Socket/` will be
updated on the next `patch.py` run. Never edit `patch_Socket/ase_energy_client.h`
directly without also updating `src_clean/`.

### Patch marker table

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

### Update workflow

When making changes to socket logic:

| File to change | Edit here | Also update |
|----------------|-----------|-------------|
| `ase_energy_client.h` | `src_clean/ase_energy_client.h` | `patch_Socket/` via `patch.py` |
| `read_data.cpp` socket section | `socket-patch/Socket/PATCH_SOCKET_read_data.cpp.txt` | `patch_Socket/` via `patch.py` |
| `main.cpp` socket setup | `socket-patch/Socket/PATCH_SOCKET_main.cpp.txt` | `patch_Socket/` via `patch.py` |
| `DNN_HostGuest_Energy_Functions.h` | `socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt` | `patch_Socket/` via `patch.py` |

**To apply patches:** `python patch.py` (requires `pandas` and `matplotlib`).
The `patch_Socket/` directory always contains the already-patched, buildable result.

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

### `simulation.input` Optional Validation Settings

```
ValidationMode        yes         # dump every MACE call to validation_dump_box<N>.xyz
ValidationMaxFrames   50          # stop dumping after this many frames (default 100)
```

When `ValidationMode=yes`, gRASPA writes an extended-XYZ file
`validation_dump_box0.xyz` (one per simulation box) and `validate_energies.py`
can be used to re-run MACE independently on those frames. See
[Section 15](#energy-validation).

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
E_host_guest = E_total - E_framework - E_adsorbate
```

### Caching Optimization

Because the framework is rigid and the adsorbate is a single rigid molecule replicated into a 3x3x3 supercell, **E_framework and E_adsorbate are effectively constant** across MC moves. `Predict()` exploits this:

- **First call** (`cache_valid == false`): Makes 3 socket round-trips (E_total, E_framework, E_adsorbate). Stores E_framework and E_adsorbate in `cached_E_framework_ev` and `cached_E_adsorbate_ev`, sets `cache_valid = true`.
- **Subsequent calls** (`cache_valid == true`): Makes **1 socket round-trip** (E_total only), reuses cached values.

This reduces socket calls by ~67% after initialization.

```
First Predict():
  1. PredictFromSocket(framework + adsorbate)  --> E_total      (computed)
  2. PredictFromSocket(framework)              --> E_framework  (computed, cached)
  3. PredictFromSocket(adsorbate)              --> E_adsorbate  (computed, cached)

Subsequent Predict():
  1. PredictFromSocket(framework + adsorbate)  --> E_total      (computed)
     E_framework, E_adsorbate                                  (from cache)
```

Debug output labels each energy as `(computed)`, `(computed, cached)`, or `(cached)`.

**For MC move deltas (after cache is warm):**
- Translation/Rotation: `delta_E = E_new_config - E_old_config` (2 socket calls)
- Insertion: `E_new_config` only (1 socket call)
- Deletion: `E_old_config` only (1 socket call)
- Reinsertion: Both old and new (2 socket calls)

---

## 10. Supercell Replication

The socket implementation builds a supercell before sending coordinates to the ML model, ensuring proper treatment of periodic boundary conditions.

**Default:** 3x3x3 (27 replicas), set at `main.cpp` initialization.

**Requirement:** All `NReplicacell` components must be **odd** so the original unit cell sits at the center.

**Process:**
1. `GenerateUCBox()` - Divide simulation cell by number of unit cells
2. `CopyAtomsFromFirstUnitcell()` - Extract one UC of atoms, filtering fictional sites
3. `WrapSuperCellAtomIntoUCBox()` - Apply PBC via fractional coordinate wrapping
4. `GenerateReplicaCells()` - Scale box and replicate atoms across all 27 images

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
5. wait_for_species_file()               (gRASPA writes this after connecting)
6. serve() main loop                     (STATUS → READY → POSDATA → ...)
```

The handshake (`do_handshake()`) is separated from the main serve loop (`serve()`) so the server can complete the handshake before needing species info. On reconnection, the species data is already loaded.

### Key Functions

| Function | Purpose |
|----------|---------|
| `do_handshake(conn, cell)` | Perform iPI handshake (STATUS/NEEDINIT/INIT) |
| `serve(conn, calc, fw, ads, cell)` | Main protocol loop (handshake must be done already) |
| `parse_species_file(path)` | Parse `socket_species.txt` → `(fw_symbols, ads_symbols)` |
| `wait_for_species_file(path)` | Poll until species file exists and is non-empty |

### Typical Launch Script (`gcmc_mace.bash`)

```bash
rm -f /tmp/ase_ipi_socket socket_species.txt          # Clean old artifacts
python ase_ipi_server_mace.py \
    --socket ase_ipi_socket \
    --model /path/to/model.pt &                       # Start server
sleep 10                                              # Wait for model load
./nvc_main.x                                          # Start gRASPA
kill %1                                               # Stop server
rm -f /tmp/ase_ipi_socket                             # Cleanup
```

---

## 12. Runtime Workflow

```
Server side:
  S1. Load ML model
  S2. Bind + listen on /tmp/ipi_ase_ipi_socket
  S3. Accept connection (blocks until gRASPA connects at step 8)
  S4. do_handshake() — STATUS → NEEDINIT → INIT
  S5. wait_for_species_file() — polls until gRASPA writes it (step 7)
  S6. serve() main loop — handles POSDATA requests

gRASPA side:
  1. Parse simulation.input -> UseSocket = true, DNNEnergyConversion = 9648.53...
  2. ReadSocketModelParameters() -> DNNModelName, MaxDNNDrift
  3. Match pseudo-atoms to element symbols
  4. Copy framework atoms into DNN.UCAtoms[0]
  5. Copy adsorbate template into DNN.UCAtoms[1]
  6. Set NReplicacell = {3,3,3}
  7. WrapSuperCellAtomIntoUCBox + GenerateReplicaCells(true) -> populate ReplicaAtoms
  8. WriteSpeciesFile("socket_species.txt") -> server can now read species
  9. connect_socket() -> handshake with server (server at step S4)
 10. First test Predict() call (ReplicaAtoms already set up)
 11. Second test with displaced atoms via MCEnergyWrapper(1, false, ...)
 12. Begin MC simulation loop:
     a. Select MC move type
     b. Generate trial configuration (GPU)
     c. cudaMemcpy trial positions to host
     d. Filter atoms via ConsiderThisAdsorbateAtom[]
     e. Copy to DNN.UCAtoms
     f. MCEnergyWrapper() -> Wrap -> Replicate -> Predict() -> 1 socket call
     g. Convert eV -> 10J/mol
     h. Check drift against MaxDNNDrift
     i. Metropolis acceptance/rejection
     j. Update system state
 13. Process exits -> OS reclaims socket fd (no destructor; Socket is copied during init)
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

1. **One socket call per energy evaluation** (after first call caches framework/adsorbate energies) — the first `Predict()` still requires 3 calls
2. **Synchronous/blocking I/O** — no pipelining or async evaluation
3. **Socket path hardcoded** — changing requires source modification and recompilation
4. **Single socket connection** — one ML server per simulation
5. **Rigid molecules and framework only** — `read_data.cpp` throws `std::runtime_error` at startup if any adsorbate component is not marked `rigid` in its `.def` file; the framework is assumed fixed throughout the simulation. If flexible molecules or frameworks are ever added, `cached_E_adsorbate_ev` and `cached_E_framework_ev` must be invalidated (set `cache_valid = false`) before each `Predict()` call.
6. **Component 1 assumption** — adsorbate is always component index 1 in several places
7. **Forces/virial unused** — read from server for protocol compliance but discarded
8. **Server reconnection** — the server accepts new connections after a client disconnects, but the C++ client has no reconnect logic (if the server dies mid-simulation, gRASPA terminates)
9. **Validation dump is append-only** — `OpenValidationFile()` opens the file in write mode (`"w"`), so a previous dump is overwritten if the simulation is restarted

---

## 15. Energy Validation

The validation system lets you confirm that:
- gRASPA sends the correct atomic species and positions to MACE on initialisation (self-energy cache integrity)
- gRASPA passes the correct structure during each MC move (geometry integrity)
- The energies recorded internally by gRASPA match an independent MACE evaluation

### Enable in `simulation.input`

```
ValidationMode        yes
ValidationMaxFrames   50   # optional; default 100
```

### Output file format

`validation_dump_box0.xyz` is a concatenated extended-XYZ file (one frame per
`Predict()` call, up to `ValidationMaxFrames`):

```
<natoms_total>
MOVE=<label> E_total_ev=<v> E_fw_ev=<v> E_ads_ev=<v> E_int_ev=<v>
  N_FW=<n_fw> Lattice="a1 a2 a3 b1 b2 b3 c1 c2 c3" Properties=species:S:1:pos:R:3 pbc="T T T"
<symbol> x y z
...
```

- `MOVE` is one of: `TRANSLATION`, `ROTATION`, `SINGLE_INSERTION`,
  `SINGLE_DELETION`, `SPECIAL_ROTATION`, `INSERTION`, `DELETION`,
  `REINSERTION`, `WIDOM`, …
- `N_FW` is the number of leading atoms that belong to the framework; all
  remaining atoms are the adsorbate. This allows `validate_energies.py` to
  split the system without a separate species file.
- `Lattice` is the 3×3 supercell in row-major order (rows = lattice vectors),
  matching the ASE `Atoms(cell=...)` convention.
- All energies are in **eV**.

### Running the validation script

```bash
python validate_energies.py \
    --xyz  validation_dump_box0.xyz \
    --model /path/to/mace-mpa-0-medium.model \
    --device cuda \
    --tol 1e-4
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--xyz` | required | Validation dump file from gRASPA |
| `--model` | required | MACE model file path (.pt) |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--dtype` | `float64` | `float32` or `float64` |
| `--tol` | `1e-4` | Failure threshold for `|ΔE_int|` in eV |
| `--max-frames` | all | Stop after N frames |

**Checks performed:**

1. **Frame 0 — self-energy verification** (computed once):
   - `MACE(framework atoms only)` vs `gRASPA cached E_fw`
   - `MACE(adsorbate atoms only)` vs `gRASPA cached E_ads`
   - A mismatch here means the wrong positions or species were sent to MACE at
     initialisation.

2. **All frames — interaction energy**:
   - `MACE(all atoms) − E_fw_mace − E_ads_mace` vs `gRASPA E_int`
   - A mismatch here means gRASPA is sending different positions during the MC
     loop than it believes it is sending (geometry integrity).

**Exit codes:** 0 = all frames pass; 1 = at least one failure or no frames
processed.

### Interpreting results

| Symptom | Likely cause |
|---------|-------------|
| Frame 0 E_fw mismatch | Wrong framework positions in `cached_xyz_framework`; check `CopyAtomsFromFirstUnitcell()` and `GenerateReplicaCells()` |
| Frame 0 E_ads mismatch | Wrong adsorbate positions at init; check the template molecule copy in `main.cpp` |
| Per-frame E_int matches but E_total doesn't | `E_fw_mace + E_ads_mace ≠ E_fw_graspa + E_ads_graspa` — cache mismatch |
| Per-frame E_int mismatch | Positions actually sent to MACE differ from what gRASPA tracks internally (PBC wrapping, atom ordering, or unit conversion bug) |
| All values match to machine precision | Pipeline is consistent ✓ |
