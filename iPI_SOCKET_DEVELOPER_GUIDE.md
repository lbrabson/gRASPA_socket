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
    size_t nstep = 0;
    bool handshake_done = false;

    // Energy caching (framework & adsorbate are constant across MC moves)
    double cached_E_framework_ev = 0.0;
    double cached_E_adsorbate_ev = 0.0;
    bool   cache_valid = false;
};
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `init(path, n_atoms)` | Set socket path and initial atom count |
| `connect_socket()` | Open UNIX socket, call `do_ipi_handshake()` |
| `send_positions(xyz)` | Full POSDATA exchange (cell + atoms) |
| `receive_energy(energy_ev)` | Read FORCEREADY response |
| `PredictFromSocket(xyz, n)` | Send coords, get energy back (eV) |
| `Predict()` | Energy decomposition with caching: 3 calls on first invocation, 1 call thereafter |
| `MCEnergyWrapper(comp, init, conv)` | Wrap + replicate + predict + unit convert |
| `CopyAtomsFromFirstUnitcell(...)` | Extract UC atoms, filter fictional sites |
| `GenerateReplicaCells(alloc)` | Build 3x3x3 supercell |
| `WrapSuperCellAtomIntoUCBox(comp)` | PBC wrapping via fractional coords |
| `Match_Element_PseudoAtom_with_model(PA)` | Map pseudo-atoms to CHGNet elements |

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
- **Lines 332-380 (patched):** Full socket setup:
  - Match elements to pseudo-atoms
  - Copy framework + adsorbate atoms into `UCAtoms`
  - Set `NReplicacell = {3,3,3}`
  - Run two test energy evaluations (one at initial position, one displaced by (1,1,1))

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

Hardcoded to `/tmp/ase_ipi_socket` in the `Socket` struct default. The Python server uses `/tmp/ipi_<socket_name>` where `<socket_name>` is passed via `--socket` flag. **These must match** (the `ipi_` prefix is added by ASE automatically).

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

**File:** `ase_ipi_server_claude.py`

```bash
# UNIX socket (recommended for local use)
python ase_ipi_server_claude.py --socket ase_ipi_socket

# INET socket (for remote use)
python ase_ipi_server_claude.py --port 31415
```

- Loads CHGNet v0.3.0 (412,525 parameters)
- Wraps it in ASE's `SocketIOCalculator` which handles the iPI protocol server side
- Supports CUDA GPU or CPU
- Stays alive accepting multiple evaluations until killed
- Socket path becomes `/tmp/ipi_ase_ipi_socket` (ASE prepends `ipi_`)

### Typical Launch Script (`gcmc_claude.bash`)

```bash
rm -f /tmp/ipi_*                                    # Clean old sockets
python ase_ipi_server_claude.py --socket ase_ipi_socket &  # Start server
sleep 20                                             # Wait for CHGNet load
./gRASPA                                             # Start simulation
kill %1                                              # Stop server
rm -f /tmp/ipi_*                                     # Cleanup
```

---

## 12. Runtime Workflow

```
1. Parse simulation.input -> UseSocket = true, DNNEnergyConversion = 9648.53...
2. ReadSocketModelParameters() -> DNNModelName, MaxDNNDrift
3. Match pseudo-atoms to element symbols
4. Copy framework atoms into DNN.UCAtoms[0]
5. Copy adsorbate template into DNN.UCAtoms[1]
6. Set NReplicacell = {3,3,3}
7. Run test MCEnergyWrapper() with Initialize=true  (first socket connection here)
8. Run test with displaced atoms to verify connection
9. Begin MC simulation loop:
   a. Select MC move type
   b. Generate trial configuration (GPU)
   c. cudaMemcpy trial positions to host
   d. Filter atoms via ConsiderThisAdsorbateAtom[]
   e. Copy to DNN.UCAtoms
   f. MCEnergyWrapper() -> Wrap -> Replicate -> Predict() -> 1 socket call (3 on first invocation)
   g. Convert eV -> 10J/mol
   h. Check drift against MaxDNNDrift
   i. Metropolis acceptance/rejection
   j. Update system state
```

---

## 13. Error Handling

| Scenario | Behavior |
|----------|----------|
| Socket connection fails | Returns -1, prints `perror()` |
| Server closes socket mid-transfer | Prints `"SOCKET CLOSED BY ASE"`, calls `exit(EXIT_FAILURE)` |
| Read/write returns partial data | Retries in loop until all bytes transferred |
| Wrong header received | Prints unexpected header to stderr |
| DNN drift exceeds threshold | Move rejected, event logged to `DNN/Outliers_*.data` |
| Missing config parameters | Throws `std::runtime_error` with descriptive message |

**Design philosophy:** Fail loudly on communication errors rather than silently producing wrong energies.

---

## 14. Known Limitations

1. **One socket call per energy evaluation** (after first call caches framework/adsorbate energies) - the first `Predict()` still requires 3 calls
2. **Synchronous/blocking I/O** - no pipelining or async evaluation
3. **Socket path hardcoded** - changing requires source modification and recompilation
4. **Single socket connection** - one ML server per simulation
5. **Rigid framework only** - `main.cpp` throws if framework is not rigid or has multiple components
6. **Component 1 assumption** - adsorbate is always component index 1 in several places
7. **Forces/virial unused** - read from server for protocol compliance but discarded
8. **No reconnection logic** - if the server dies, the simulation terminates immediately
