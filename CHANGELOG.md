# Changelog

## 2026-02-25 — N-body socket HG energy scheme

**Summary:** The socket path's host-guest energy evaluation has been upgraded from a
1-body to an N-body scheme. Previously, each MC move evaluated E_HG for a single
adsorbate against the bare framework in isolation: `E_HG = E(fw + 1 mol) - E_fw - E_ads`.
This misses many-body contributions from adsorbates already present in the box (framework
polarization, adsorbate–adsorbate interactions captured by the ML model, etc.).

The physically correct quantity for each MC move is:
- **Insertion**: `ΔE = E(fw + N_ads + mol_new) − E(fw + N_ads)`
- **Deletion**: `ΔE = E(fw + N_ads − mol_del) − E(fw + N_ads)`
- **Translation/Rotation/Reinsertion**: `ΔE = E(fw + N_ads with mol at new_pos) − E(fw + N_ads)`

**Design:** The Python server maintains the current accepted configuration
(`ads_config[comp]` = list of accepted molecule positions per adsorbate component) and
`E_current = E(fw + all ads)`. Per move: C++ sends one trial molecule's coordinates →
server returns `ΔE = E(trial_config) − E_current`. On acceptance: C++ sends a COMMIT
message → server updates `ads_config` and `E_current`. On rejection: no message needed
(server state is already consistent with C++ state).

### Wire Protocol Extensions

New `send_positions_extended()` function inserts `mol_idx [int32]` immediately after
`config_type` in the POSDATA payload for all `config_type ≥ 200`.

New config_type values:

| Value | Name | Meaning |
|-------|------|---------|
| `200 + comp` | `DELTA_QUERY_BASE` | Return `ΔE = E(trial_config) − E_current` for adsorbate `comp` |
| `300` | `COMMIT_INSERT` | Append new mol to server's `ads_config`; recompute `E_current` |
| `301` | `COMMIT_DELETE` | Swap-with-last delete `mol_idx` from server's `ads_config`; recompute `E_current` |
| `302` | `COMMIT_MOVE` | Update `mol_idx` position in server's `ads_config`; recompute `E_current` |
| `303` | `SYNC_FULL` | Replace entire adsorbate component config (post-CreateMolecule prime) |
| `500` | `QUERY_TOTAL` | Return `E_current − E_fw_cached` (for FxnMain energy check) |

### Sign Convention (CBMC Deletion)

`QueryDelta(DELETION)` returns `E(N-1) − E(N)` which is **positive** for a bound molecule.
CBMC conventions expect `DNN_New` to be **negative** (binding energy convention). The
CBMC deletion patch therefore negates the result: `DNN_New = -QueryDelta(DELETION, ...)`.
For `SINGLE_DELETION`, the delta convention is already consistent (positive ΔE = cost to
remove); no negation is applied.

### Changes

1. **`src_clean/ase_energy_client.h`** — Extended `ConfigType` enum with new values:
   `DELTA_QUERY_BASE = 200`, `COMMIT_INSERT = 300`, `COMMIT_DELETE = 301`,
   `COMMIT_MOVE = 302`, `SYNC_FULL = 303`, `QUERY_TOTAL = 500`.
   Added `send_positions_extended()` which inserts `mol_idx [int32]` after `config_type`
   in the POSDATA payload. Added new public methods: `QueryDelta()`, `CommitInsert()`,
   `CommitDelete()`, `CommitMove()`, `SyncFull()`, `QueryTotal()`. Existing `Predict()`
   and `MCEnergyWrapper()` are retained (still used by startup validation and FxnMain
   fallback path).

2. **`src_clean/mc_utilities.h`** — Added patch markers at the end of `AcceptTranslation`,
   `AcceptInsertion`, and `AcceptDeletion`: `//###PATCH_SOCKET_COMMIT_MOVE###//`,
   `//###PATCH_SOCKET_COMMIT_INSERT###//`, `//###PATCH_SOCKET_COMMIT_DELETE###//`.

3. **`src_clean/move_struct.h`** — Added `//###PATCH_SOCKET_COMMIT_REINSERTION###//` at
   the end of `ReinsertionMove::Acceptance`, after `Update_Reinsertion_data` kernel.

4. **`src_clean/main.cpp`** — Added `//###PATCH_SOCKET_POST_CREATEMOL###//` between
   `CreateMolecule_InOneBox` and `Check_Simulation_Energy` to call `SyncFull` for each
   adsorbate component after initial molecule placement.

5. **`socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt`** — All
   five move patches replaced to use `QueryDelta()` instead of `MCEnergyWrapper()`.
   `PATCH_SOCKET_FXNMAIN` replaced per-molecule loop with a single `QueryTotal()` call.

6. **`socket-patch/Socket/PATCH_SOCKET_mc_utilities.h.txt`** *(new file)* — Contains
   `PATCH_SOCKET_COMMIT_MOVE`, `PATCH_SOCKET_COMMIT_INSERT`, `PATCH_SOCKET_COMMIT_DELETE`
   patches that call `CommitMove`, `CommitInsert`, `CommitDelete` on acceptance.

7. **`socket-patch/Socket/PATCH_SOCKET_move_struct.h.txt`** *(new file)* — Contains
   `PATCH_SOCKET_COMMIT_REINSERTION` which calls `CommitMove` on accepted reinsertion.

8. **`socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`** — Added `PATCH_SOCKET_POST_CREATEMOL`
   section: calls `SyncFull` for each adsorbate component using `HostSystem[comp].pos`
   after `CreateMolecule_InOneBox` completes.

9. **`socket-patch/Socket/PATCH_SOCKET_data_struct.h.txt`** — Added forward declaration
   for `Check_DNNAtom_and_copy_pos_to_UCAtoms` in the `PATCH_SOCKET_H` section. This is
   needed because the function is defined in `VDW_Coulomb.cu` but the commit patches in
   `mc_utilities.h` and `move_struct.h` compile in `axpy.cu`'s translation unit.

### Server Changes Required

The Python server (`ase_ipi_server_mace.py`) must implement the server-side state machine:
- Maintain `ads_config[comp]` = list of accepted mol position arrays per adsorbate component
- Maintain `E_current` = E(fw + all ads), recomputed after every COMMIT
- New handlers dispatched by `config_type`:
  - `200+comp` → `handle_delta_query`: return `E(trial_config) − E_current`
  - `300` → `handle_commit_insert`: append mol; recompute `E_current`
  - `301` → `handle_commit_delete`: swap-with-last pop of `mol_idx`; recompute `E_current`
  - `302` → `handle_commit_move`: update `mol_idx` to new position; recompute `E_current`
  - `303` → `handle_sync_full`: replace entire component config; recompute `E_current`
  - `500` → `handle_query_total`: return `E_current − E_fw_cached`
- Mol-index ordering must mirror C++ `d_a[comp]`: delete uses swap-with-last matching
  `Update_deletion_data_Parallel`

### Drift Check Note

The drift check (`Check_DNN_Drift`) compares `DNN_E` (now a many-body marginal) against
the classical 1-body HGVDW. The many-body correction will be larger by construction for
systems with significant adsorbate–adsorbate or polarization effects. Users must set
`MaxDNNDrift` large enough that legitimate many-body corrections are not rejected.

---

## 2026-02-24 — Remove SocketReplicaCell: send UCBox directly to MACE

**Summary:** The supercell replication mechanism (`ReplicaAtoms`, `ReplicaBox`,
`NReplicacell`, `GenerateReplicaCells`, `ReplicateAtomsPerComponent`) has been
removed from the socket path. gRASPA now sends the unit cell atoms directly with
`UCBox.Cell` and `pbc=True`. MACE handles periodic boundary conditions natively.

**Root cause / motivation:** The replica machinery was fragile and introduced a
coordinate-system mismatch: in `PredictTotal`, framework atoms were in ReplicaBox
coordinates while adsorbate atoms were wrapped into UCBox. Additionally, the
`SocketReplicaCell` input keyword created a user-facing configuration parameter whose
correct value depends on the ML model cutoff and framework unit cell dimensions — but
gRASPA already enforces an equivalent cutoff constraint for the classical force field.

**New user constraint:** Each UCBox dimension must be > 2 × MACE cutoff. This is
identical to the gRASPA classical force field cutoff constraint; no separate input
keyword is needed.

### Changes

1. **`src_clean/ase_energy_client.h`** — Removed `ReplicaBox`, `ReplicaAtoms`,
   `NReplicacell` fields from `Socket` struct. Removed `GenerateReplicaCells()` and
   `ReplicateAtomsPerComponent()` methods. Updated `send_positions()` to use `UCBox.Cell`
   instead of `ReplicaBox.Cell`. Updated `PrimeFrameworkCache()`, `Predict()`,
   `PredictTotal()`, and `MCEnergyWrapper()` to use `UCAtoms` instead of `ReplicaAtoms`.

2. **`socket-patch/Socket/PATCH_SOCKET_read_data.cpp.txt`** — Removed `SocketReplicaCell`
   keyword parsing, `found_replica` boolean, and the "SocketReplicaCell not found"
   `runtime_error` guard.

3. **`socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`** — PREP section no longer calls
   `GenerateReplicaCells(true)`. Now explicitly calls `WrapSuperCellAtomIntoUCBox(0)`
   (framework) before connecting. Previously the framework wrap happened implicitly
   inside `ReplicateAtomsPerComponent` via fractional-coord round-trip; it must now be
   done explicitly.

---

## 2026-02-24 — Fix endianness bug in types array

**Summary:** `send_positions()` was writing the per-atom types array as native
little-endian integers. The Python server reads them with
`struct.unpack(f"!{natoms}i", ...)` (network byte order = big-endian). This caused
type indices to be byte-swapped: type index 4 (Zn) was received as 0x04000000 =
67108864, triggering an "ERROR: type index 67108864 not in species map" crash.

All other int32 fields in the protocol (`config_type`, `n_mol`, `natoms`, extras
length, force/energy responses) already used `send_int32()` with `htonl` — the types
array was the only field missing the byte-swap.

### Changes

1. **`src_clean/ase_energy_client.h`** — In `send_positions()`, replaced:
   ```cpp
   write_all(types, sizeof(int32_t) * n_atoms);
   ```
   with:
   ```cpp
   std::vector<int32_t> types_net(n_atoms);
   for (size_t i = 0; i < n_atoms; i++)
       types_net[i] = htonl((uint32_t)types[i]);
   write_all(types_net.data(), sizeof(int32_t) * n_atoms);
   ```

---

## 2026-02-18 — Fix GPU startup race condition and socket path inconsistency

**Summary:** gRASPA failed with "Connection refused" on the first run, and with
a segfault on the second. The root cause in both cases was a timing mismatch
between the Python server startup and gRASPA launch.

### Root cause

`gcmc_mace.bash` used a fixed `sleep 1` before launching gRASPA. On GPU,
`torch.load` of the MACE model takes 30–60+ seconds. This caused two failure
modes depending on ordering:

- **"Connection refused"**: gRASPA tried to connect before the server had even
  created the socket.
- **Segfault**: if the socket is created before model loading completes, gRASPA
  launches while `torch.load` is still occupying the GPU. gRASPA's own CUDA
  allocations then race against the model load for GPU memory, causing a CUDA
  fault that manifests as SIGSEGV.

The safe invariant is: **gRASPA must not start until the MACE model is fully
resident on the GPU.** This is guaranteed by keeping the original server order
(load model → create socket) and having the bash script poll for the socket file
rather than sleeping a fixed amount.

### Changes

1. **`ase_ipi_server_mace.py`** — Restored original ordering: `get_calculator()`
   runs before `bind()`+`listen()`. The socket file therefore appears only after
   the model is fully loaded, so gRASPA never launches while the GPU is busy with
   `torch.load`.

2. **`gcmc_mace.bash`** — Replaced `sleep 1` with a poll loop (`until [ -S
   "${SOCKET_PATH}" ]`) that waits up to 300 seconds for the socket file to
   appear. gRASPA is launched only once the socket exists, which implies the
   model is loaded and the server is ready to accept.

3. **`gcmc_mace.bash`** — Fixed stale `ipi_` prefix in socket path. The old
   script cleaned up and polled `/tmp/ipi_ase_ipi_socket` while the Python server
   and C++ client both use `/tmp/ase_ipi_socket`. Introduced a `SOCKET_PATH`
   variable derived consistently from `SOCKET_NAME`. Also corrected `--device`
   from `cpu` to `cuda` for GPU runs.

---

## 2026-02-17 — Fix socket path mismatch and destructor double-close

**Summary:** Two bugs found during live testing on the cluster.

### Changes

1. **`ase_ipi_server_mace.py`** — Changed socket path from `/tmp/ipi_<name>` to `/tmp/<name>` to match the C++ client's hardcoded default (`/tmp/ase_ipi_socket`). The `ipi_` prefix was a leftover from ASE's SocketIOCalculator convention which we no longer use.

2. **`src_clean/ase_energy_client.h`** — Removed `~Socket()` destructor. gRASPA copies `Socket` objects when `SystemComponents` vectors are resized during init, causing the destructor to close the shared `fd` on each temporary copy. This resulted in "Bad file descriptor" errors on subsequent socket calls. The OS handles fd cleanup on process exit; `close_socket()` remains available for explicit use.

3. **`patch_Socket/ase_energy_client.h`** — Regenerated from `src_clean/`.

---

## 2026-02-17 — Fix socket init/shutdown and startup sequencing

**Summary:** `connect_socket()` was defined but never called, causing gRASPA to
crash on the first socket write (`fd = -1`). Additionally, a startup deadlock
existed: the server waited for `socket_species.txt` before accepting connections,
but gRASPA couldn't write that file without a live socket. The shutdown path
also never called `close_socket()`.

### Changes

1. **`ase_ipi_server_mace.py`** — Extracted handshake from `serve()` into standalone `do_handshake()`. Restructured `main()` startup order to: bind+listen → accept → handshake → wait for species file → serve loop. This breaks the deadlock because the server now accepts the connection and completes the handshake before needing the species file.

2. **`socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`** — Restructured the `PATCH_SOCKET_MAIN_PREP` init block. Now calls `WrapSuperCellAtomIntoUCBox()` + `GenerateReplicaCells(true)` to populate `ReplicaAtoms`, then `WriteSpeciesFile()`, then `connect_socket()`, then the first test `Predict()`. Previously `MCEnergyWrapper()` bundled all of this but required a connected socket.

3. **`src_clean/ase_energy_client.h`** — Added lazy-connect safety net in `PredictFromSocket()`: if `fd < 0`, auto-connects or exits fatally. Added `~Socket()` destructor that calls `close_socket()` for clean shutdown.

4. **`test_ipi_server.py`** — Updated all server thread functions to call `do_handshake()` then `serve()` separately. Added `do_handshake` to imports.

5. **`patch_Socket/`** — Regenerated from `src_clean/` + `socket-patch/`.

---

## 2026-02-17 — Cache framework and adsorbate energies in Predict()

**Summary:** `Predict()` previously made 3 socket round-trips per MC move
(E_total, E_framework, E_adsorbate). Since the framework is rigid and the
adsorbate molecule is replicated identically each time, E_framework and
E_adsorbate are constant. They are now computed once and cached, reducing
socket calls by ~67% after initialization.

### Changes

1. **`src_clean/ase_energy_client.h`** — Added cache fields (`cached_E_framework_ev`, `cached_E_adsorbate_ev`, `cache_valid`) to the `Socket` struct. Modified `Predict()` to compute and cache E_framework and E_adsorbate on the first call, then reuse cached values on all subsequent calls. Debug output labels each energy as `(computed)`, `(computed, cached)`, or `(cached)`.

2. **`patch_Socket/ase_energy_client.h`** — Regenerated from `src_clean/` via `patch.py` to include the caching changes.

3. **`test_ipi_server.py`** — Added `TestCachingBehaviour` test class with `test_cache_reduces_calls` that simulates 3 Predict() cycles: the first issues 3 socket evaluations (total + framework + adsorbate), the second and third issue only 1 each (total only), verifying cached values remain consistent.

4. **`iPI_SOCKET_DEVELOPER_GUIDE.md`** — Updated Overview, Socket Struct, Key Methods, Energy Calculation Strategy, Runtime Workflow, and Known Limitations sections to document the caching optimization.
