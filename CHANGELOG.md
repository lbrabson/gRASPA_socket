# Changelog

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
