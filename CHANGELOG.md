# Changelog

## 2026-02-18 — Energy validation dump, unit logging, and move-type tracking

**Summary:** Added an energy validation system that writes every MACE call to an
extended-XYZ file so structures can be independently re-evaluated to confirm
that gRASPA is sending the correct geometry and energies to the ML potential.
Also added a one-time unit sanity log and per-1000-step energy summary.

### Changes

1. **`patch_Socket/ase_energy_client.h`** — Added the following fields to the
   `Socket` struct:
   - `cached_xyz_framework` / `cached_xyz_adsorbate` — member vectors holding
     the framework and adsorbate positions used in the first `Predict()` call
     (promoted from static locals so `DumpValidationFrame()` can access them).
   - `validation_mode` / `validation_max` / `validation_count` / `validation_fp`
     — control and state for the validation dump (file pointer, frame counter,
     maximum frames).
   - `current_move_type` — set by the caller before each `MCEnergyWrapper()` so
     that the dump records which MC move type produced each configuration.
   - `nstep` — counts total `Predict()` calls for the periodic summary log.

   Added two new methods:
   - `OpenValidationFile(path)` — opens the dump file; called once after
     `connect_socket()` when `ValidationMode=yes`.
   - `DumpValidationFrame(xyz, n, E_total_ev, move_type)` — writes one
     extended-XYZ frame with the comment line:
     ```
     MOVE=<label> E_total_ev=<v> E_fw_ev=<v> E_ads_ev=<v> E_int_ev=<v>
     N_FW=<n_fw> Lattice="<9 floats>" Properties=species:S:1:pos:R:3 pbc="T T T"
     ```
     `N_FW` encodes the framework/adsorbate split so the validation script does
     not need a separate species file.

   Added in `Predict()`:
   - **One-time unit validation log** — on the first call, prints supercell cell
     vectors, atom counts, first few atom positions (Angstrom-scale check), and
     all three self-energies in eV. Includes a note about cache invalidation
     requirements if flexible molecules or frameworks are ever added.
   - **Per-1000-step summary** — `if (nstep % 1000 == 0)` prints a single line
     with E_total, E_fw, E_ads, and E_int for ongoing monitoring.
   - **Validation dump call** — after computing E_total, calls
     `DumpValidationFrame()` when `validation_mode` is active and
     `validation_count < validation_max`.

   Added `fclose(validation_fp)` in `close_socket()`.

2. **`patch_Socket/read_data.cpp`** — Added two optional keywords parsed in
   `ReadSocketModelParameters()`:
   - `ValidationMode yes|true` → sets `DNN.validation_mode = true`
   - `ValidationMaxFrames <N>` → sets `DNN.validation_max = N` (default 100)

3. **`patch_Socket/main.cpp`** — After `connect_socket()`, calls
   `OpenValidationFile("validation_dump_box<N>.xyz")` for each simulation box
   when `validation_mode` is enabled.

4. **`patch_Socket/DNN_HostGuest_Energy_Functions.h`** — Added
   `SystemComponents.DNN.current_move_type = <MoveType>` immediately before
   each of the six `MCEnergyWrapper()` call sites (INSERTION, DELETION,
   TRANSLATION/ROTATION/SINGLE new, TRANSLATION/ROTATION/SINGLE old,
   REINSERTION new, REINSERTION old), so each dumped frame is correctly labelled
   with its MC move type.

5. **`validate_energies.py`** (new file, repo root) — Python script that
   re-runs MACE independently on every frame in a validation dump file and
   compares the energies to those recorded by gRASPA. Usage:
   ```bash
   python validate_energies.py \
       --xyz  validation_dump_box0.xyz \
       --model /path/to/model.pt \
       [--device cpu|cuda] [--dtype float32|float64] \
       [--tol 1e-4] [--max-frames N]
   ```
   Checks performed:
   - Frame 0: `MACE(fw only) == gRASPA cached E_fw` (species + positions correct at init)
   - Frame 0: `MACE(ads only) == gRASPA cached E_ads`
   - All frames: `MACE(total) − E_fw − E_ads == gRASPA E_int` (MC geometry integrity)

### Patch system sync

All changes above were also propagated to the patch source files so that
`python patch.py` regenerates an identical `patch_Socket/`:

- **`src_clean/ase_energy_client.h`** — copied directly from `patch_Socket/`
  (this file has no patch markers; `src_clean/` is its authoritative source).
- **`socket-patch/Socket/PATCH_SOCKET_read_data.cpp.txt`** — added the
  `ValidationMode` and `ValidationMaxFrames` keyword parsing blocks.
- **`socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`** — added the
  `OpenValidationFile()` call block after `connect_socket()`.
- **`socket-patch/Socket/PATCH_SOCKET_DNN_HostGuest_Energy_Functions.h.txt`**
  — added `current_move_type = MoveType/REINSERTION` before each of the six
  `MCEnergyWrapper()` call sites (INSERTION, DELETION, SINGLE×2, REINSERTION×2).

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
