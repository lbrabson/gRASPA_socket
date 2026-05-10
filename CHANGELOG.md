# Changelog

## 2026-05-09 — Fix: Gibbs particle transfer uses single-body moves when `TURN_OFF_CBMC_SWAP yes`

**Summary:** When `TURN_OFF_CBMC_SWAP yes` is set, `SingleSwap = true` gates GCMC
insertions and deletions to the classical single-body path (`SingleBodyMove`) — but
`GibbsParticleTransfer` had no equivalent gate. It always called `Insertion_Body` and
`Deletion_Body`, which build a multi-trial CBMC Rosenbluth weight from the classical
force field and then multiply in the DNN correction on top. For `UsePureDNN yes` (no
framework, all guest–guest interactions via MLIP), the stored host–guest terms are zero
so `DNN_Correction()` returns `DNN_E` unchanged, leaving the classical GG Rosenbluth
— `exp(−β·U_GG_classical)` — in the acceptance weight alongside the DNN energy. This
double-counts guest–guest interactions for every Gibbs particle transfer, over-accepting
vapor→liquid insertions and under-accepting liquid→vapor deletions, which systematically
overestimates the liquid-phase density.

**Fix:** Added a `SingleSwap` branch inside `GibbsParticleTransfer` that mirrors the
pattern used for GCMC swaps in `axpy.cu`. When `SingleSwap = true`, the function calls
`SingleBody_Prepare` + `SingleBody_Calculation` for both the insertion box and the
deletion box instead of `Insertion_Body` / `Deletion_Body`. No classical Rosenbluth is
ever formed. The Gibbs acceptance criterion is computed directly:

```
PAcc = exp(−β·(InsertionEnergy.total() + DeletionEnergy.total()))
       × N_B·V_A / ((N_A+1)·V_B)
```

For `UsePureDNN yes`, `InsertionEnergy.total() = DNN_E_ins = E(N_A+1) − E_isolated − E(N_A)`
and `DeletionEnergy.total() = DNN_E_del = E(N_B−1) + E_isolated − E(N_B)`, so the
isolated-molecule energies cancel and the exponent reduces to the correct total Gibbs
energy change `E(N_A+1) + E(N_B−1) − E(N_A) − E(N_B)`.

On acceptance, `AcceptInsertion(..., SINGLE_INSERTION)` and `AcceptDeletion(..., SINGLE_DELETION)`
are used, matching the mechanics of the GCMC single-body path. The existing CBMC path
(`Insertion_Body` / `Deletion_Body`) is fully preserved for `SingleSwap = false`.

**Files changed:**

- `src_clean/mc_swap_moves.h` `GibbsParticleTransfer` — added `if(SingleSwap)` branch
  before the CBMC path; single-body insertion + deletion + Gibbs PAcc + accept block

---

## 2026-04-24 — Fix: out-of-bounds Ewald write after Gibbs volume move acceptance

**Summary:** After a Gibbs (or NPT) volume move was accepted, `Box.tempEik` and
`Box.AdsorbateEik` were swapped correctly but the size trackers diverged: the old code
did `EikAllocateSize = tempEikAllocateSize`, which copied the (possibly enlarged)
`tempEikAllocateSize` into `EikAllocateSize` but left `tempEikAllocateSize` unchanged.
After the pointer swap, `Box.tempEik` is the old `AdsorbateEik` (smaller allocation), yet
`tempEikAllocateSize` still reported the larger post-reallocation value. On the next call
to `Ewald_TotalEnergy`, the reallocation guard (`Nblock > tempEikAllocateSize`) evaluated
false, so no reallocation occurred, and `TotalFourierEwald` wrote beyond the end of the
undersized `Box.tempEik` array → "CUDA Error: illegal memory access."

This bug only manifested in UMA runs (classical Ewald active, `noCharges=false`); MACE
runs use `noCharges=true` and never enter the Ewald volume-move path.

**Fix:** Changed `EikAllocateSize = tempEikAllocateSize` to
`std::swap(EikAllocateSize, tempEikAllocateSize)` at both acceptance sites. After the
swap, each size tracker correctly follows its corresponding pointer: `EikAllocateSize`
reflects the capacity of `AdsorbateEik` and `tempEikAllocateSize` reflects the capacity
of `tempEik`.

**Files changed:**

- `src_clean/mc_box.h` NPT volume acceptance block (~line 300) — `std::swap` instead of
  one-directional assignment
- `src_clean/mc_box.h` NVTGibbs volume acceptance block (~line 531) — same fix

---

## 2026-04-23 — Fix: stale `HostSystem` positions passed to `SyncFull` after `CreateMolecule_InOneBox`

**Summary:** When `CreateNumberOfMolecules > 0` (e.g. GEMC pre-filling boxes) and the socket
DNN is active, `SyncFull` was called immediately after `CreateMolecule_InOneBox` using
`HostSystem[comp].pos` (CPU), which had never been synced from the GPU. All molecule positions
appeared at zero / uninitialized memory, so the ML server received a degenerate configuration
and returned infinite energy. GCMC runs were unaffected because they start with
`CreateNumberOfMolecules 0`, causing `SyncFull` to hit its `if(n_mol == 0) return` early exit.

**Fix:** A `Copy_AtomData_from_Device` call is inserted in the `PATCH_SOCKET_POST_CREATEMOL`
patch block immediately before the `SyncFull` loop, pulling all accepted GPU positions into
`HostSystem[comp].pos` before they are read.

**Files changed:**

- `socket-patch/Socket/PATCH_SOCKET_main.cpp.txt` `PATCH_SOCKET_POST_CREATEMOL` — add
  `Copy_AtomData_from_Device` call before the `SyncFull` loop

---

## 2026-04-23 — Feature: DNN-enabled Gibbs volume moves (`UsePureDNN` + socket)

**Summary:** When `UsePureDNN yes` and `UseSocket` are both active, `NVTGibbsMove` now
queries the ML server for the energy of the scaled configuration and uses the DNN ΔE in
the acceptance criterion instead of the classical energy. The implementation uses two new
protocol messages (`QUERY_VOLUME` / `COMMIT_VOLUME`) and does not rely on `SYNC_FULL`.

**Protocol additions** (config_type values):
- `QUERY_VOLUME = 400` — trial query: C++ sends all molecules of one adsorbate component
  at their new (scaled) Cartesian positions plus the new cell matrix. Server computes
  `E(trial) - E_current` without modifying stored state, and saves `pending_volume_mols`.
- `COMMIT_VOLUME = 401` — zero-payload commit: C++ sends after an accepted move; server
  commits `pending_volume_mols → comp_molecules` and sets `E_current = pending_E`.

**C++ flow in `NVTGibbsMove`:**
1. `ScalePositions` kernel runs as before (new positions in second half of GPU array).
2. After the classical energy loop, if `UsePureDNN && UseSocket`: `cudaDeviceSynchronize`,
   copy scaled positions from `d_a[comp].pos + Allocate_size` device→host, call
   `DNN.QueryVolume` per box per component, accumulate `DNN_dE[sim]` per box.
3. Inside the acceptance block, after computing classical `DeltaE[sim]`, set
   `DeltaE[sim].DNN_E = DNN_dE[sim]` and call `DeltaE[sim].DNN_Replace_Energy(UsePureDNN)`.
   This zeros all classical terms and stores the DNN ΔE in the HG field — identical
   to the pattern used for insertion, deletion, and translation moves. The original
   `Pacc` formula using `DeltaE[0].total() + DeltaE[1].total()` is unchanged.
4. On accept: call `DNN.CommitVolume` (permanently updates `UCBox.Cell`) before
   `CopyScaledPositions`.
5. On reject: `QueryVolume` already restores `UCBox.Cell` internally; no extra action needed.

**Cell matrix handling:** `QueryVolume` temporarily installs the scaled cell into `UCBox`
(updating both `UCBox.Cell` and `UCBox.InverseCell` in-place via new helper
`compute_inverse_cell_inplace`) so that `WrapPositionIntoUCBox` and the wire-format cell
send both use the correct new cell. The original cell is restored on return.
`CommitVolume` makes the new cell permanent.

**Limitation:** Currently handles one adsorbate component per `QUERY_VOLUME` call.
Multi-component volume moves would require sequential queries where each updates the
server's `pending_trial_mols`; not implemented (single-component GEMC is the target use case).

**Files changed:**

- `src_clean/ase_energy_client.h` — `ConfigType` enum gains `QUERY_VOLUME = 400` and
  `COMMIT_VOLUME = 401`; new helper `compute_inverse_cell_inplace`; new functions
  `QueryVolume` and `CommitVolume`; timing fields and `PrintTimingSummary` updated
- `src_clean/mc_box.h` `NVTGibbsMove` — DNN volume energy block + commit in accept path;
  acceptance criterion switches between classical ΔE and `DNN_dE_total`
- `ase_ipi_server_uma_pure.py` — `QUERY_VOLUME` / `COMMIT_VOLUME` handlers added to
  `serve()`; `pending_volume_mols` / `pending_volume_comp` state variables; timing dict updated
- `ase_ipi_server_mace_pure.py` — identical server-side changes (kept in sync)

---

## 2026-04-22 — Fix: SIGFPE and SIGSEGV during socket init with empty-box (GEMC) framework

**Summary:** Two zero-atom guards added to the socket client initialization path, fixing
crashes that occurred when running GEMC simulations where the framework component is an
empty box (no atoms).

1. `CopyAtomsFromFirstUnitcell` computed `HostAtoms.size % NAtoms` where `NAtoms` was
   derived from `HostAtoms.Molsize / denom`. For an empty-box framework `Molsize == 0`,
   giving `NAtoms == 0`, and the modulo caused a SIGFPE (integer modulo-by-zero).

2. `PrimeFrameworkCache` unconditionally printed `UCAtoms[0].pos[0]` regardless of
   `n_fw`. Because the early return added in fix 1 skips `AllocateUCSpace`, `UCAtoms[0].pos`
   is uninitialized when `n_fw == 0`, so the dereference caused a SIGSEGV. The printf is
   now guarded by `if(n_fw > 0)`. The subsequent `PredictFromSocket` call with `n_fw == 0`
   is still made so the server receives the expected `FRAMEWORK_PRIME` exchange and caches
   `E_fw = 0.0`.

**Files changed:**

- `src_clean/ase_energy_client.h` `CopyAtomsFromFirstUnitcell` — early return when
  `denom == 0 || HostAtoms.Molsize == 0`, setting `UCAtoms[comp].size = 0`
- `src_clean/ase_energy_client.h` `PrimeFrameworkCache` — guard on `n_fw > 0` before
  printing first-atom coordinates

---

## 2026-04-22 — Enable restarts of a simulation with two boxes; handle two socket paths simultaneously

**Summary:** Socket preparation is edited to enable using separate socket paths for different
simulation boxes and to allow for simultaneous restarting of multiple simulation boxes

**Files changed:**

- `socket_patch/PATCH_SOCKET_main.cpp` (line 62) — read user-specified socket path from env variables
- `src_clean/main.cpp` (line 260) — restart box of a specific simulation when one command 
file runs several sims
- `src_clean/read_data.cpp` (line 2814) — edit filename of Restart file to 
account for sim ID
- `src_clean/read_data.h` (line 40) — update `RestartFileParser` class declaration
to include number of simulation boxes

---

## 2026-04-07 — Fix: zero tail correction when `UsePureDNN yes`

**Summary:** `DNN_Replace_Energy()` in `MoveEnergy` now zeroes `TailE` when
`UsePureDNN` is true. Previously, when all classical host-guest, host-host, and
guest-guest terms were zeroed, the tail correction (a classical long-range VDW
correction) was left non-zero. For a pure-DNN simulation the tail correction is
physically wrong — the MLIP already captures all long-range interactions — so
retaining it double-counted a classical correction against a fully ML-computed energy.

**Files changed:**

- `src_clean/data_struct.h` (line 492) — added `TailE = 0.0;` inside the
  `if(UsePureDNN)` block of `DNN_Replace_Energy()`
- `patch_Socket/data_struct.h` (line 494) — same change (kept in sync)

---

## 2026-03-12 — Add pure DNN mode (`UsePureDNN`)

**Summary:** Added a `UsePureDNN yes` simulation input option that lets the MLIP handle all interatomic interactions (host–host, host–guest, and guest–guest) without any classical force-field contribution. Previously `DNN_Replace_Energy` only zeroed the host–guest classical terms; with `UsePureDNN`, the host–host and guest–guest VDW, Real, and Ewald terms are zeroed as well.

**New keyword:** `UsePureDNN yes` in the DNN model setup block of `simulation.input`.

**Files changed:**

- `src_clean/data_struct.h` — `DNN_Replace_Energy()` gains a `bool UsePureDNN = false` argument; when `true`, additionally zeroes `HHVDW`, `HHReal`, `HHEwaldE`, `GGVDW`, `GGReal`, `GGEwaldE`. Added `bool UsePureDNN = false` field to `Components` struct.
- `src_clean/read_data.cpp` — Parses `UsePureDNN yes` in `ReadDNNModelSetup`.
- `src_clean/main.cpp` — Propagates `UsePureDNN` flag when copying component settings across simulation boxes.
- `src_clean/mc_swap_utilities.h` — `Insertion_Body` and `Deletion_Body` pass `SystemComponents.UsePureDNN` to `DNN_Replace_Energy`.
- `ase_ipi_server_mace_pure_dnn.py` *(new file)* — Alternative server variant for pure DNN mode. Returns `[E(fw+trial) − E_isolated_mol] − E_current` for insertions and `[E(fw+trial) + E_isolated_mol] − E_current` for deletions, so that guest–guest interactions are fully captured by the MLIP. Mirrors the protocol and state machine of `ase_ipi_server_mace.py` but subtracts an isolated-molecule self-energy on each delta query.

---

## 2026-03-11 — Add debug mode to print constituent energies

**Summary:** Added a `DebugMode yes` simulation input option that prints a per-move breakdown of all energy components before and after `DNN_Replace_Energy`, useful for diagnosing socket energy issues.

**New keyword:** `DebugMode yes` in the DNN model setup block of `simulation.input`.

**Files changed:**

- `src_clean/data_struct.h` — Added `bool DebugMode = false` field to `Components`.
- `src_clean/read_data.cpp` — Parses `DebugMode yes` in `ReadDNNModelSetup`.
- `src_clean/main.cpp` — Propagates `DebugMode` flag when copying component settings.
- `src_clean/mc_single_particle.h` — When debug mode is active, prints HH/HG/GG VDW, Real, Ewald, `DNN_E`, `preFactor`, `Beta`, `TailE`, and total energy for translation/rotation moves both before and after `DNN_Replace_Energy`.
- `src_clean/mc_swap_utilities.h` — Same per-component printout for insertion and deletion moves.

---

## 2026-03-05 — Fix: massless-site support in many-body socket patch

**Summary:** Pseudo-atoms with zero mass (e.g. TIP4P M-site charge site) are now excluded from the atom list sent over the socket. Previously, all adsorbate atoms were forwarded regardless of mass, causing the server to receive unphysical massless-site positions that the MLIP cannot handle.

**Root cause:** `Match_Element_PseudoAtom_with_model()` tried to match every pseudo-atom symbol — including massless virtual sites — against the MLIP's element list. The default "pass all atoms" path in `main.cpp` also included them without checking mass.

**Files changed:**

- `src_clean/ase_energy_client.h`
  - `Match_Element_PseudoAtom_with_model()`: skips pseudo-atoms whose `PseudoAtoms.mass[i] <= 0.0` (logs a `massless, skipping` message instead of attempting to match).
- `src_clean/main.cpp`
  - Default `ConsiderThisAdsorbateAtom` initialisation (no `DNNPseudoAtoms` keyword) now checks `PseudoAtoms.mass[pseudoAtomType] > 0.0` per atom-site; massless sites receive `false`.
- `socket-patch/Socket/PATCH_SOCKET_main.cpp.txt`
  - Element symbol list built for the socket now filters out symbols that have no mass-bearing pseudo-atom entry, preventing massless species from appearing in the type map sent to the server.

---

## 2026-03-04 — Auto-generate socket name; propagate via `GRASPA_SOCKET_PATH`

**Summary:** The UNIX socket path is now auto-generated per run, eliminating manual
coordination between the Python server and gRASPA, and preventing collisions when
multiple jobs run concurrently on the same node.

**Problem:** The socket name was hardcoded as `ase_ipi_socket` in all launch scripts.
Concurrent SLURM jobs on the same node would share `/tmp/ase_ipi_socket`, causing
cross-job interference. Users also had to manually keep the name in sync between the
bash script and any documentation.

**Solution:** The bash script generates a unique name (`graspa_<6hex>`) via a Python
one-liner, exports it as `GRASPA_SOCKET_PATH`, passes it to the server via `--socket`,
and gRASPA's C++ client reads it from the environment. No manual coordination needed.

**Files changed:**

- `ase_ipi_server_mace.py` — `--socket` default changed from `"ase_ipi_socket"` to
  `None`; `parser.error()` guard added so omitting `--socket` gives a clear message.

- `runs/profiles/run_05/gcmc_mace.bash` — replaced `SOCKET_NAME="ase_ipi_socket"` with
  an inline Python one-liner:
  ```bash
  SOCKET_NAME=$(python3 -c "import random, string; print('graspa_' + ''.join(random.choices(string.hexdigits.lower(), k=6)))")
  SOCKET_PATH="/tmp/${SOCKET_NAME}"
  export GRASPA_SOCKET_PATH="${SOCKET_PATH}"
  ```
  Removed `--species-file` from the server invocation (already deprecated/ignored).
  Removed `SPECIES_FILE` from cleanup `rm -f`.

**Note:** `src_clean/ase_energy_client.h` already reads `GRASPA_SOCKET_PATH` — no C++
changes required. The inline one-liner is used instead of importing `generate_random_socket_name()`
directly because the module may not be on the Python path when bash runs from the run
subdirectory.

---

## 2026-02-27 — Fix: DNNDrift rejection blocking all N-body socket moves

**Summary:** Guarded the DNNDrift rejection check with `!SystemComponents.UseSocket`
in three locations. The check compares DNN_E (full N-body ΔE_HG over all adsorbates)
against classical_HG (1-molecule 1-body energy), which is physically a large quantity
for any non-zero loading — not a sign of model instability. The false comparison
caused every GCMC insertion, deletion, translation, rotation, and single-swap attempt
to be rejected immediately, freezing loading at the `CreateMolecule_InOneBox` count.

**Files changed:**
- `src_clean/mc_swap_utilities.h` — `Insertion_Body`: line 119, `Deletion_Body`: line 212
- `src_clean/DNN_HostGuest_Energy_Functions.h` — `Check_DNN_Drift`: line 307
- `ase_ipi_server_mace.py` — `_build_atoms`: added shape-consistency assertion for
  per-component mol_pos vs mol_syms to give a clearer error message on mismatch;
  `fw_symbols` now stored as an explicit list copy instead of an alias.

---

## 2026-02-27 — Fix: constant low loading via N-body stateful consistency

**Summary:** Resolved the "constant low loading" bug by restoring strict coordinate consistency between the gRASPA client and the MACE server. The server now caches trial configurations during queries, eliminating the risk of coordinate-wrapping mismatches or data-transfer desync during commits.

### 1. Server-side Trial Caching
- **Implementation:** Introduced a `pending_mol` variable in the `serve()` loop of `ase_ipi_server_mace.py`.
- **Logic:** During a `DELTA_QUERY`, the server stores the trial molecule's coordinates. On a subsequent `COMMIT_INSERT` or `COMMIT_MOVE`, it applies this cached configuration to the primary state (`comp_molecules`) instead of reading new positions from the socket.
- **Benefit:** Guarantees that the configuration accepted by gRASPA is *exactly* the same one evaluated by the ML potential, preventing loading stagnation caused by floating-point drift or wrapping discrepancies.

### 2. Protocol Simplification & Efficiency
- **Client Changes:** Simplified `CommitInsert` and `CommitMove` in `src_clean/ase_energy_client.h` and the corresponding socket patches to send null payloads (0 atoms).
- **Reduced Overhead:** Eliminated redundant `cudaMemcpy` operations and coordinate-wrapping logic on the client for every accepted move.
- **Improved Reliability:** Reduces the total number of MACE forward passes per accepted move from 2 to 0 (the query evaluation results are reused), making commits essentially free.

---

## 2026-02-26 — Timing profile, graceful shutdown, and incremental build

**Summary:** Three operational improvements for production runs and development
iteration: end-to-run performance timing, a race-free socket shutdown that ensures
timing summaries are always printed, and an incremental `Makefile` that replaces the
monolithic `NVC_COMPILE` rebuild script.

### 1. End-to-run timing profile

Per-call-type socket timing is now accumulated on both sides and printed as a formatted
summary at the end of each run.

**C++ client** (`src_clean/ase_energy_client.h`):
- Added `#include <omp.h>`.
- Added six timing field pairs to `struct Socket`:
  `t_query_delta/n_query_delta`, `t_commit_insert/n_commit_insert`,
  `t_commit_move/n_commit_move`, `t_commit_delete/n_commit_delete`,
  `t_query_total/n_query_total`, `t_sync_full/n_sync_full`.
- Each public method (`QueryDelta`, `CommitInsert`, `CommitDelete`, `CommitMove`,
  `SyncFull`, `QueryTotal`) wraps its `PredictFromSocketExtended` call with
  `omp_get_wtime()` bookends that accumulate into the appropriate pair.
- New `PrintTimingSummary(FILE* out) const` method prints a table with total time,
  call count, and ms/call for each category, plus a grand total.
- Called from `EndOfSimulationWrapUp()` in `src_clean/main.cpp` behind a
  `if(UseSocket)` guard.

**Python server** (`ase_ipi_server_mace.py`):
- Added `import time` and `import json`.
- Five timing phases tracked per message: `t_recv`, `t_build`, `t_mace_full`,
  `t_mace_ads`, `t_send` — accumulated into a `_timing` dict keyed by message type
  (`FRAMEWORK`, `DELTA_QUERY`, `COMMIT_INSERT`, `COMMIT_DELETE`, `COMMIT_MOVE`,
  `SYNC_FULL`, `QUERY_TOTAL`).
- New `_print_timing_summary()` helper prints the table at end of session.
- JSON profile written to the path given by `--profile-output`
  (default: `./runs/profiles/server_profile.json`).
- New `runs/profiles/` directory created for profile files.

### 2. Graceful socket shutdown

**Root cause:** `close_socket()` called `send_command("EXIT")` then immediately
`close(fd)`. The kernel could send a TCP RST before the server read the EXIT bytes,
causing `ConnectionError` (ECONNRESET) on the Python side. Because the exception
propagated out of `serve()` to the outer `try/except` in `main()`, the post-loop
timing summary code was never reached.

**Fixes (two-part):**

1. **`src_clean/ase_energy_client.h`** — `close_socket()` now calls
   `shutdown(fd, SHUT_WR)` instead of `send_command("EXIT")`. `shutdown` flushes
   buffered bytes and sends a clean FIN; no RST race is possible.

2. **`ase_ipi_server_mace.py`** — The `send_header(STATUS)` / `recv_header()` block
   at the top of the `while True` loop is now wrapped in
   `try/except (ConnectionError, BrokenPipeError, OSError)` that `break`s cleanly out
   of the loop. The post-loop timing summary therefore always executes.

### 3. Incremental build Makefile

**Problem:** `NVC_COMPILE` compiled all five translation units unconditionally and
ended with `rm *.o`, preventing any object-file reuse between builds. A single header
change triggered a full ~5-minute recompile.

**Solution:** New `src_clean/Makefile` (copied to `patch_Socket/Makefile` by
`patch.py`) uses the same `nvc++` flags as `NVC_COMPILE` and adds `-MMD -MP` for
automatic header dependency tracking. Object files are retained between builds; `make`
recompiles only changed translation units.

```bash
cd patch_Socket/
make            # incremental build
make clean      # remove binary + .o + .d files
make cleanobj   # remove .o + .d only (keep binary)
```

### Changes

| File | Change |
|------|--------|
| `src_clean/ase_energy_client.h` | Added timing fields, `PrintTimingSummary()`, fixed `close_socket()` |
| `src_clean/main.cpp` | Call `PrintTimingSummary()` in `EndOfSimulationWrapUp()` |
| `src_clean/Makefile` | New file — incremental build with `-MMD -MP` |
| `ase_ipi_server_mace.py` | Timing infrastructure, `--profile-output` flag, graceful disconnect |

---

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
