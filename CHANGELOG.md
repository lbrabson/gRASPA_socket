# Changelog

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
