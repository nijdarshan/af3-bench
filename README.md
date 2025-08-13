## AF3 Bench — Project README

### Overview
AF3 Bench is a Rust CLI that benchmarks inference-time performance for two AlphaFold3 engines by calling tiny Python shims:
- Google DeepMind AlphaFold 3 (JAX/Haiku/XLA)
- Ligo-Biosciences AlphaFold3 (PyTorch)

You can benchmark:
- Orchestration overhead (toy mode)
- Model trunk compute (trunk mode)
- Full end-to-end (full mode; best on Linux/CUDA)

Outputs are JSONL/CSV with per-pass timings and metadata.

## Directory layout

### cli/
Rust binary crate for the user-facing CLI.
- Parses flags:
  - `--engine {deepmind|ligo|both}`
  - `--mode {toy|trunk|full}` (default: trunk)
  - `--device {cpu|mps|cuda}` (default: cpu)
  - `--passes <N>`, `--seq-len <int>`, `--full` (legacy flag), `--notes`, `--out`
- Calls into `core` to execute the selected mode for each engine.
- Writes artifacts:
  - `results/<timestamp>/<engine>.jsonl`
  - `results/<timestamp>/summary.csv`
- Prints console summaries (min/median/p95) and artifact paths.

Key file:
- `cli/src/main.rs`

### core/
Rust library crate handling Python interop via PyO3.
- Exposes `call_shim(engine, passes, device, seq_len, notes, full, mode)` which:
  - Imports `deepmind_shim` or `ligo_shim`
  - Calls `forward_once(...)`
  - Returns results as JSON
- Keeps Rust clean by pushing engine-specific details into Python.

Key file:
- `core/src/lib.rs`

### py/
Python shims and helpers.
- `ligo_shim.py`
  - Modes:
    - `toy`: tiny Torch op (matmul + tanh), for orchestration overhead.
    - `trunk`: real Ligo trunk (embedder/MSA/Pairformer) with minimal config and dummy features.
    - `full`: for Linux/CUDA later (on macOS MPS, diffusion is stubbed due to fp64 limits).
  - Device selection:
    - `mps` if available, else `cpu`
    - Triton kernel shadowed with a PyTorch fallback.
  - Suppresses Python warnings for clean CLI output.

- `deepmind_shim.py`
  - Modes:
    - `toy`: tiny JAX op (matmul + tanh), runs on CPU (macOS maps `mps` → `cpu`).
    - `trunk`/`full`: planned for Linux/CUDA; on macOS we currently use toy for parity/testing.
  - Ensures venv site-packages are visible when called from Rust.

- `af3_fallback/`
  - `msa_kernel.py`: fallback implementation to replace Triton-only kernels for Ligo on macOS.

Notes:
- “Dummy features” = synthetic tensors matching expected shapes/dtypes; sufficient to exercise the model trunk without a full preprocessing pipeline.
- Random-init params used (no actual weights) per the initial scope.

### third_party/
Pinned engine repositories; not modified here.
- `third_party/alphafold3-ligo/`: Ligo PyTorch AF3
- `third_party/alphafold3-deepmind/`: DeepMind AF3 (JAX/Haiku/XLA)

We import these via `PYTHONPATH` (set by `scripts/dev.sh`).

## Setup

- Use the provided venv and environment script:
  ```bash
  cd /Users/darsh/ai/af3-bench
  source scripts/dev.sh
  ```
- Build:
  ```bash
  cargo build
  ```
- Recommended Python deps (already used during development):
  - Ligo: torch, lightning, hydra-core, rich, ml-collections
  - DeepMind toy: jax[cpu]
  - We do not alter third_party repos; shims add fallbacks where needed.

## Running

### Common flags
- `--engine {deepmind|ligo|both}`
- `--mode {toy|trunk|full}`:
  - toy: tiny op (overhead only; both engines supported)
  - trunk: model’s main blocks
    - Ligo on macOS MPS works (with fallbacks)
    - DeepMind trunk: use Linux/CUDA later (macOS impractical)
  - full: end-to-end including diffusion (best on Linux/CUDA for both)
- `--device {cpu|mps|cuda}`
- `--passes N`, `--seq-len L`
- `--notes "text"`, `--out results/<dir>`

### Examples
- Orchestration-only compare (CPU):
  ```bash
  target/debug/af3-bench run --engine both --mode toy --device cpu --passes 10 --seq-len 64
  ```
- Ligo trunk on macOS MPS:
  ```bash
  target/debug/af3-bench run --engine ligo --mode trunk --device mps --passes 5 --seq-len 32
  ```
- Both engines toy on MPS:
  ```bash
  target/debug/af3-bench run --engine both --mode toy --device mps --passes 3 --seq-len 16
  ```
- Full end-to-end (Linux/CUDA; planned flow):
  ```bash
  target/debug/af3-bench run --engine both --mode full --device cuda --passes 5 --seq-len 64
  ```

## Outputs

- JSONL per engine: one JSON object per pass
  - Fields: engine, pass_index, start_ts, elapsed_ms, device, seq_len, notes, commit, cpu_brand, ok, (optional error)
- CSV summary consolidating both engines:
  - `engine,pass_index,start_ts,elapsed_ms,device,seq_len,commit,notes,cpu_brand`

Example:
```
results/<timestamp>/deepmind.jsonl
results/<timestamp>/ligo.jsonl
results/<timestamp>/summary.csv
```

## Platform notes

- macOS (Apple Silicon M4, MPS):
  - Ligo trunk is supported with fallbacks; diffusion is stubbed in MPS path (no fp64).
  - DeepMind (JAX) on MPS is not supported; toy JAX runs on CPU. Trunk/full to be run on Linux/CUDA.

- Linux/CUDA (NVIDIA):
  - Best environment for fair, end-to-end comparison of both engines.
  - Triton kernels and JAX/XLA GPU available.

## Current status

- CLI/core implemented; JSONL/CSV writing and console summaries done.
- Ligo:
  - toy/trunk modes working on macOS MPS (with Triton fallback).
  - full planned for Linux/CUDA.
- DeepMind:
  - toy mode (JAX CPU) working.
  - trunk/full planned for Linux/CUDA; macOS impractical.

## Suggested usage patterns

- Validate plumbing/overhead parity:
  - `--mode toy` on same device for both.
- Get meaningful model timing on macOS now:
  - `--engine ligo --mode trunk --device mps`
- Deliver production-like results:
  - Move to Linux/CUDA and run `--mode full --device cuda` for both engines with the same `passes` and `seq_len`.

## Development tips

- Always:
  ```bash
  source scripts/dev.sh
  ```
  to set `PYO3_PYTHON` and `PYTHONPATH`.

- If you see import issues under PyO3:
  - Ensure `jax[cpu]` (for DeepMind toy) or `torch` (for Ligo) is installed in `.venv`.
  - The shims prepend venv site-packages path when embedded; `scripts/dev.sh` also aligns environment.

- Extend modes easily by adding branches in `forward_once(...)` in `py/*_shim.py`.

- Keep third_party repos unmodified; any quirks go in `py/` shims.

## Next steps

- Linux/CUDA (recommended)
  - DeepMind: implement minimal trunk/full forwards with dummy features and random-init params; report compile_ms vs exec_ms (JAX/XLA); add `--warmup`.
  - Ligo: run full end-to-end without fallbacks; optionally expose standard model configs.
- CLI and reporting
  - Add flags for warmup and compile/exec split; unify CSV columns across engines; capture GPU name and device details.
  - Optionally emit a Markdown summary alongside JSONL/CSV.
- Features
  - Provide a standardized synthetic feature generator and a loader for provided example features; allow parametric `seq_len`.
- macOS
  - Keep toy/trunk modes; document MPS limitations (no fp64, no Triton); ensure dtype guards and suppressed warnings for clean output.