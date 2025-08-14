## AF3 Bench

### Overview
AF3 Bench is a Rust CLI that benchmarks inference-time performance of two AlphaFold3 engines via light Python shims:
- DeepMind AlphaFold3 (JAX/Haiku/XLA)
- Ligo AlphaFold3 (PyTorch)

We run real model graphs with random-initialized parameters and synthetic inputs (no databases) to measure compilation and steady-state execution on a single GPU.

Outputs are per-pass JSONL and a CSV summary with min/median/p95.

---

## Prerequisites

- External repositories (install/clone as needed):
  - AlphaFold3 (Ligo Biosciences): [github.com/Ligo-Biosciences/AlphaFold3](https://github.com/Ligo-Biosciences/AlphaFold3)
  - AlphaFold3 (DeepMind): [github.com/google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)

- Optional suggested layout (clone under project `third_party/`):

```bash
mkdir -p third_party
# Ligo Biosciences AlphaFold3
git clone https://github.com/Ligo-Biosciences/AlphaFold3 third_party/alphafold3_ligo
# DeepMind AlphaFold3
git clone https://github.com/google-deepmind/alphafold3 third_party/alphafold3_deepmind
```

Note: These are external dependencies under their respective licenses. The shims add `third_party/...` to `PYTHONPATH` via `scripts/dev.sh` so AF3 engines are importable at runtime.

## Architecture

- `cli/` (Rust): end-user CLI, result writing, and console summaries.
- `core/` (Rust): Python interop (PyO3) — loads and calls `py/deepmind_shim.py` or `py/ligo_shim.py`.
- `py/` (Python): engine shims and lightweight fallbacks.
  - `deepmind_shim.py`: Builds a synthetic DeepMind AF3 batch (features.BatchDict) and runs the actual `alphafold3.model.model.Model` forward via Haiku and JAX/XLA. Includes import-time stubs for optional C++ and RDKit modules.
  - `ligo_shim.py`: Builds a minimal Ligo AF3 config + synthetic batch and runs the actual PyTorch model. Includes a Triton-free MSA fallback (`py/src/models/components/msa_kernel.py`).
  - `py/src/models/components/msa_kernel.py`: A small replacement for a fused Triton op to keep the trunk path portable.

`scripts/dev.sh` sets a venv, `PYO3_PYTHON`, and `PYTHONPATH` so both shims can import third_party engines when invoked from Rust.

---

## Deep dive: `py/deepmind_shim.py`

Key responsibilities:
- Make JAX/Haiku and DeepMind AF3 importable under PyO3
  - Inserts venv site-packages to `sys.path`
  - Adds `third_party/alphafold3{,-deepmind,-AlphaFold3}/(src)` to `sys.path`
  - Stubs optional modules if missing: `alphafold3.cpp.*`, `rdkit`, and `zstandard`

- Build a synthetic AF3 `features.BatchDict` with consistent shapes
  - Padding shapes: `features.PaddingShapes(num_tokens=L, msa_size=16, num_chains=1, num_templates=0, num_atoms=ceil(L*D/32)*32)`
    - We use `D=32` dense atoms per token; `num_atoms` is padded to multiple of 32 to match atom-attention subsets.
  - Token features: `residue_index`, `token_index`, `aatype`, `seq_mask`, `seq_length`, `asym_id`, `entity_id`, `sym_id`, `is_*` booleans
  - MSA features: arrays of shape `[16, L]` (`msa`, `msa_mask`, `deletion_matrix`), plus per-token `profile (L,31)` and `deletion_mean (L,)`, with `num_alignments=16`
  - Templates: present but empty (`template_aatype/positions/mask` with 0 templates, padded by model as needed)
  - Reference structure: zeros for `ref_pos (L, D, 3)`, masks and metadata
  - Predicted structure info: `pred_dense_atom_mask (L,D)`, `residue_center_index (L,)`
  - Bonds: empty `GatherInfo` dicts for polymer-ligand and ligand-ligand bonds
  - Frames: `frames_mask (L,) = True`
  - Atom layouts and cross-attention:
    - Construct a per-token atom layout with only atom 0 valid (`'CA'`) and fill optional layout fields (element, residue name, chain type)
    - Compute `features.AtomCrossAtt.compute_features` to generate gather indices for token/atom queries and keys with `queries_subset_size=32`, `keys_subset_size=128`

- Configure and run the real Haiku model
  - Build `dm_model.Model.Config()` and set:
    - `config.num_recycles = 1` for trunk; `0` for full
    - `config.return_embeddings = False`, `config.return_distogram = False`
    - Diffusion steps: `2` in trunk mode, `8` in full mode for reasonable runtime
    - Cap Evoformer MSA use: `config.evoformer.num_msa = min(config.evoformer.num_msa, 16)` to match our constructed MSA
  - Transform and JIT:
    - `@hk.transform` → `init` with RNG and synthetic batch, `apply` JIT on target device
  - Measure `compile_ms` on a warmup call; then run N passes capturing `elapsed_ms`
  - Device selection: `--device cuda` will pick the first JAX GPU; use `CUDA_VISIBLE_DEVICES=<single_gpu>` to pin to one GPU
  - MPS maps to CPU (JAX limitation)

Returned JSON fields per pass:
- `engine, pass_index, start_ts, elapsed_ms, device, seq_len, notes, compile_ms, gpu_name, mode, ok` (and `error` if failure)

What is simplified vs. the full DeepMind pipeline:
- Random-init params (no official weights)
- Synthetic features (no external MSA/template databases)
- Reduced `num_recycles` and diffusion steps
- Import-time stubs for C++ and RDKit to avoid heavy system deps

Despite the simplifications, the JAX/Haiku graph is the real model; compute/memory/compilation reflect genuine execution.

---

## Deep dive: `py/ligo_shim.py`

Key responsibilities:
- Make Ligo AF3 importable and stable under PyO3
  - Insert venv site-packages and relevant third_party roots
  - Monkey patch `src.models.components.msa_kernel` with our pure-PyTorch fallback (`py/src/models/components/msa_kernel.py`) to avoid Triton dependency

- Build a minimal configuration
  - Small channel sizes for tokens/pairs/atoms
  - Short stacks for MSA and Pairformer blocks
  - Diffusion module configured but steps are light; in non-`full` mode the sampling is replaced by a no-op tensor on some backends

- Build a synthetic batch
  - `seq_len = L`, atoms per token = 4
  - Token indices (`residue_index`, `token_index`) and masks, `is_*` flags
  - Atom-wise reference tensors (`ref_pos`, `ref_mask`, `ref_element` one-hot limited to H/C/N/O)
  - Mapping features: `atom_to_token`, `token_atom_idx`, `token_repr_atom`, `ref_space_uid`
  - MSA features: `msa_feat (B, n_msa, L, 49)`, `msa_mask (B, n_msa, L)`

- Run the real PyTorch model
  - `AlphaFold3(cfg).eval()` on selected device (`cuda|mps|cpu`)
  - Warmup call measured as `compile_ms`; then N passes capturing `elapsed_ms`

Returned JSON fields mirror the DeepMind shim where applicable, including `compile_ms` and `gpu_name`.

Fallback kernel: `py/src/models/components/msa_kernel.py`
- Implements `MSAWeightedAveragingFused` as a pure-PyTorch op; matches expected interface so the trunk path runs without Triton.

---

## Rust crates

### `core/` — PyO3 interop
- Function: `af3_core::call_shim(engine, passes, device, seq_len, notes, full, mode)`
  - Imports the shim module (`deepmind_shim` or `ligo_shim`)
  - Calls `forward_once(...)` with kwargs
  - Converts Python result to JSON via Python’s `json.dumps` and returns a `serde_json::Value`

### `cli/` — Command-line interface
- Flags:
  - `--engine {deepmind|ligo|both}` (default: both)
  - `--mode {toy|trunk|full}` (default: trunk)
  - `--device {cpu|mps|cuda}` (default: cpu)
  - `--passes N` (default: 1 unless `--dry-run`)
  - `--seq-len L` (default: 32)
  - `--full` (alias for full path; kept for compatibility)
  - `--notes`, `--out results/<dir>`
- Writes `results/<timestamp>/{deepmind,ligo}.jsonl` and a `summary.csv` with host and commit metadata. Prints per-engine min/median/p95.

---

## Setup (Linux/CUDA single-GPU)

1) Create and activate venv, then source environment:
```bash
cd /home/adminsteve/ceph/mlx-server/k8s/metaphor-genie/af3-bench
python3 -m venv .venv
source .venv/bin/activate
source scripts/dev.sh
```

2) Install Python deps (examples):
```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install jax[cuda12_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install haiku jaxtyping typeguard==2.13.3 numpy
```

3) Build CLI:
```bash
cargo build
```

4) Pin to a single GPU (example: GPU 1) and recommended XLA memory env:
```bash
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85
```

---

## Usage (single GPU)

Trunk, 5 passes (both engines):
```bash
export CUDA_VISIBLE_DEVICES=1; source scripts/dev.sh; \
cargo run -q --manifest-path cli/Cargo.toml -- run --engine both --device cuda --seq-len 64 --passes 5 --mode trunk
```

Full, 5 passes (both engines):
```bash
export CUDA_VISIBLE_DEVICES=1; source scripts/dev.sh; \
cargo run -q --manifest-path cli/Cargo.toml -- run --engine both --device cuda --seq-len 64 --passes 5 --mode full --full
```

Single-engine examples (1 pass):
```bash
export CUDA_VISIBLE_DEVICES=1; source scripts/dev.sh; \
cargo run -q --manifest-path cli/Cargo.toml -- run --engine deepmind --device cuda --seq-len 64 --passes 1 --mode trunk

export CUDA_VISIBLE_DEVICES=1; source scripts/dev.sh; \
cargo run -q --manifest-path cli/Cargo.toml -- run --engine deepmind --device cuda --seq-len 64 --passes 1 --mode full --full
```

Outputs:
- `results/<timestamp>/deepmind.jsonl`
- `results/<timestamp>/ligo.jsonl`
- `results/<timestamp>/summary.csv`

Each JSONL line contains timings and metadata per pass. CSV aggregates basic stats.

---

## How to modify behavior

DeepMind (`py/deepmind_shim.py`):
- MSA rows: change `msa_n` (default 16) and `features.PaddingShapes.msa_size`. Also cap `config.evoformer.num_msa` accordingly.
- Atom subsets: tune `queries_subset_size` (default 32) and `keys_subset_size` (default 128).
- Diffusion steps: set `config.heads.diffusion.eval.steps` (2 in trunk, 8 in full by default here).
- Recycles: set `config.num_recycles` (1 in trunk, 0 in full by default here).
- Device: control with `--device` and `CUDA_VISIBLE_DEVICES`.

Ligo (`py/ligo_shim.py`):
- Model depth/channels: edit `_build_minimal_ligo_config()` (`pairformer_stack.no_blocks`, channel sizes, diffusion heads/blocks).
- Diffusion sampling: trunk mode may stub sampling depending on backend; force real sampling by running `--mode full`.
- Synthetic batch: change atoms per token (default 4) and feature sizes in `_build_dummy_batch()`.
- Fused MSA kernel: see `py/src/models/components/msa_kernel.py` if you want to swap back to Triton.

Rust CLI (`cli/src/main.rs`):
- Extend flags, add warmup-only passes, or new report columns. The CLI writes JSONL per engine and a combined CSV.

---

## Troubleshooting

- JAX OOM with DeepMind full:
  - Pick a free GPU via `export CUDA_VISIBLE_DEVICES=<id>`
  - Reduce `config.heads.diffusion.eval.steps` or increase `XLA_PYTHON_CLIENT_MEM_FRACTION` cautiously
  - Ensure `XLA_PYTHON_CLIENT_PREALLOCATE=false`

- Import errors under PyO3:
  - Always `source scripts/dev.sh`
  - Verify `.venv` packages: `torch`, `jax`, `dm-haiku`, `numpy`, `typeguard==2.13.3`

---

## Sample results (CUDA, single GPU)

DeepMind trunk (5 passes, L=64):
```
min≈50.53ms median≈51.999ms p95≈53.632ms
```

DeepMind full (5 passes, L=64):
```
min≈66.964ms median≈69.988ms p95≈72.064ms
```

Ligo trunk (5 passes, L=64):
```
min≈19.01ms median≈19.05ms p95≈19.95ms
```

Ligo full (5 passes, L=64):
```
median≈1.61s (diffusion sampling)
```

Exact JSONL artifacts are saved in `results/<timestamp>/` and include `compile_ms` for the first compiled call.

---

## Notes on “real inference”

These runs execute the genuine model graphs end-to-end (JAX/Haiku or PyTorch), but with:
- Random-initialized parameters (no released AF3 weights)
- Synthetic features instead of database-driven MSA/templates
- Reduced recycles and diffusion steps for repeatable performance benchmarking

Therefore outputs are not meaningful structures, but compilation/memory/runtime are representative of the true compute.
