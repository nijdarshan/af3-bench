# AF3 Bench — Requirements & Plan

## 1) Goal (what we’re doing)

Build a **Rust benchmark CLI** that measures **model inference speed** (CPU now; GPU later) for:

* **DeepMind AlphaFold 3** (baseline), and
* **Ligo-Biosciences AlphaFold3** (second engine).

We will **not** train models or run the heavy preprocessing pipeline. We’ll do **one or many forward passes** on **dummy/minimal inputs** with **random-initialized parameters**, exactly as the client requested, to compare **inference-time performance only**.

## 2) What already exists (inputs to us)

* Two Python repos:

  * `google-deepmind/alphafold3` (official AF3 inference code; weights gated, but we don’t need them for timing a random init).
  * `Ligo-Biosciences/AlphaFold3` (independent implementation).
* Client has a **separate preprocessing pipeline**; we **ignore** it for this benchmark.
* We’ve prepared a project skeleton (`af3-bench/`) with:

  * `third_party/` containing both repos (main branches),
  * a Python **venv**,
  * environment helper `scripts/dev.sh`.

## 3) Scope & non-goals

**In scope**

* Rust CLI that:

  * Calls each repo’s **inference** via Python (PyO3),
  * Runs **one forward pass** (`--dry-run`) or **N passes** (`--passes N`),
  * Records **timings** and writes **JSONL/CSV** results.
* CPU runs now; **GPU later** without changing the CLI interface.

**Out of scope**

* Training, dataset downloads, template/MSA search, or end-to-end accuracy checks.
* Any redistribution of gated weights/artifacts.

## 4) Architecture (how we’ll do it)

* **Rust CLI** (top-level UX, flags, output).
* **Rust core** (PyO3 glue, metrics structs, file I/O).
* **Python shims** (one per engine):

  * Import the engine repo,
  * Build the model with **random params**,
  * Create **minimal dummy features** acceptable by the model,
  * Run `forward_once(...)` for **N passes** on **CPU** (later: `cuda/rocm`),
  * Return per-pass timing and basic metadata (e.g., seq length, output shapes).

### Why shims?

We keep all repo-specific API quirks in tiny Python files so the Rust code stays clean and stable even if repo internals change.

## 5) Runtime flow

1. User runs CLI:
   `af3-bench run --engine both --dry-run` (or `--passes 5`)
2. CLI initializes PyO3 and imports `deepmind_shim` and/or `ligo_shim`.
3. Shim **loops N** times and times each pass.
4. CLI collects results and writes:

   * **JSONL** (one line per pass),
   * **CSV** summary,
   * Console table.

> Note: JAX/XLA typically **compiles on first call**. For CPU dry runs, we’ll include that in the first-pass time. Later (GPU), we’ll add a **warmup** and split **compile vs execute**.

## 6) CLI UX (initial)

```
af3-bench run
  --engine {deepmind|ligo|both}   # default: both
  --passes <N>                    # default: 1
  --dry-run                       # alias for --passes 1
  --device {cpu|cuda|rocm}        # default: cpu (later)
  --seq-len <int>                 # default: small value for quick CPU tests
  --out <dir>                     # default: ./results/<timestamp>
  --notes "<free text>"
```

## 7) Timing methodology

* **Per pass**: wall-clock elapsed (ms).
* (Later) Optional split:

  * **compile\_ms** (first pass JIT),
  * **exec\_ms** (steady-state).
* **Metadata captured**: engine name, pass index, device, seq\_len, datetime, git commit of each engine repo, host info (CPU/GPU name), and any user-supplied `--notes`.

## 8) Outputs (artifacts)

* `results/<ts>/deepmind.jsonl` and `ligo.jsonl`
  (JSON lines: one object per pass)
* `results/<ts>/summary.csv`
  (engine, pass\_index, elapsed\_ms, device, seq\_len, commit, notes)
* Console: quick summary table + path to artifacts.

## 9) Environments & setup (already covered)

* macOS dev: local venv, PyO3 pointed at venv’s Python, `PYTHONPATH` includes `py/` and both `third_party` repos.
* Linux/GPU later: identical CLI; shims will accept `--device` and do the same calls.

## 10) Security & licensing

* We **won’t** download/share AF3 weights.
* Any weights or artifacts shared by the client stay local and unshared.
* Only **random-init** timing is required for the initial proof.

## 11) Risks & mitigations

* **Repo API mismatch** → We confine adaptation in shims; minimal code surface.
* **JAX on macOS** → If CPU JAX install fails, we still verify Rust↔Python plumbing; forward-pass runs can be executed on a Linux host without changing the CLI.
* **JIT affecting timing** → We’ll label first-pass times clearly; later add warmup and compile/exec split.

## 12) Acceptance criteria (for this phase)

* `af3-bench run --engine both --dry-run` completes on CPU with **both engines**, writing JSONL + CSV.
* Output contains **per-pass timings** and required metadata.
* Repo commits are recorded for reproducibility.

---

# Next steps (actionable, no code pasted yet)

### 1) Wiring Rust ↔ Python (PyO3) without pain

* Ensure the shell is prepared by running `./scripts/dev.sh` before building/running.
* Confirm environment vars:

  * `PYO3_PYTHON="<repo>/.venv/bin/python"`
  * `PYTHONPATH` includes:

    * `./py` (our shims),
    * `./third_party/alphafold3-deepmind`,
    * `./third_party/alphafold3-ligo`.
* The CLI will call: `import <engine>_shim; forward_once(passes=N, device="cpu", seq_len=...)`.
  The shim loops N times and times each pass; first-call JAX compile stays **inside** the shim for now.

### 2) Dry run vs multi-pass

* `--dry-run` → `--passes 1` (smoketest wiring: one forward per engine).
* Real benches use `--passes {2,3,5,...}`.
* Later (GPU), shims gain:

  * `--warmup <k>`,
  * separate `compile_ms` vs `exec_ms`,
  * device pinning (`cpu`/`cuda`/`rocm`).

### 3) Outputs (for easy diffing)

* **JSONL**: `{engine, pass_index, start_ts, elapsed_ms, seq_len, device, notes, commit}`.
* **CSV summary**: same fields for quick spreadsheet views.
* Console prints: per-engine stats (min/median/p95) and artifact paths.

### 4) Day-0 checklist

* Install rustup; create Rust workspace with two crates: `cli/` and `core/`.
* (Already done) Create venv; set `PYO3_PYTHON` and `PYTHONPATH`.
* Add two **stub shims** returning `{"ok": true}` and prove Rust→Python import via PyO3.
* Implement **stub timing**: call shims with `--passes 1`, print a dummy JSON line per engine.
* Upgrade shims from stub → **random-init model + forward once (CPU)** with tiny `seq_len` so it returns fast on Mac.

### 5) When GPUs arrive (no CLI change)

* Add `--device cuda` support to shims.
* Add `--warmup 1` and split timing where possible.
* Run larger `--passes` (e.g., 20) and export complete reports.

### 6) macOS gotchas & escapes

* Always run `./scripts/dev.sh` before `cargo run` to align Python.
* If JAX on macOS is flaky, keep going: verify imports and CLI plumbing locally; run the real forward on a Linux or GPU box with the same CLI and shims.
* If repo APIs don’t expose “init + forward,” implement that tiny glue **inside the shim**, not in Rust.
