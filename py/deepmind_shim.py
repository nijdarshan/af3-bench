from __future__ import annotations

import time
from typing import Any, Dict, List
import warnings
import sys, os
warnings.filterwarnings("ignore")
# Ensure venv site-packages visible under embedded Python
try:
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        site_pkgs = os.path.join(venv, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
    else:
        site_pkgs = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
    if site_pkgs not in sys.path:
        sys.path.insert(0, site_pkgs)
except Exception:
    pass
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone


def hello() -> Dict[str, Any]:
    return {"engine": "deepmind", "ok": True, "message": "hello from deepmind_shim"}


def _select_device(device: str) -> str:
    # JAX on macOS MPS is not supported; map mps->cpu
    if device.lower() == "mps":
        return "cpu"
    return device.lower()


def forward_once(
    *, passes: int = 1, device: str = "cpu", seq_len: int = 32, notes: str | None = None, full: bool = False, mode: str = "trunk"
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    resolved_device = _select_device(device)
    try:
        # Attempt a minimal JAX compute to verify environment; AF3 full init is Linux/CUDA target.
        import jax
        import jax.numpy as jnp
        # Toy mode: small matmul+tanh to mirror Ligo toy
        @jax.jit
        def _toy(x, w):
            return jnp.tanh(x @ w).sum()
        m = max(4, int(seq_len))
        x = jnp.ones((m, 8), dtype=jnp.float32)
        w = jnp.ones((8, 8), dtype=jnp.float32)
        _ = _toy(x, w).block_until_ready()
        for i in range(passes):
            start_ts = datetime.now(timezone.utc).isoformat()
            start = time.time()
            _ = _toy(x, w).block_until_ready()
            elapsed_ms = (time.time() - start) * 1000.0
            results.append({
                "engine": "deepmind",
                "pass_index": i,
                "start_ts": start_ts,
                "elapsed_ms": elapsed_ms,
                "device": resolved_device,
                "seq_len": int(seq_len),
                "notes": notes,
                "ok": True,
            })
        return results
    except Exception as e:
        # No JAX or environment not set; return informative stub
        for i in range(passes):
            start_ts = datetime.now(timezone.utc).isoformat()
            start = time.time()
            time.sleep(0.01)
            elapsed_ms = (time.time() - start) * 1000.0
            results.append(
                {
                    "engine": "deepmind",
                    "pass_index": i,
                    "start_ts": start_ts,
                    "elapsed_ms": elapsed_ms,
                    "device": resolved_device,
                    "seq_len": int(seq_len),
                    "notes": notes,
                    "ok": False,
                    "error": f"{e.__class__.__name__}: {e}",
                }
            )
        return results

