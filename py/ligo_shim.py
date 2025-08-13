from __future__ import annotations

import time
from typing import Any, Dict, List
from datetime import datetime, timezone
import sys, os, warnings
# Suppress noisy warnings during benchmarking
warnings.filterwarnings("ignore")
# Ensure venv site-packages is on sys.path for embedded Python
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

# Ensure Ligo repo root and its src/ are on sys.path (support multiple repo folder names)
try:
    third_party = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party'))
    candidates = [
        os.path.join(third_party, 'alphafold3-ligo'),
        os.path.join(third_party, 'AlphaFold3'),
        os.path.join(third_party, 'alphafold3'),
    ]
    for root in candidates:
        src = os.path.join(root, 'src')
        for p in [root, src]:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
except Exception:
    pass


def hello() -> Dict[str, Any]:
    return {"engine": "ligo", "ok": True, "message": "hello from ligo_shim"}


def _build_minimal_ligo_config() -> Any:
    # Lightweight config object supporting both attribute and item access
    class C(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)
        def __setattr__(self, k, v):
            self[k] = v
    # Use sizes that align with common AF3 trunk defaults
    c_token = 384
    c_pair = 128
    c_atom = 64
    c_atompair = 32
    cfg = C()
    cfg.globals = C(
        chunk_size=None,
        use_deepspeed_evo_attention=False,
        use_triton_kernel=False,
        samples_per_trunk=1,
        rollout_samples_per_trunk=1,
        eps=1e-8,
        matmul_precision="high",
        clear_cache_between_blocks=False,
    )
    cfg["input_embedder"] = C(
        c_token=c_token,
        c_trunk_pair=c_pair,
        c_atom=c_atom,
        c_atompair=c_atompair,
    )
    cfg["msa_module"] = C(
        no_blocks=1,
        c_msa=128,
        c_token=c_token,
        c_z=c_pair,
        c_hidden=128,
        no_heads=4,
        c_hidden_tri_mul=64,
        c_hidden_pair_attn=64,
        no_heads_tri_attn=4,
        transition_n=2,
        pair_dropout=0.0,
        fuse_projection_weights=False,
        clear_cache_between_blocks=False,
        blocks_per_ckpt=1,
        inf=1e8,
    )
    cfg["pairformer_stack"] = C(
        c_s=c_token,
        c_z=c_pair,
        no_blocks=4,
        c_hidden_mul=128,
        c_hidden_pair_attn=64,
        no_heads_tri_attn=4,
        no_heads_single_attn=8,
        transition_n=2,
        pair_dropout=0.0,
        fuse_projection_weights=False,
        blocks_per_ckpt=1,
        clear_cache_between_blocks=False,
        inf=1e8,
    )
    cfg["diffusion_module"] = C(
        c_atom=c_atom,
        c_atompair=c_atompair,
        c_token=c_token,
        c_tokenpair=c_pair,
        atom_encoder_blocks=1,
        atom_encoder_heads=4,
        dropout=0.0,
        atom_attention_n_queries=32,
        atom_attention_n_keys=128,
        atom_decoder_blocks=1,
        atom_decoder_heads=4,
        token_transformer_blocks=2,
        token_transformer_heads=4,
        sd_data=16.0,
        s_max=16.0,
        s_min=0.0004,
        p=7.0,
        clear_cache_between_blocks=False,
        blocks_per_ckpt=1,
    )
    cfg["distogram_head"] = C(
        c_z=c_pair,
        no_bins=32,
    )
    return cfg


def _select_device(device: str):
    import torch
    req = device.lower()
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if req == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _build_dummy_batch(seq_len: int, device: str = "cpu") -> "Dict[str, Any]":
    import torch
    dev, _ = _select_device(device)
    bs = 1
    n_tokens = int(seq_len)
    n_atoms = n_tokens * 4
    n_msa = 1
    n_cycle = 1

    def add_cycle(x):
        return x.unsqueeze(-1)  # add recycling dim

    # Token-level indices
    residue_index = torch.arange(n_tokens, device=dev).view(1, n_tokens)
    token_index = residue_index.clone()
    # Atom-to-token mapping (4 atoms per token)
    token_indices = torch.arange(n_tokens, device=dev)
    atom_to_token = torch.repeat_interleave(token_indices, 4).view(1, n_atoms)
    # Within-token atom index 0..3
    token_atom_idx = (torch.arange(n_atoms, device=dev) % 4).view(1, n_atoms)
    # Representative atom index per token: first atom of each token in flat layout
    token_repr_atom = (torch.arange(n_tokens, device=dev) * 4).view(1, n_tokens)
    # Space UID: same as token index for simplicity
    ref_space_uid = atom_to_token.clone()

    batch: Dict[str, Any] = {
        # Token-wise
        "residue_index": add_cycle(residue_index.expand(bs, -1)),
        "token_index": add_cycle(token_index.expand(bs, -1)),
        "asym_id": add_cycle(torch.zeros((bs, n_tokens), dtype=torch.long, device=dev)),
        "entity_id": add_cycle(torch.zeros((bs, n_tokens), dtype=torch.long, device=dev)),
        "sym_id": add_cycle(torch.zeros((bs, n_tokens), dtype=torch.long, device=dev)),
        "aatype": add_cycle(torch.randint(0, 21, (bs, n_tokens), dtype=torch.long, device=dev)),
        "is_protein": add_cycle(torch.ones((bs, n_tokens), dtype=torch.float32, device=dev)),
        "is_rna": add_cycle(torch.zeros((bs, n_tokens), dtype=torch.float32, device=dev)),
        "is_dna": add_cycle(torch.zeros((bs, n_tokens), dtype=torch.float32, device=dev)),
        "is_ligand": add_cycle(torch.zeros((bs, n_tokens), dtype=torch.float32, device=dev)),
        "token_mask": add_cycle(torch.ones((bs, n_tokens), dtype=torch.float32, device=dev)),

        # Atom-wise
        "ref_pos": add_cycle(torch.zeros((bs, n_atoms, 3), dtype=torch.float32, device=dev)),
        "ref_mask": add_cycle(torch.ones((bs, n_atoms), dtype=torch.float32, device=dev)),
        # Element one-hot limited to 4 per model impl (H,C,N,O)
        "ref_element": add_cycle(torch.zeros((bs, n_atoms, 4), dtype=torch.float32, device=dev)),
        "ref_charge": add_cycle(torch.zeros((bs, n_atoms), dtype=torch.float32, device=dev)),
        # Provide [N_atom, 4] so downstream reshape(batch,n_atoms,4) is valid
        "ref_atom_name_chars": add_cycle(torch.zeros((bs, n_atoms, 4), dtype=torch.float32, device=dev)),
        "ref_space_uid": add_cycle(ref_space_uid.to(torch.long)),

        # MSA features
        "msa_feat": add_cycle(torch.zeros((bs, n_msa, n_tokens, 49), dtype=torch.float32, device=dev)),
        "msa_mask": add_cycle(torch.ones((bs, n_msa, n_tokens), dtype=torch.float32, device=dev)),

        # Mapping features
        "atom_to_token": add_cycle(atom_to_token.to(torch.long)),
        "token_atom_idx": add_cycle(token_atom_idx.to(torch.long)),
        "token_repr_atom": add_cycle(token_repr_atom.to(torch.long)),

        # Training-time placeholders (not used in inference path)
        "all_atom_positions": add_cycle(torch.zeros((bs, n_atoms, 3), dtype=torch.float32, device=dev)),
        "atom_mask": add_cycle(torch.ones((bs, n_atoms), dtype=torch.float32, device=dev)),
        "atom_exists": add_cycle(torch.ones((bs, n_atoms, 3), dtype=torch.float32, device=dev)),
    }
    return batch


def forward_once(
    *, passes: int = 1, device: str = "cpu", seq_len: int = 32, notes: str | None = None, full: bool = False, mode: str = "trunk"
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        import torch
        # Point imports so that af3_fallback.msa_kernel shadows Triton usage
        import sys, os
        os.environ.setdefault('PYTHONPATH', '')
        # Prepend fallback to sys.path
        fallback_dir = os.path.join(os.path.dirname(__file__), 'af3_fallback')
        if fallback_dir not in sys.path:
            sys.path.insert(0, fallback_dir)
        # Monkey patch before importing model: replace msa_kernel import
        import importlib
        sys.modules['src.models.components.msa_kernel'] = importlib.import_module('msa_kernel')
        from src.models.model import AlphaFold3
        # Toy mode: use a tiny torch op to mirror DeepMind toy JAX
        if mode.lower() == 'toy':
            import torch
            torch.set_grad_enabled(False)
            dev, resolved_device = _select_device(device)
            x = torch.ones((max(4, int(seq_len)), 8), device=dev, dtype=torch.float32)
            w = torch.ones((8, 8), device=dev, dtype=torch.float32)
            # warmup
            _ = torch.tanh(x @ w).sum().item()
            for i in range(passes):
                start_ts = datetime.now(timezone.utc).isoformat()
                start = time.time()
                _ = torch.tanh(x @ w).sum().item()
                elapsed_ms = (time.time() - start) * 1000.0
                results.append({
                    "engine": "ligo",
                    "pass_index": i,
                    "start_ts": start_ts,
                    "elapsed_ms": elapsed_ms,
                    "device": resolved_device,
                    "seq_len": int(seq_len),
                    "notes": notes,
                    "ok": True,
                })
            return results

        # Skip heavy diffusion unless explicitly full mode
        resolved_device_name = device.lower()
        if (resolved_device_name in ('mps','cpu','cuda') and mode.lower() != 'full') or (resolved_device_name == 'mps' and not full):
            try:
                import src.models.diffusion_module as _dm
                import torch as _torch
                def _mps_safe_sample(self, features, s_inputs, s_trunk, z_trunk, n_steps: int = 1, samples_per_trunk: int = 1, **kwargs):
                    bs, n_atoms, _ = features["ref_pos"].shape
                    return _torch.zeros((bs, 1, n_atoms, 3), dtype=_torch.float32, device=s_inputs.device)
                _dm.DiffusionModule.sample = _mps_safe_sample
            except Exception:
                pass

        torch.set_grad_enabled(False)
        cfg = _build_minimal_ligo_config()
        torch_device, resolved_device = _select_device(device)
        model = AlphaFold3(cfg).to(torch_device).eval()
        batch = _build_dummy_batch(max(4, int(seq_len)), device=resolved_device)

        # Warmup (measure initial pass as compile/init time proxy)
        warmup_start = time.time()
        _ = model({k: v.clone() for k, v in batch.items()}, train=False)
        compile_ms = (time.time() - warmup_start) * 1000.0

        for i in range(passes):
            start_ts = datetime.now(timezone.utc).isoformat()
            start = time.time()
            _ = model({k: v.clone() for k, v in batch.items()}, train=False)
            elapsed_ms = (time.time() - start) * 1000.0
            gpu_name = None
            try:
                if resolved_device == 'cuda':
                    gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None
            results.append({
                "engine": "ligo",
                "pass_index": i,
                "start_ts": start_ts,
                "elapsed_ms": elapsed_ms,
                "device": resolved_device,
                "seq_len": int(seq_len),
                "notes": notes,
                "compile_ms": compile_ms,
                "gpu_name": gpu_name,
                "mode": mode,
                "ok": True,
            })
        return results
    except Exception as e:
        # Fallback to stub if imports or forward fail
        for i in range(passes):
            start_ts = datetime.now(timezone.utc).isoformat()
            start = time.time()
            time.sleep(0.01)
            elapsed_ms = (time.time() - start) * 1000.0
            results.append(
                {
                    "engine": "ligo",
                    "pass_index": i,
                    "start_ts": start_ts,
                    "elapsed_ms": elapsed_ms,
                    "device": device,
                    "seq_len": seq_len,
                    "notes": notes,
                    "ok": False,
                    "error": str(e.__class__.__name__) + ": " + str(e),
                }
            )
        return results

