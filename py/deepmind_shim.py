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

# Ensure DeepMind repo root and its src/ are on sys.path (support multiple repo folder names)
try:
    third_party = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party'))
    candidates = [
        os.path.join(third_party, 'alphafold3'),
        os.path.join(third_party, 'AlphaFold3'),
    ]
    for root in candidates:
        src = os.path.join(root, 'src')
        for p in [root, src]:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
except Exception:
    pass


def _build_dm_synthetic_batch(seq_len: int) -> "Dict[str, Any]":
    """Construct a minimal DeepMind AF3 BatchDict with valid shapes.

    - Random-init model forward only; no weights or external DBs.
    - Single chain, protein-only tokens; trivial MSA, frames, ref structure.
    - AtomCrossAtt gather indices computed via library for consistency.
    """
    import numpy as np
    from alphafold3.model import features
    from alphafold3.model.atom_layout import atom_layout as al

    L = int(max(4, seq_len))
    D = 4  # atoms per token for masks/positions
    # Total atoms rounded up to a multiple of queries subset size (32)
    queries_subset_size = 32
    keys_subset_size = 128
    total_atoms = L * D
    num_atoms = int(np.ceil(total_atoms / queries_subset_size) * queries_subset_size)

    padding = features.PaddingShapes(
        num_tokens=L,
        msa_size=1,
        num_chains=1,
        num_templates=0,
        num_atoms=num_atoms,
    )

    # Token features (single chain A, protein)
    residue_index = np.arange(L, dtype=np.int32)
    token_index = np.arange(1, L + 1, dtype=np.int32)
    aatype = np.zeros(L, dtype=np.int32)
    seq_mask = np.ones(L, dtype=bool)
    seq_length = np.array(L, dtype=np.int32)
    asym_id = np.ones(L, dtype=np.int32)
    entity_id = np.ones(L, dtype=np.int32)
    sym_id = np.ones(L, dtype=np.int32)
    is_protein = np.ones(L, dtype=bool)
    is_rna = np.zeros(L, dtype=bool)
    is_dna = np.zeros(L, dtype=bool)
    is_ligand = np.zeros(L, dtype=bool)
    is_nonstandard_polymer_chain = np.zeros(L, dtype=bool)
    is_water = np.zeros(L, dtype=bool)

    # MSA minimal (1 row)
    msa = np.zeros((1, L), dtype=np.int8)
    msa_mask = np.ones((1, L), dtype=bool)
    deletion_matrix = np.zeros((1, L), dtype=np.int8)
    # Profile has 31 channels
    profile = np.zeros((L, 31), dtype=np.float32)
    deletion_mean = np.zeros((L,), dtype=np.float32)
    num_alignments = np.array(1, dtype=np.int32)

    # Reference structure: zeros with masks
    ref_pos = np.zeros((L, D, 3), dtype=np.float32)
    ref_mask = np.ones((L, D), dtype=bool)
    ref_element = np.zeros((L, D), dtype=np.int32)
    ref_charge = np.zeros((L, D), dtype=np.float32)
    ref_atom_name_chars = np.zeros((L, D, 4), dtype=np.int32)
    # Give each token its own reference space id
    ref_space_uid = np.repeat(np.arange(L, dtype=np.int32)[:, None], D, axis=1)

    # PredictedStructureInfo minimal
    pred_dense_atom_mask = np.ones((L, D), dtype=bool)
    residue_center_index = np.zeros((L,), dtype=np.int32)

    # PseudoBeta: map each token to its first atom (index 0)
    t_idx = np.arange(L, dtype=np.int64)
    pb_gather_idxs = t_idx  # shape (L,)
    pb_gather_mask = np.ones((L,), dtype=bool)
    pb_input_shape = np.array((L, D), dtype=np.int64)

    # Polymer/Ligand bonds: empty gather infos with valid shapes
    # tokens_to_polymer_ligand_bonds: (L, 2)
    pl_gather_shape = (L, 2)
    empty_gather_idxs = np.zeros(pl_gather_shape, dtype=np.int64)
    empty_gather_mask = np.zeros(pl_gather_shape, dtype=bool)
    tokens_to_polymer_ligand_bonds = {
        'tokens_to_polymer_ligand_bonds:gather_idxs': empty_gather_idxs,
        'tokens_to_polymer_ligand_bonds:gather_mask': empty_gather_mask,
        'tokens_to_polymer_ligand_bonds:input_shape': np.array((L,), dtype=np.int64),
    }
    token_atoms_to_polymer_ligand_bonds = {
        'token_atoms_to_polymer_ligand_bonds:gather_idxs': empty_gather_idxs,
        'token_atoms_to_polymer_ligand_bonds:gather_mask': empty_gather_mask,
        'token_atoms_to_polymer_ligand_bonds:input_shape': np.array((L, D), dtype=np.int64),
    }
    # Ligand-ligand bonds: (L, 2) all masked out
    ll_gather = {
        'tokens_to_ligand_ligand_bonds:gather_idxs': empty_gather_idxs,
        'tokens_to_ligand_ligand_bonds:gather_mask': empty_gather_mask,
        'tokens_to_ligand_ligand_bonds:input_shape': np.array((L,), dtype=np.int64),
    }

    # Frames
    frames_mask = np.ones((L,), dtype=bool)

    # Build a minimal atom layout for per-atom attention indices
    # Atom names: first column valid, rest padding
    atom_name = np.empty((L, D), dtype=object)
    atom_name[:, :] = ''
    atom_name[:, 0] = 'CA'
    res_id = np.repeat(np.arange(1, L + 1, dtype=int)[:, None], D, axis=1)
    chain_id = np.empty((L, D), dtype=object)
    chain_id[:, :] = 'A'
    all_token_atoms_layout = al.AtomLayout(
        atom_name=atom_name,
        res_id=res_id,
        chain_id=chain_id,
    )

    # Compute AtomCrossAtt gather infos via library to ensure consistency
    atom_cross = features.AtomCrossAtt.compute_features(
        all_token_atoms_layout=all_token_atoms_layout,
        queries_subset_size=queries_subset_size,
        keys_subset_size=keys_subset_size,
        padding_shapes=padding,
    )
    atom_cross_dict = atom_cross.as_data_dict()

    batch: Dict[str, Any] = {
        # Token features
        'residue_index': residue_index,
        'token_index': token_index,
        'aatype': aatype,
        'seq_mask': seq_mask,
        'seq_length': seq_length,
        'asym_id': asym_id,
        'entity_id': entity_id,
        'sym_id': sym_id,
        'is_protein': is_protein,
        'is_rna': is_rna,
        'is_dna': is_dna,
        'is_ligand': is_ligand,
        'is_nonstandard_polymer_chain': is_nonstandard_polymer_chain,
        'is_water': is_water,

        # MSA
        'msa': msa,
        'msa_mask': msa_mask,
        'deletion_matrix': deletion_matrix,
        'profile': profile,
        'deletion_mean': deletion_mean,
        'num_alignments': num_alignments,

        # Templates (empty)
        'template_aatype': np.zeros((0, L), dtype=np.int32),
        'template_atom_positions': np.zeros((0, L, 24, 3), dtype=np.float32),
        'template_atom_mask': np.zeros((0, L, 24), dtype=bool),

        # Ref structure
        'ref_pos': ref_pos,
        'ref_mask': ref_mask,
        'ref_element': ref_element,
        'ref_charge': ref_charge,
        'ref_atom_name_chars': ref_atom_name_chars,
        'ref_space_uid': ref_space_uid,

        # Predicted structure info
        'pred_dense_atom_mask': pred_dense_atom_mask,
        'residue_center_index': residue_center_index,

        # Bonds
        **tokens_to_polymer_ligand_bonds,
        **token_atoms_to_polymer_ligand_bonds,
        **ll_gather,

        # Pseudo beta mapping
        'token_atoms_to_pseudo_beta:gather_idxs': pb_gather_idxs,
        'token_atoms_to_pseudo_beta:gather_mask': pb_gather_mask,
        'token_atoms_to_pseudo_beta:input_shape': pb_input_shape,

        # Frames
        'frames_mask': frames_mask,

        # Atom cross attention gather infos
        **atom_cross_dict,
    }
    return batch


def hello() -> Dict[str, Any]:
    return {"engine": "deepmind", "ok": True, "message": "hello from deepmind_shim"}


def _select_device(device: str) -> str:
    # JAX on macOS MPS is not supported; map mps->cpu
    d = device.lower()
    if d == "mps":
        return "cpu"
    return d


def forward_once(
    *, passes: int = 1, device: str = "cpu", seq_len: int = 32, notes: str | None = None, full: bool = False, mode: str = "trunk"
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    resolved_device = _select_device(device)
    try:
        # Attempt real AF3 Haiku model forward (random-init) when trunk/full.
        import jax
        import jax.numpy as jnp
        import haiku as hk
        from alphafold3.model import model as dm_model
        from alphafold3.model import features as dm_features

        # Select device (cuda if available and requested)
        target = None
        try:
            if resolved_device == 'cuda':
                gpus = [d for d in jax.devices() if d.platform == 'gpu']
                if gpus:
                    target = gpus[0]
            if target is None:
                target = jax.devices()[0]
        except Exception:
            target = None

        # Toy mode retains lightweight JAX path for sanity
        if mode.lower() == 'toy':
            @jax.jit
            def _toy(x, w):
                return jnp.tanh(x @ w).sum()
            x = jnp.ones((max(4, int(seq_len)), 8), dtype=jnp.float32)
            w = jnp.ones((8, 8), dtype=jnp.float32)
            if target is not None:
                x, w = jax.device_put(x, target), jax.device_put(w, target)
            t0 = time.time()
            _ = _toy(x, w).block_until_ready()
            compile_ms = (time.time() - t0) * 1000.0
            gpu_name = None
            try:
                gpus = [d for d in jax.devices() if d.platform == 'gpu']
                if gpus:
                    gpu_name = str(gpus[0])
            except Exception:
                gpu_name = None
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
                    "compile_ms": compile_ms,
                    "gpu_name": gpu_name,
                    "mode": mode,
                    "ok": True,
                })
            return results

        # Build synthetic batch for DM model
        np_batch = _build_dm_synthetic_batch(int(seq_len))

        # Configure model: small num_recycles, diffusion steps depending on mode
        config = dm_model.Model.Config()
        # Reduce work for benchmarking
        config.num_recycles = 1 if mode.lower() == 'trunk' else 0
        # Keep embeddings off by default
        config.return_embeddings = False
        config.return_distogram = False
        # Lower diffusion steps for full mode to keep runtime reasonable
        if mode.lower() == 'full':
            config.heads.diffusion.eval.steps = 8
            config.heads.diffusion.eval.num_samples = 1
        else:
            # trunk: still call model; diffusion path will run with small steps
            config.heads.diffusion.eval.steps = 2
            config.heads.diffusion.eval.num_samples = 1

        @hk.transform
        def _forward(batch_dict):
            return dm_model.Model(config)(batch_dict)

        # Initialize random params (no weights)
        rng = jax.random.PRNGKey(0)
        # Minimal init requires a small example; reuse our batch
        init_params = _forward.init(rng, np_batch)

        apply = jax.jit(_forward.apply, device=target)

        # Warmup to capture compile time
        t0 = time.time()
        _ = apply(init_params, rng, np_batch)
        jax.block_until_ready(_)
        compile_ms = (time.time() - t0) * 1000.0

        gpu_name = None
        try:
            gpus = [d for d in jax.devices() if d.platform == 'gpu']
            if gpus:
                gpu_name = str(gpus[0])
        except Exception:
            gpu_name = None

        for i in range(passes):
            start_ts = datetime.now(timezone.utc).isoformat()
            start = time.time()
            out = apply(init_params, rng, np_batch)
            jax.block_until_ready(out)
            elapsed_ms = (time.time() - start) * 1000.0
            results.append({
                "engine": "deepmind",
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

