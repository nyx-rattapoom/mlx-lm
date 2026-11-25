# About this branch

`internal-use` is [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) `main` plus three
small patches. It is what [`nyx-rattapoom/exo`](https://github.com/nyx-rattapoom/exo) `internal-use`
pins, on a two-node Apple M4 (24 GB) cluster.

| | |
|---|---|
| Upstream merged | `ml-explore/mlx-lm` `main` @ `6b572ab1b531dd8fd2d13f6f36a89c2eccda18ea` (2026-10-30), merged 2026-11-26 (`025b742`): the rev upstream exo pins since exo#2509 |
| Diverges in | `mlx_lm/models/base.py`, `mlx_lm/models/gated_delta.py`, `mlx_lm/models/cache.py` |
| Needs | `mlx >= 0.32.3` (exo runs the PyPI 0.32.4 wheel) |
| Previous lineage | `backup/internal-use-60f738d9` (branch): `rltakashige/mlx-lm@leo/deepseek-v4` + upstream `1f9883c9`, which exo pinned until 2026-11-26 |

Until 2026-11-26 this branch was a fork of `rltakashige/mlx-lm@leo/deepseek-v4`, because exo
imported `mlx_lm.models.deepseek_v4` and that module only existed there. Upstream exo now ships
DeepSeek V4 itself (`exo/worker/engines/mlx/models/deepseek_v4/`) and pins stock mlx-lm, and its
registry test requires that mlx-lm has **no** `deepseek_v4` model type. So the base-branch
DeepSeek V4 code (model, chat template, convert/tokenizer hooks, e8m0 loader, deepseek_v32
changes) was dropped in `90226c1`. The base branch's cache fixes stay.

## What this branch diverges from upstream on

### force_fused SDPA for head_dim 192/256 (`models/base.py`)

For `head_dim` **192 / 256** prefill (`q_len > 8`), stock mlx's `ScaledDotProductAttention::use_fallback`
takes the **unfused** path on GPUs without NAX (M4 included). That path materialises the full
`[heads, q_len, kv_len]` bf16 score matrix: for a 16-head model with a 2048-token prefill chunk,
**2 GiB at 32k and 8 GiB at 128k**. Measured on 2× M4: unfused was +9.5 % on attention time at 32k
and **OOMed 128k 3/3**; `force_fused=True` fits 128k and ties on decode. Still needed on mlx 0.32.4:
on non-NAX GPUs it adds a fused case for head_dim 256 only when `q_len >= 2048` **and** `q_len == kv_len`,
i.e. the first chunk of a prefill. Every later chunk (`q_len < kv_len`) still takes the unfused path.

`scaled_dot_product_attention` passes `force_fused=True` on prefill calls with those head dims when the
running mlx exposes the kwarg (detected once at import from the nanobind docstring). Decode is left to mlx.

- Kill switch: `MLX_LM_FORCE_FUSED_SDPA=0` (read at import, so launch-time only).
- Fingerprint: every process prints
  `SDPA config: {'force_fused_supported': ..., 'force_fused_enabled': ..., 'head_dims': (192, 256)}`.
- Drop this block if upstream mlx ever changes the 192/256 routing for non-NAX GPUs.

### f32 gate and beta in the GDN kernel path (`models/gated_delta.py`)

Upstream computes the gating terms in the activation dtype (bf16 for Qwen3.5/3.6):
`beta = mx.sigmoid(b)` and `nn.softplus(a + dt_bias)`. Here their inputs are cast to f32 first
(`compute_g`, `compute_lower_bound_g`'s `dt_bias`, and `gated_delta_update`'s `beta`, which then
doubles for `allow_neg_eigval`). The delta rule carries an f32 recurrent state, so a bf16 `beta`
injects error at every step and compounds it: on the 2-node M4 A/B against the older fused kernel,
bf16 `beta` was 100 % of the accuracy gap (~37,000× worse state error); f32 closed it.
[adurham/mlx-lm](https://github.com/adurham/mlx-lm) made the same fix independently. No kernel change
is needed: `g` and `beta` are untemplated float buffers in the Metal kernels. Expect a throughput tie.

### ArraysCache / BatchKVCache fixes from the base branch (`models/cache.py`)

- `ArraysCache.extract` and `merge` keep a slot that is `None` in every batch item as `None` instead of
  failing on it.
- `BatchKVCache.update_and_fetch` makes `keys` depend on `offset` and `left_padding`
  (`mx.depends`), so they are evaluated each step instead of building an ever-longer lazy graph (the
  "ArraysCache leak" the base branch reproduced).

Both matter only to batched generation (exo's `BatchGenerator`); exo's MTP path runs batch size 1.

## Taken from upstream on the 2026-11-26 merge (no longer diverging)

- `models/qwen3_5.py`: upstream's `normalize_qk` (`mx.fast.rms_norm`) replaces the manual `mx.rsqrt`
  q/k normalisation this branch kept to stay byte-identical with the build served in August. exo's MTP
  verify pass uses the same `normalize_qk`.
- `generate.py`: upstream's file. The base branch kept a step's logprobs as one array; exo now does
  that itself by patching `GenerationBatch._step`/`extend` (`exo/worker/engines/mlx/patches/opt_batch_gen.py`).

## The branch name is load-bearing

exo pins this repo **by branch name** (`mlx-lm = { git = "...", branch = "internal-use" }`), so renaming or
deleting `internal-use` breaks every from-scratch exo build, and anything pushed here is what exo's next
`uv lock` picks up (an existing `uv.lock` stays on its recorded SHA). Do not use it as a scratch branch.

Tags: the package version comes from `git describe` (setuptools-scm). A tag reachable from this branch
whose name does not parse as a version (e.g. `backup/...`) breaks every build of it, so keep backups as
**branches**. Upstream's release tags `v0.32.0`–`v0.32.2` are mirrored here so the version reads `0.32.3.devN`.

## Installing

```sh
uv add "mlx-lm @ git+https://github.com/nyx-rattapoom/mlx-lm@internal-use"
```

Not published to PyPI. Everything in the [upstream README](./README.md) otherwise applies.
