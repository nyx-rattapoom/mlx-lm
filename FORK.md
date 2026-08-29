# About this branch

`internal-use-upstream-gdn` is a fork of
[`rltakashige/mlx-lm@leo/deepseek-v4`](https://github.com/rltakashige/mlx-lm/tree/leo/deepseek-v4)
with current [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm) `main` merged in.

| | |
|---|---|
| Base | `rltakashige/mlx-lm` `leo/deepseek-v4` @ `6a3df6cd6b00a347ee40f12d97a182aaf86ea599` |
| Upstream merged | `ml-explore/mlx-lm` `main` @ `1f9883c91ab726c6a44fc0249adbfea283ca0b33` |
| GDN kernel | **upstream's** (PR #1559), `gated_delta.py` sha256 `d97103bc…3cf72b` |

The base branch is what upstream [exo](https://github.com/exo-explore/exo) itself pins
(`pyproject.toml`: `mlx-lm = { git = ".../rltakashige/mlx-lm", branch = "leo/deepseek-v4" }`),
so this branch is "exo's own dependency, brought up to date".

## Why this branch exists, and what changed

The previous lineage (`internal-use`) carried a **fused** gated-delta kernel that computed
`g` and `beta` in fp32 inside the Metal kernel. This branch drops that and takes upstream's
**unfused** packed kernel instead.

That was a measured decision, not a preference. On the 2-node M4 cluster the two kernels
**tie** end-to-end at 32k and 128k, upstream costs a constant **+275 KiB**, and the only
behavioural difference is bf16 `beta` precision, which perturbs an occasional reasoning
token without changing the final answer at 32k. Carrying a hand-maintained kernel that
conflicts with every upstream merge was not buying anything measurable.

> The fused lineage is **not lost**. In this repo it is preserved at branch
> **`internal-use-legacy`** (`9494098796e888b4d96d8d0c009b65a236e2f662`) — the same commit that
> `internal-use` still points at, so it is doubly referenced. The matching exo commit is at
> `nyx-rattapoom/exo` `internal-use-fused-gdn-2026-08-28`
> (`d3db334b65e295ae014594bd60d12a78ea4af105`).
>
> ⚠️ The two repos use different names for the same keep-alive: this repo's branch was renamed
> from `internal-use-fused-gdn-2026-08-28` to `internal-use-legacy` on 2026-08-29, while exo's
> kept the dated name. GitHub redirects the old name, but write the new one.

## What this branch still diverges from upstream on

- **DeepSeek V4** — `mlx_lm/models/deepseek_v4.py` and `mlx_lm/chat_templates/deepseek_v4.py`
  exist only here and on the base branch. exo imports `mlx_lm.models.deepseek_v4` at module
  scope in several files, so **a repin to plain upstream `main` breaks every runner at model
  load**. This is the reason the relationship with upstream is a merge, never a repin.
- **DeepSeek V3.2**, `convert.py`, `tokenizer_utils.py` — base-branch changes, carried through.
- **`mlx_lm/models/qwen3_5.py`** — keeps a manual `mx.rsqrt` q/k normalisation where upstream
  uses `mx.fast.rms_norm`. Deliberate: it is byte-identical to the file currently served in
  production, so this branch changes the GDN kernel and nothing else in that model's path.
- **`mlx_lm/generate.py`** — keeps the base branch's `_as_array` logprobs-normalisation helper
  on top of upstream's `stop_matchers` rename. Both are required; see the merge commit.
- **`mlx_lm/models/base.py`** — fused-SDPA routing for head_dim 192/256, see below.

## Runs on stock `mlx` (PyPI, >= 0.32.1) — and forces the fused SDPA kernel

Since 2026-08-30 exo pins **PyPI `mlx==0.32.2`** instead of the
`rltakashige/mlx-jaccl-fix-small-recv` fork, so this branch keeps upstream's
`MIN_MLX_VERSION = "0.32.1"` unchanged (the earlier `0.31.2` override, `f68463d5`, is reverted).

Moving to stock mlx exposed one dispatch difference that matters on 24 GB nodes. For
`head_dim` **192 / 256** prefill (`q_len > 8`) upstream mlx's `ScaledDotProductAttention::use_fallback`
deliberately takes the **unfused** path on GPUs without NAX ("unfused is faster for these shapes"),
which materialises the full `[heads, q_len, kv_len]` bf16 score matrix: for a 16-head model with a
2048-token prefill chunk that is **2 GiB at 32k context and 8 GiB at 128k**. The fork mlx ran those
head dims through its fused steel kernel and never allocated it. Measured on 2x M4: stock-unfused
was +9.5 % on attention time at 32k but OOMed 128k 3/3; stock with `force_fused=True` fits 128k with
a peak **byte-identical** to the fork build and ties it on decode.

So `mlx_lm/models/base.py:scaled_dot_product_attention` passes `force_fused=True` on prefill calls
with `head_dim in (192, 256)` **when the running mlx exposes that kwarg** (detected once at import from
the nanobind docstring; older/fork mlx builds see a silent no-op). Decode (`q_len <= 8`) is left to
mlx. `force_fused` raises rather than silently falling back if no fused kernel exists for a shape.

- Kill switch: `MLX_LM_FORCE_FUSED_SDPA=0` (read at import — launch-time only, like `MLX_GDN_PACKED`).
- Fingerprint: every process prints one line
  `SDPA config: {'force_fused_supported': ..., 'force_fused_enabled': ..., 'head_dims': (192, 256)}`;
  `mlx_lm.models.base.sdpa_config()` returns the same dict.
- Upstream tracking: this is a fork-only routing decision. If upstream ever changes the 192/256
  routing (or exposes a global switch), drop this block and follow upstream.

## Installing

```sh
uv add "mlx-lm @ git+https://github.com/nyx-rattapoom/mlx-lm@internal-use-upstream-gdn"
```

Not published to PyPI. Everything in the [upstream README](./README.md) otherwise applies.
