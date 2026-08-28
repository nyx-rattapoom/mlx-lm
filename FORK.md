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

> The fused lineage is **not lost**. It is preserved at branch
> `internal-use-fused-gdn-2026-08-28` (`9494098796e888b4d96d8d0c009b65a236e2f662`), and the
> matching exo commit at `nyx-rattapoom/exo` `internal-use-fused-gdn-2026-08-28`
> (`d3db334b65e295ae014594bd60d12a78ea4af105`).

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
- **`setup.py`** — see below.

## `setup.py` pins a lower MLX floor than upstream

Upstream [#1753](https://github.com/ml-explore/mlx-lm/pull/1753) raised `MIN_MLX_VERSION` to
`0.32.1`. **This branch holds it at `0.31.2`**, and that override must be re-applied on every
future upstream merge — `setup.py` does not otherwise conflict, so it will silently take
upstream's value unless someone acts.

The reason is exo's mlx pin, and it is not a matter of taste. Checked 2026-08-28:

- upstream exo `main` pins `mlx==0.32.0` →
  `rltakashige/mlx-jaccl-fix-small-recv@address-rdma-gpu-locks#cc3f3e60`;
- that branch's head (`e9835615`, one commit ahead of the pin) **still reports
  `MLX_VERSION 0.32.0`** in `mlx/version.h`.

So no build of exo's mlx satisfies `>=0.32.1` today, and upstream exo has not moved. Keeping
the floor at `0.31.2` is what lets exo's `uv lock` resolve at all. Revisit when upstream exo
adopts an mlx that reports `0.32.1` or newer — and follow upstream exo's pin rather than
jumping ahead of it.

## Installing

```sh
uv add "mlx-lm @ git+https://github.com/nyx-rattapoom/mlx-lm@internal-use-upstream-gdn"
```

Not published to PyPI. Everything in the [upstream README](./README.md) otherwise applies.
