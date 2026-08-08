# About this fork

This repository is a fork of [`rltakashige/mlx-lm@leo/deepseek-v4`](https://github.com/rltakashige/mlx-lm/tree/leo/deepseek-v4),
which is itself a fork of [`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm).

The default branch is **`internal-use`**. It contains everything on the base branch,
plus current upstream `main`, plus the Apple-silicon inference work described below.

## Why this fork exists

The base branch stopped following upstream — its most recent commit is from
2026-04-26, and it has since diverged from `ml-explore/mlx-lm` `main`.

That left its DeepSeek-V4 work stranded on an increasingly old upstream base.
This fork exists to keep that lineage current: upstream `main` is merged into
`internal-use`, so the branch carries the base branch's changes *and* current
upstream, rather than forcing a choice between them.

Merges are done by hand as needed, not on a schedule.

## What uses it

This fork is the `mlx-lm` dependency pinned by our [exo](https://github.com/exo-explore/exo)
build. exo pins it **by branch name** (`internal-use`), not by revision.

Two consequences, if you are considering depending on this yourself:

- The branch is load-bearing for an internal build, so it is not a scratch branch —
  but it *does* move. **Pin a commit SHA, not the branch.**
- It is not published to PyPI.

## Installing

```sh
pip install "git+https://github.com/nyx-rattapoom/mlx-lm@internal-use"
```

or, with [uv](https://github.com/astral-sh/uv):

```sh
uv add "mlx-lm @ git+https://github.com/nyx-rattapoom/mlx-lm@internal-use"
```

Everything in the [upstream README](./README.md) applies — the CLI entry points,
the Python API, and the model coverage are unchanged.

## Packed gated-delta kernel

`mlx_lm/models/gated_delta.py` carries a packed Metal kernel for gated delta
networks (GDN), ported from upstream PR
[ml-explore/mlx-lm#1559](https://github.com/ml-explore/mlx-lm/pull/1559).

It is a port, not a cherry-pick. This lineage fuses the gate computation into the
kernel, taking `(a, b, A_log, dt_bias)` and deriving `g` on-device, where the PR
takes precomputed `g` and `beta` buffers. The Metal source here is the PR's
kernel adapted to that convention — the packing layout and the explicit butterfly
reduction are carried over unchanged, and only the gate plumbing differs.

Upstream PR #1559 is open and unmerged at the time of writing, so this remains
fork-local.

### `MLX_GDN_PACKED`

The packed kernel is **enabled by default**. To fall back to the unpacked path:

```sh
MLX_GDN_PACKED=0 python your_script.py
```

Only the exact string `0` disables it; any other value leaves it enabled.

> **It is read once, at import time.** Setting `MLX_GDN_PACKED` on an
> already-running process does nothing, and setting it after `mlx_lm` has been
> imported does nothing. It must be in the environment before the import.

Some shapes fall back to the unpacked kernel regardless of this setting — see the
comments at the top of `gated_delta.py` for the current conditions.

## Branches

| Branch | Role |
|---|---|
| `internal-use` | Default branch. The deployed lineage: base branch + upstream `main` + the packed GDN kernel. This is what to use. |
| `leo/deepseek-v4` | Copy of the fork base, [`rltakashige/mlx-lm@leo/deepseek-v4`](https://github.com/rltakashige/mlx-lm/tree/leo/deepseek-v4). Kept so the divergence point stays visible; not updated. |
| `main` | Mirror of upstream `ml-explore/mlx-lm` `main`. The fork point, carrying no local changes. |

## Contributing

Anything generally useful here belongs upstream. Please send it to
[`ml-explore/mlx-lm`](https://github.com/ml-explore/mlx-lm), or to the
[#1559](https://github.com/ml-explore/mlx-lm/pull/1559) discussion for the
GDN kernel specifically.
