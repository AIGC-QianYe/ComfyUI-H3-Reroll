# H3 Reroll

ComfyUI custom nodes for **MiniMax H3 "Reroll" (gacha-style) workflows** — save the
first-stage (一采 / SelfLift) AV latent to disk and reload it later, so you can re-roll
seeds cheaply and only upscale / second-sample (二采) the take you actually like.

> **Reroll** = re-roll. In an H3 video pipeline the expensive first-stage sampling is
> repeated every time you change the seed. Instead of immediately chaining it into the
> upscaler + second sampler, this node **caches each first-stage latent to disk**, so you
> can browse the cheap previews, pick a winner, and re-run *only* the upscale + second
> sample on that one latent — skipping the first-stage entirely (saves ~100 s per take).

## Nodes

Both nodes live under the `H3 Reroll` category.

### H3 Reroll · Save Latent (`H3RerollSave`)

Passthrough node — inserts anywhere on the main chain, passes the latent through
unchanged while writing it to disk.

| Input | Type | Notes |
|---|---|---|
| `latent` | `LATENT` | The first-stage AV latent (5-D video latent). |
| `folder` | `STRING` | Output sub-folder under ComfyUI `output/`. Default: `H3一采抽卡`. |
| `scene_number` | `INT` | Card number. With auto-number **off**, the same number overwrites itself. |
| `auto_next` | `BOOLEAN` | **Off** (default): fixed number, same-name overwrite (draw → look → redraw loop, disk keeps one copy). **On**: auto-pick the next free number, never overwrites (batch draw + keep all for later). |

- Returns `(latent, saved_path)`.
- `IS_CHANGED` always returns `NaN` so a fresh draw is **always** written (cache never skips it).
- File: `output/<folder>/scene_NNN.latent` (~5.8 MB per take).

### H3 Reroll · Load Latent (`H3RerollLoad`)

Reads a previously saved latent back **in full (no cropping)** and feeds it to the
upscaler / second sampler.

| Input | Type | Notes |
|---|---|---|
| `folder` | `STRING` | Same sub-folder used by Save. |
| `scene_number` | `INT` | Which take to load (the one you picked). |

- Returns `(latent,)`.
- `IS_CHANGED` uses **file mtime + size** as a fingerprint: if the file is unchanged the
  node is skipped (cache hit), but changing the seed/file forces a re-run.
- If the file is missing it raises a friendly error telling you to run Save with that number first.

## Format

Self-contained `safetensors` with `{video, audio}` tensors plus metadata
(`format = h3_av_latent_v1`). **No dependency on any third-party plugin** (not even VRGDG —
its loader crops to the tail N frames, which breaks re-roll reuse; this one stores and
returns the *complete* latent).

Missing audio track is auto-padded with `torch.zeros((B, 32, 2, 1))` to satisfy H3's joint
AV structure.

## Install

Copy / clone this folder into ComfyUI's `custom_nodes`:

```bash
git clone <this-repo> ComfyUI/custom_nodes/H3_Reroll
```

Restart ComfyUI. Requires `torch` and `safetensors` (both already present in a standard
ComfyUI install).

## Typical gacha flow

1. Run the **first-stage (SelfLift) chain** with `Save Latent` on it. Change `seed`
   (and bump `scene_number`, or enable `auto_next`) each run → cheap previews + latents on disk.
2. Watch the previews, pick a winner (e.g. `scene_003`).
3. Enable the **upscale + second-sample chain**, set `Load Latent`'s `scene_number = 3`, run.
   The first stage is **not** re-run — it loads straight from disk.

## License

MIT — see [LICENSE](./LICENSE).
