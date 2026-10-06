# Backburner iPhone 13 Pro compatibility experiment

These are experimental patches, not an upstream-supported release or a default MP CoD backend. The actual physical-device results and operating limits are in [the experiment report](../../docs/distributed_iphone13_backburner_20261006.md).

Base revisions:

- `https://github.com/StayLameBro/backburner` at `c03893748386a65c6a7312be39659c9d35026de3`
- its `llama.cpp` submodule at `1839b78175d4f89f74023a2ced04f4e558abc4d4`

Apply `backburner.patch` in the Backburner root and `llama-qwen35.patch` in its `llama.cpp` directory. Check both revisions and run `git apply --check` before applying. Do not apply to an unrelated or dirty upstream checkout without reviewing the overlapping changes.

The patches disable unavailable A15 SME2, retain the existing runtime availability checks around newer Metal APIs, use ordinary iOS memory entitlements, permit an APFS iOS build directory, and accept the cached Qwen3.5 GGUF's legacy RoPE, SSM tensor name and per-layer KV-head metadata. The trial additionally launches the phone with `PA_NA_DISABLE=1` and `GGML_METAL_FA_PREFILL_NA=0`. Loader changes were tested with the same patched loader on both workers; equivalence to other engines and general model accuracy remain unverified.

`scripts/text-only-gguf-a15.py` makes a new decoder-only GGUF, omitting vision and MTP tensors without rewriting the retained tensor data. `scripts/split-gguf.py --no-head` generates a small tail from the same source. Original GGUF files and third-party weights are not included here.

Use `BB_A15_LITE=1 BB_INCREMENTAL_IOS=1` when building this experiment. Set `BB_IOS_BUILD_DIR` to an absolute APFS build path. Existing externally specified directories are never recursively deleted by the new clean-build branch. Model files may remain on the external volume.

Both upstream projects use the MIT license. Their notices are retained in [UPSTREAM-LICENSE.txt](UPSTREAM-LICENSE.txt). MP CoD changes do not grant rights to the model weights; use each model's own license.
