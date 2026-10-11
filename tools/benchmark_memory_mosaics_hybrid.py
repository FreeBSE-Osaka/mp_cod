#!/usr/bin/env python3.11
"""Isolated CPU optimizer smoke for research blocks, never an LLM quality test."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cod_model as cod
from tools.general_body_training import guarded_job, sha


def record_failure(out, variant, stage, details, completed_rows):
    """Keep completed evidence before numerical failure; do not sanitize gradients."""
    cod.write_json(out, {'schema_version': 1, 'artifact_type': 'architecture_optimizer_smoke',
        'status': 'failed', 'failed_variant': variant, 'failed_stage': stage,
        'details': details, 'results': completed_rows, 'all_variants_completed': False,
        'script_sha256': sha(__file__), 'block_sha256': sha(Path(__file__).with_name('memory_mosaics_hybrid.py')),
        'promotion_allowed': False, 'default_replacement_allowed': False,
        'normal_cod_runtime_changed': False, 'whole_advisor_goal_complete': False})


def probe(args):
    if args.out.exists():
        raise ValueError("output already exists; keep original evaluations")
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    mx.set_default_device(mx.cpu)
    mx.disable_compile()
    from tools.memory_mosaics_hybrid import HybridBlock, VARIANTS
    mx.random.seed(240506394)
    x = mx.random.normal((2, 10, 16)) * 0.2
    target = mx.concatenate([mx.zeros((2, 1, 16)), x[:, :-1]], axis=1)
    rows = []
    for variant in VARIANTS:
        mx.random.seed(240506394)
        started = time.perf_counter()
        layer = HybridBlock(variant)
        before = layer(x)
        suffix = layer(x.at[:, 5:].add(3))
        prefix = layer(x[:, :5])
        other_actor = layer(x.at[1].add(3))
        mx.eval(before, suffix, prefix, other_actor)
        future_error = float(mx.max(mx.abs(before[:, :5]-suffix[:, :5])).item())
        prefix_error = float(mx.max(mx.abs(before[:, :5]-prefix)).item())
        actor_error = float(mx.max(mx.abs(before[0]-other_actor[0])).item())
        assert max(future_error, prefix_error, actor_error) <= 1e-5
        initial = {key: np.array(value) for key, value in tree_flatten(layer.trainable_parameters())}
        optimizer = optim.AdamW(learning_rate=1e-4)
        loss_and_grad = nn.value_and_grad(layer, lambda model: mx.mean((model(x)-target)**2))
        losses = []
        for _ in range(args.steps):
            loss, grads = loss_and_grad(layer)
            mx.eval(loss, grads)
            if not bool(mx.isfinite(loss).item()):
                record_failure(args.out, variant, 'loss', ['nonfinite loss'], rows)
                raise FloatingPointError(f'{variant}: nonfinite loss')
            nonfinite = [name for name, value in tree_flatten(grads) if not bool(mx.all(mx.isfinite(value)).item())]
            if nonfinite:
                record_failure(args.out, variant, 'gradients', nonfinite, rows)
                raise FloatingPointError(f'{variant}: nonfinite gradient arrays {nonfinite}')
            optimizer.update(layer, grads)
            mx.eval(layer.parameters(), optimizer.state)
            bad_parameters = [name for name, value in tree_flatten(layer.parameters()) if not bool(mx.all(mx.isfinite(value)).item())]
            if bad_parameters:
                record_failure(args.out, variant, 'updated parameters', bad_parameters, rows)
                raise FloatingPointError(f'{variant}: nonfinite updated parameters {bad_parameters}')
            losses.append(float(loss.item()))
        changed = sum(not np.array_equal(initial[key], np.array(value))
                      for key, value in tree_flatten(layer.trainable_parameters()))
        assert changed > 0
        row = {"variant": variant, "parameter_count": sum(value.size for _, value in tree_flatten(layer.trainable_parameters())),
               "gradient_steps": args.steps, "changed_parameter_arrays": changed,
               "loss_before_each_update": losses, "future_suffix_max_error": future_error,
               "prefix_max_error": prefix_error, "cross_actor_max_error": actor_error,
               "seconds": time.perf_counter()-started, "status": "cpu_smoke_pass"}
        rows.append(row)
        print(variant, row["status"], "changed", changed, flush=True)
    package = Path(__import__('mlx_lm').__file__).parent
    report = {"schema_version": 1, "artifact_type": "architecture_optimizer_smoke",
              "scope": "random tiny blocks; copying previous hidden vectors, not token generation or ICL evaluation",
              "device": str(mx.default_device()), "input_shape": list(x.shape), "results": rows,
              "script_sha256": sha(__file__), "block_sha256": sha(Path(__file__).with_name('memory_mosaics_hybrid.py')),
              "native_provider_sha256": {name: sha(package/'models'/name) for name in ('mamba.py','rwkv7.py')},
              "parameter_matched_comparison": False, "language_quality_evaluated": False,
              "long_context_speed_evaluated": False, "full_memory_mosaics_v2_replica": False,
              "normal_cod_runtime_changed": False, "weights_exported": False,
              "fresh_rwkv_key_scale_initialized_to_one": True,
              "all_input_rwkv_gradient_stability_verified": False,
              "promotion_allowed": False, "default_replacement_allowed": False,
              "whole_advisor_goal_complete": False}
    cod.write_json(args.out, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    args.command = 'architecture-probe'
    args.min_free_gib = 20
    args.resource_check_seconds = 2
    return guarded_job(args, probe)


if __name__ == '__main__':
    raise SystemExit(main())
