"""Frozen Kev graph traversal, REINFORCE and windowed GRPO on WebQSP.

    python -m kev.webqsp evaluate --help
    python -m kev.webqsp train --help

See docs/webqsp.md for data downloads, the protocol and single-A100 commands.
"""
import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import os

os.environ.setdefault("HF_HOME", "/nfs/turbo/umms-vgvinodv/users/zzhaozhe/cache")
os.environ.setdefault("XET_LOG_DIR", "/nfs/turbo/umms-vgvinodv/users/zzhaozhe/cache/xet/logs")

import torch

from .device import select
from .model import SERVE_MAX_BRANCH_8K, SERVE_MAX_BRANCH
from .predictors import kernel_environment
from .suite import digest, write_json, write_jsonl
from .webqsp_data import Dataset, graph_diagnostics, score
from .webqsp_grpo import WindowSelector, grpo, window_decision
from .webqsp_policy import CandidateSampler, Policy, beam_search, decision, reinforce, rollout


def candidate_selection(mode, candidate_edges, seed, nominees_per_window):
    if mode == "windowed":
        return WindowSelector(candidate_edges, nominees_per_window), window_decision
    if mode == "sampled":
        return CandidateSampler(candidate_edges, seed), decision
    raise ValueError(f"unknown candidate mode: {mode}")


def resolve_candidates(args, policy):
    """Evaluation inherits saved selection settings unless explicitly overridden."""
    saved = policy.checkpoint.meta.extra.get("webqsp", {}).get("args", {}) if args.command == "evaluate" else {}
    defaults = {"candidate_mode": "windowed" if getattr(args, "algorithm", None) == "grpo" else "sampled",
                "candidate_edges": 128, "nominees_per_window": 2}
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, saved.get(key, default))
    if args.candidate_mode == "windowed":
        if not 1 <= args.nominees_per_window < args.candidate_edges:
            raise ValueError("windowed selection requires 1 <= nominees-per-window < candidate-edges")
        if args.command == "train" and args.algorithm != "grpo":
            raise ValueError("windowed training requires --algorithm grpo")


def evaluate(policy, dataset, hops, beam_width, out, candidate_edges=128, seed=42,
             candidate_mode="sampled", nominees_per_window=2):
    """Keep every eligible question in the denominator, even without a graph/path."""
    out.mkdir(parents=True, exist_ok=False)
    rows, predictions = [], []
    sampler, prepare = candidate_selection(candidate_mode, candidate_edges, seed, nominees_per_window)
    for index, example in enumerate(dataset.examples):
        graph = dataset.graph(example)
        before = sampler.counts()
        paths, failures = beam_search(policy, example.query, graph, dataset.names, hops, beam_width, sampler, prepare=prepare)
        endpoints = list(dict.fromkeys(path.endpoint for path in paths))
        gold = example.answers
        rows.append({"id": example.query.id, **score(example, endpoints),
                     "hits_at_1": float(bool(endpoints) and endpoints[0] in gold),
                     "hits_at_k": float(bool(set(endpoints) & gold)),
                     "predicted": endpoints, "failures": failures, "sampling": sampler.counts(before),
                     "paths": [{"endpoint": p.endpoint, "log_probability": p.log_probability,
                                "moves": [asdict(a) for a in p.path]} for p in paths],
                     **graph_diagnostics(example, graph, dataset.names, hops)})
        predictions.append({"QuestionId": example.query.id, "Answers": endpoints})
        if (index + 1) % 25 == 0:
            print(f"{dataset.split}: {index + 1}/{len(dataset.examples)} questions", flush=True)
    n = len(rows)
    metrics = ("precision", "recall", "f1", "hits_at_1", "hits_at_k", "reachable_answer_recall")
    report = {"questions": n, **{key: sum(r[key] for r in rows) / n for key in metrics},
              "exact_match": sum(r["f1"] == 1 for r in rows) / n,
              "graph_answer_coverage": sum(r["reachable_answer"] for r in rows) / n,
              "missing_graphs": sum(r["missing_graph"] for r in rows),
              "missing_starts": sum(r["missing_start"] for r in rows),
              "questions_with_context_overflow": sum(any(f.startswith("context_overflow:") for f in r["failures"]) for r in rows),
              "empty_predictions": sum(not r["predicted"] for r in rows),
              "entity_occurrences": sum(r["entity_ids"] for r in rows),
              "named_entity_occurrences": sum(r["named_entity_ids"] for r in rows),
              "hops": hops, "beam_width": beam_width, "inverse_edges": dataset.inverse,
              "row_tokens": policy.row_tokens, "candidate_sampling": sampler.provenance(), "sampling": sampler.counts(),
              "path_probability": "final moves conditional on deterministic nominations" if candidate_mode == "windowed"
                                  else "moves conditional on sampled candidate sets",
              "policy": policy.provenance(), "dataset": dataset.provenance,
              "environment": kernel_environment(policy.model, policy.device)}
    write_jsonl(out / "rows.jsonl", rows)
    write_json(out / "predictions.json", predictions)  # Microsoft's eval.py input format.
    write_json(out / "report.json", report)
    return report


def train(policy, dataset, development, args, out, provenance):
    optimizer = torch.optim.AdamW(policy.model.trainable_parameters(), lr=args.lr, weight_decay=0.0)
    use_grpo = args.algorithm == "grpo"
    use_value = not use_grpo and args.baseline == "value"
    value_optimizer = (torch.optim.AdamW(policy.value_head.parameters(), lr=args.value_lr, weight_decay=0.0)
                       if use_value else None)
    rng = random.Random(args.seed)
    sampler, prepare = candidate_selection(args.candidate_mode, args.candidate_edges, args.seed, args.nominees_per_window)
    baseline, updates, best_f1, best_checkpoint = 0.0, 0, -1.0, None
    groups, zero_variance_groups = 0, 0
    with (out / "train.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        for epoch in range(1, args.epochs + 1):
            examples = list(dataset.examples)
            rng.shuffle(examples)
            for example in examples:
                started = time.perf_counter()
                graph = dataset.graph(example)
                before = sampler.counts()
                trajectories = [rollout(policy, example.query, graph, dataset.names, args.hops, sampler,
                                        with_value=use_value, prepare=prepare)
                                for _ in range(args.rollouts)]
                rollout_seconds = time.perf_counter() - started
                if use_grpo:
                    iterations = min(args.grpo_iterations, args.max_updates - updates) if args.max_updates else args.grpo_iterations
                    stats = grpo(policy, trajectories, example.answers, optimizer, args.clip_epsilon, iterations, args.entropy)
                else:
                    stats = reinforce(policy, trajectories, example.answers, optimizer, baseline, args.entropy, value_optimizer)
                updates += stats.get("optimizer_steps", int(stats["optimizer_step"]))
                groups += 1
                zero_variance_groups += int(stats.get("zero_variance_group", False))
                row = {"epoch": epoch, "id": example.query.id, "update": updates, "group": groups,
                       "algorithm": args.algorithm, "reward_ema": baseline, **stats,
                       "baseline": stats.get("baseline_mean") if use_value else (stats["reward"] if use_grpo else baseline),
                       "zero_variance_group_fraction": zero_variance_groups / groups if use_grpo else None,
                       "rollout_seconds": rollout_seconds, "group_seconds": time.perf_counter() - started,
                       "sampling": sampler.counts(before),
                       "failures": [t.failure for t in trajectories if t.failure]}
                # The EMA uses only PREVIOUS questions for gradients. In value
                # mode it is logged as a reference, not used as the actor baseline.
                baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * stats["reward"]
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                print(f"epoch {epoch} {example.query.id}: update={updates} reward={stats['reward']:.3f} "
                      f"failed={stats['failed_rollouts']}/{args.rollouts}", flush=True)
                if (args.max_updates and updates >= args.max_updates) or (args.max_groups and groups >= args.max_groups):
                    break
            if not updates and not use_grpo:
                raise RuntimeError("no policy updates: inspect train.jsonl for missing graphs or context overflow")
            if not updates:
                print("no GRPO updates yet: inspect reward_std, zero_variance_group_fraction and failures in train.jsonl", flush=True)
            checkpoint = out / f"epoch-{epoch}" / "checkpoint"
            policy.save(checkpoint, {**provenance, "epoch": epoch, "updates": updates,
                                     "reward_ema": baseline, "baseline_mode": args.baseline,
                                     "objective": "terminal-membership GRPO" if use_grpo else "terminal-membership REINFORCE",
                                     "groups": groups, "zero_variance_groups": zero_variance_groups,
                                     "candidate_selection": sampler.provenance()})
            policy.model.eval()
            report = evaluate(policy, development, args.hops, args.beam_width, out / f"epoch-{epoch}" / "dev",
                              args.candidate_edges, args.seed, args.candidate_mode, args.nominees_per_window)
            policy.model.train()
            if report["f1"] > best_f1:
                best_f1, best_checkpoint = report["f1"], checkpoint
                write_json(out / "best.json", {"checkpoint": str(checkpoint.resolve()), "epoch": epoch,
                                              "dev_f1": best_f1, "updates": updates}, atomic=True)
            print(f"epoch {epoch}: dev F1={report['f1']:.4f}; best checkpoint={best_checkpoint}", flush=True)
            if (args.max_updates and updates >= args.max_updates) or (args.max_groups and groups >= args.max_groups):
                break
    write_json(out / "result.json", {"algorithm": args.algorithm, "updates": updates, "groups": groups,
                                     "zero_variance_groups": zero_variance_groups if use_grpo else None, "best_dev_f1": best_f1,
                                     "checkpoint": str(best_checkpoint.resolve())})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "train"):
        command = commands.add_parser(name)
        command.add_argument("--questions", required=True, help="Microsoft WebQSP.train.json or WebQSP.test.json")
        command.add_argument("--nsm-dir", required=True, help="NSM webqsp directory with all three splits and vocabularies")
        command.add_argument("--names", required=True, help="independent MID-to-name JSON, e.g. GNN-RAG entities_names.json")
        command.add_argument("--checkpoint", default="jaredpalmer/kev-4b@v1.0")
        command.add_argument("--out", required=True, help="new output directory; existing directories are refused")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--precision", choices=("bf16", "fp32"), default="bf16", help="forward precision; master weights stay fp32")
        command.add_argument("--hops", type=int, default=3, help="raw graph edges, including CVT edges and NO_OPs")
        command.add_argument("--beam-width", type=int, default=8, help="evaluation beam size; 1 is greedy")
        command.add_argument("--inverse-edges", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--row-tokens", type=int, default=SERVE_MAX_BRANCH_8K, help="strict state+options token limit")
        command.add_argument("--candidate-mode", choices=("sampled", "windowed"),
                             help="default: windowed for GRPO, sampled for REINFORCE; evaluation inherits checkpoint settings")
        command.add_argument("--candidate-edges", type=int,
                             help="edge cap per sample/window, plus NO_OP (default 128); sampled mode only: 0 means all actions")
        command.add_argument("--nominees-per-window", type=int, help="distinct nominees per window (default 2)")
        command.add_argument("--temperature", type=float, help="fixed policy temperature; default: checkpoint value")
        command.add_argument("--seed", type=int, default=42)
        if name == "evaluate":
            command.add_argument("--split", choices=("train", "dev", "test"), default="test")
        else:
            command.set_defaults(split="train")
            command.add_argument("--algorithm", choices=("reinforce", "grpo"), default="reinforce")
            command.add_argument("--epochs", type=int, default=3)
            command.add_argument("--rollouts", type=int, default=4, help="sampled trajectories per question/update")
            command.add_argument("--lr", type=float, default=5e-6)
            command.add_argument("--entropy", type=float, help="entropy weight (default: REINFORCE 0.01, GRPO 0)")
            command.add_argument("--baseline", choices=("ema", "value"),
                                 help="value: direct V(s,C) from detached <decide> features; ema: version-1 moving average")
            command.add_argument("--value-lr", type=float, default=1e-4, help="separate value-head learning rate; used only with --baseline value")
            command.add_argument("--baseline-decay", type=float, default=0.9)
            command.add_argument("--max-updates", type=int, default=0, help="0: all epochs; otherwise save/dev-evaluate at this limit")
            command.add_argument("--max-groups", type=int, default=0,
                                 help="cap question groups, including zero-variance groups (0: all epochs)")
            command.add_argument("--clip-epsilon", type=float, default=0.2, help="GRPO probability-ratio clip range")
            command.add_argument("--grpo-iterations", type=int, default=1, help="updates per collected GRPO group (default 1)")
    args = parser.parse_args(argv)
    if args.hops < 1 or args.beam_width < 1 or not 1 <= args.row_tokens <= SERVE_MAX_BRANCH:
        parser.error("hops/beam-width must be positive; row-tokens must be within the Kev row context")
    if args.candidate_edges is not None and args.candidate_edges < 0:
        parser.error("candidate-edges must be nonnegative")
    if args.nominees_per_window is not None and args.nominees_per_window < 1:
        parser.error("nominees-per-window must be positive")
    if args.precision == "bf16" and args.device != "cuda":
        parser.error("bf16 autocast requires CUDA; use --precision fp32 on CPU")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
        parser.error("temperature must be finite and positive")
    if args.command == "train":
        if args.algorithm == "grpo" and (args.rollouts < 2 or args.baseline is not None):
            parser.error("GRPO requires rollouts >= 2 and uses a group baseline; omit --baseline")
        args.baseline = "group" if args.algorithm == "grpo" else (args.baseline or "value")
        if args.entropy is None:
            args.entropy = 0.0 if args.algorithm == "grpo" else 0.01
        if args.epochs < 1 or args.rollouts < 1 or args.max_updates < 0 or args.max_groups < 0:
            parser.error("epochs/rollouts must be positive and max-updates/max-groups nonnegative")
        if not 0 < args.lr <= 2e-5 or not 0 <= args.baseline_decay < 1 or not math.isfinite(args.entropy) or args.entropy < 0:
            parser.error("require 0 < lr <= 2e-5, 0 <= baseline-decay < 1, finite entropy >= 0")
        if not math.isfinite(args.value_lr) or args.value_lr <= 0:
            parser.error("value-lr must be finite and positive")
        if not 0 < args.clip_epsilon < 1 or args.grpo_iterations < 1:
            parser.error("require 0 < clip-epsilon < 1 and grpo-iterations >= 1")
        # Training defaults do not depend on checkpoint metadata. Reject invalid
        # selection settings before loading data or allocating model weights.
        try:
            resolve_candidates(args, None)
        except ValueError as error:
            parser.error(str(error))
    if Path(args.out).exists():
        parser.error("refusing to overwrite an existing output directory")
    return args


def main():
    args = parse_args()
    dataset = Dataset(args.questions, args.nsm_dir, args.names, args.split, args.inverse_edges)
    development = Dataset(args.questions, args.nsm_dir, args.names, "dev", args.inverse_edges) if args.command == "train" else None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select(args.device, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)
    policy = Policy(args.checkpoint, device, args.precision, args.row_tokens, args.temperature,
                    training=args.command == "train", use_value_head=args.command == "train" and args.baseline == "value")
    resolve_candidates(args, policy)
    code = [Path(__file__).with_name(name) for name in
            ("webqsp.py", "webqsp_data.py", "webqsp_policy.py", "webqsp_grpo.py", "model.py", "checkpoint.py", "api.py", "predictors.py")]
    provenance = {"args": vars(args), "dataset": dataset.provenance,
                  "initial_policy": policy.provenance(), "environment": kernel_environment(policy.model, device),
                  "code_sha256": {p.name: digest(p) for p in code}}
    out = Path(args.out)
    if args.command == "evaluate":
        report = evaluate(policy, dataset, args.hops, args.beam_width, out, args.candidate_edges, args.seed,
                          args.candidate_mode, args.nominees_per_window)
        write_json(out / "provenance.json", provenance)
        print(json.dumps({k: report[k] for k in ("questions", "f1", "hits_at_1", "hits_at_k", "empty_predictions")}, indent=2))
    else:
        out.mkdir(parents=True, exist_ok=False)
        provenance["development"] = development.provenance
        write_json(out / "provenance.json", provenance)
        train(policy, dataset, development, args, out, provenance)


if __name__ == "__main__":
    main()
