"""Frozen Kev graph traversal and MINERVA-style REINFORCE on WebQSP.

    python -m kev.webqsp evaluate --help
    python -m kev.webqsp train --help

See docs/webqsp.md for data downloads, the protocol and single-A100 commands.
"""
import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .device import select
from .model import SERVE_MAX_BRANCH_8K, SERVE_MAX_BRANCH
from .predictors import kernel_environment
from .suite import digest, write_json, write_jsonl
from .webqsp_data import Dataset, graph_diagnostics, score
from .webqsp_policy import Policy, beam_search, reinforce, rollout


def evaluate(policy, dataset, hops, beam_width, out):
    """Keep every eligible question in the denominator, even without a graph/path."""
    out.mkdir(parents=True, exist_ok=False)
    rows, predictions = [], []
    for index, example in enumerate(dataset.examples):
        graph = dataset.graph(example)
        paths, failures = beam_search(policy, example.query, graph, dataset.names, hops, beam_width)
        endpoints = list(dict.fromkeys(path.endpoint for path in paths))
        gold = example.answers
        rows.append({"id": example.query.id, **score(example, endpoints),
                     "hits_at_1": float(bool(endpoints) and endpoints[0] in gold),
                     "hits_at_k": float(bool(set(endpoints) & gold)),
                     "predicted": endpoints, "failures": failures,
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
              "row_tokens": policy.row_tokens, "policy": policy.provenance(), "dataset": dataset.provenance,
              "environment": kernel_environment(policy.model, policy.device)}
    write_jsonl(out / "rows.jsonl", rows)
    write_json(out / "predictions.json", predictions)  # Microsoft's eval.py input format.
    write_json(out / "report.json", report)
    return report


def train(policy, dataset, development, args, out, provenance):
    optimizer = torch.optim.AdamW(policy.model.trainable_parameters(), lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    baseline, updates, best_f1, best_checkpoint = 0.0, 0, -1.0, None
    with (out / "train.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        for epoch in range(1, args.epochs + 1):
            examples = list(dataset.examples)
            rng.shuffle(examples)
            for example in examples:
                graph = dataset.graph(example)
                trajectories = [rollout(policy, example.query, graph, dataset.names, args.hops)
                                for _ in range(args.rollouts)]
                stats = reinforce(policy, trajectories, example.answers, optimizer, baseline, args.entropy)
                updates += int(stats["optimizer_step"])
                row = {"epoch": epoch, "id": example.query.id, "update": updates, "baseline": baseline, **stats,
                       "failures": [t.failure for t in trajectories if t.failure]}
                # Baseline uses only PREVIOUS questions at the point a gradient is
                # computed. Including this group's rewards would bias the estimate.
                baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * stats["reward"]
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                print(f"epoch {epoch} {example.query.id}: update={updates} reward={stats['reward']:.3f} "
                      f"failed={stats['failed_rollouts']}/{args.rollouts}", flush=True)
                if args.max_updates and updates >= args.max_updates:
                    break
            if not updates:
                raise RuntimeError("no policy updates: inspect train.jsonl for missing graphs or context overflow")
            checkpoint = out / f"epoch-{epoch}" / "checkpoint"
            policy.save(checkpoint, {**provenance, "epoch": epoch, "updates": updates,
                                     "baseline": baseline, "objective": "terminal-membership REINFORCE"})
            policy.model.eval()
            report = evaluate(policy, development, args.hops, args.beam_width, out / f"epoch-{epoch}" / "dev")
            policy.model.train()
            if report["f1"] > best_f1:
                best_f1, best_checkpoint = report["f1"], checkpoint
                write_json(out / "best.json", {"checkpoint": str(checkpoint.resolve()), "epoch": epoch,
                                              "dev_f1": best_f1, "updates": updates}, atomic=True)
            print(f"epoch {epoch}: dev F1={report['f1']:.4f}; best checkpoint={best_checkpoint}", flush=True)
            if args.max_updates and updates >= args.max_updates:
                break
    write_json(out / "result.json", {"updates": updates, "best_dev_f1": best_f1,
                                     "checkpoint": str(best_checkpoint.resolve())})


def parse_args():
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
        command.add_argument("--temperature", type=float, help="fixed policy temperature; default: checkpoint value")
        command.add_argument("--seed", type=int, default=42)
        if name == "evaluate":
            command.add_argument("--split", choices=("train", "dev", "test"), default="test")
        else:
            command.set_defaults(split="train")
            command.add_argument("--epochs", type=int, default=3)
            command.add_argument("--rollouts", type=int, default=4, help="sampled trajectories per question/update")
            command.add_argument("--lr", type=float, default=5e-6)
            command.add_argument("--entropy", type=float, default=0.01)
            command.add_argument("--baseline-decay", type=float, default=0.9)
            command.add_argument("--max-updates", type=int, default=0, help="0: all epochs; otherwise save/dev-evaluate at this limit")
    args = parser.parse_args()
    if args.hops < 1 or args.beam_width < 1 or not 1 <= args.row_tokens <= SERVE_MAX_BRANCH:
        parser.error("hops/beam-width must be positive; row-tokens must be within the Kev row context")
    if args.precision == "bf16" and args.device != "cuda":
        parser.error("bf16 autocast requires CUDA; use --precision fp32 on CPU")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
        parser.error("temperature must be finite and positive")
    if args.command == "train":
        if args.epochs < 1 or args.rollouts < 1 or args.max_updates < 0:
            parser.error("epochs/rollouts must be positive and max-updates nonnegative")
        if not 0 < args.lr <= 2e-5 or not 0 <= args.baseline_decay < 1 or not math.isfinite(args.entropy) or args.entropy < 0:
            parser.error("require 0 < lr <= 2e-5, 0 <= baseline-decay < 1, finite entropy >= 0")
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
    policy = Policy(args.checkpoint, device, args.precision, args.row_tokens, args.temperature, training=args.command == "train")
    code = [Path(__file__).with_name(name) for name in
            ("webqsp.py", "webqsp_data.py", "webqsp_policy.py", "model.py", "checkpoint.py", "api.py", "predictors.py")]
    provenance = {"args": vars(args), "dataset": dataset.provenance,
                  "initial_policy": policy.provenance(), "environment": kernel_environment(policy.model, device),
                  "code_sha256": {p.name: digest(p) for p in code}}
    out = Path(args.out)
    if args.command == "evaluate":
        report = evaluate(policy, dataset, args.hops, args.beam_width, out)
        write_json(out / "provenance.json", provenance)
        print(json.dumps({k: report[k] for k in ("questions", "f1", "hits_at_1", "hits_at_k", "empty_predictions")}, indent=2))
    else:
        out.mkdir(parents=True, exist_ok=False)
        provenance["development"] = development.provenance
        write_json(out / "provenance.json", provenance)
        train(policy, dataset, development, args, out, provenance)


if __name__ == "__main__":
    main()
