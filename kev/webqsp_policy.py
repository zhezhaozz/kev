"""One graph policy shared by frozen evaluation and on-policy REINFORCE."""
import math
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch

from .checkpoint import Checkpoint, LoadOptions, write_meta
from .model import ContextOverflow
from .suite import digest
from .webqsp_data import Action, observation


class Policy:
    def __init__(self, checkpoint, device, precision, row_tokens, temperature=None, training=False):
        self.checkpoint = Checkpoint(checkpoint)
        meta = self.checkpoint.meta
        if "Qwen3.5" not in meta.base:
            raise ValueError("this WebQSP policy path is for Qwen3.5 Kev checkpoints")
        if self.checkpoint.full or meta.special_embeddings:
            raise ValueError("WebQSP supports LoRA + pointer-head checkpoints without trained token embeddings")
        if meta.weights_dtype != "fp32":
            raise ValueError("WebQSP requires an fp32-backbone checkpoint, such as jaredpalmer/kev-4b@v1.0")
        self.temperature = meta.temperature if temperature is None else temperature
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("policy temperature must be finite and positive")
        self.device, self.precision, self.row_tokens = device, precision, row_tokens
        # Identical unmerged fp32 master weights in both arms; bf16 is autocast,
        # not a cast of the trainable LoRA parameters. The head's implicit eval-only
        # temperature is disabled, then applied exactly once below in both modes.
        self.tok, self.model = self.checkpoint.load(device, LoadOptions(
            dtype=torch.float32, merge=False, temperature=1.0, backend="torch", fused=False, cuda_graphs=False))
        self.model.requires_grad_(False)
        self.model.lm.config.use_cache = False
        if training:
            self.model.lm.set_adapter("default")
            self.model.lm.peft_config["default"].inference_mode = False
            self.model.head.requires_grad_(True)
            unexpected = [name for name, p in self.model.named_parameters()
                          if p.requires_grad and not (name.startswith("head.") or ".lora_" in name)]
            if unexpected:
                raise ValueError(f"refusing to train backbone parameters: {unexpected[:5]}")
            if not any(p.requires_grad for p in self.model.lm.parameters()):
                raise ValueError("checkpoint has no trainable LoRA parameters")
            self.model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train()
        # Keep rollout and gradient recomputation the same policy. In training we
        # retain train mode for gradient checkpointing, but disable dropout in both
        # passes. Qwen3.5 uses Dropout modules for LoRA and a float for attention.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if isinstance(getattr(module, "attention_dropout", None), (int, float)):
                module.attention_dropout = 0.0

    def encode(self, record):
        return self.model.encode(self.tok, record, max_state=self.row_tokens, max_branch=self.row_tokens, strict=True)

    def distribution(self, encoded):
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bf16" else nullcontext()
        with autocast:
            logits = self.model(encoded)[0]
        logits = logits.float() / self.temperature
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite WebQSP action logits")
        return torch.distributions.Categorical(logits=logits)

    def provenance(self):
        return {"requested": self.checkpoint.requested, "resolved": self.checkpoint.path,
                "weights_sha256": self.checkpoint.weights_sha256(),
                "head_sha256": digest(self.checkpoint.file("head.pt")),
                "base": self.checkpoint.meta.base, "base_revision": self.checkpoint.meta.base_revision,
                "temperature": self.temperature, "precision": self.precision, "master_weights": "fp32",
                "adapter_merged": False, "dropout": 0.0}

    def save(self, directory, provenance):
        directory.mkdir(parents=True, exist_ok=False)
        self.model.lm.save_pretrained(directory)
        self.tok.save_pretrained(directory)
        # Old held-out calibration is no longer evidence about the updated policy.
        # Preserve its numeric temperature as an explicit policy hyperparameter.
        meta = replace(self.checkpoint.meta,
                       head={k: v.detach().cpu() for k, v in self.model.head.state_dict().items()},
                       temperature=self.temperature, holdout=[],
                       extra={"webqsp": provenance, "calibration_status": "not calibrated after WebQSP RL"})
        write_meta(directory, meta)
        # Subsequent dev reports must fingerprint the updated weights, rather
        # than labelling an in-memory trained policy with its parent's hashes.
        self.checkpoint = Checkpoint(directory)


@dataclass
class Decision:
    encoded: dict
    action_index: int


@dataclass
class Trajectory:
    endpoint: str | None
    path: tuple[Action, ...] = ()
    log_probability: float = 0.0
    decisions: tuple[Decision, ...] = ()
    failure: str | None = None


def initial_trajectory(query, graph):
    if graph is None:
        return Trajectory(None, failure="missing_graph")
    if query.start not in graph.nodes:
        return Trajectory(None, failure="missing_start")
    return Trajectory(query.start)


def decision(policy, query, graph, names, trajectory, hops):
    actions = graph.actions(trajectory.endpoint)
    if len(actions) == 1:
        return actions, None, None  # A forced NO_OP has log probability and entropy zero.
    record = observation(query, trajectory.endpoint, trajectory.path, actions, names, hops)
    encoded = policy.encode(record)
    return actions, encoded, policy.distribution(encoded)


@torch.no_grad()
def rollout(policy, query, graph, names, hops):
    trajectory = initial_trajectory(query, graph)
    if trajectory.failure:
        return trajectory
    for _ in range(hops):
        try:
            actions, encoded, distribution = decision(policy, query, graph, names, trajectory, hops)
        except ContextOverflow as error:
            # No silent truncation or gold-based action pruning. Failure is an
            # absorbing terminal state, with reward zero even at a gold node.
            return replace(trajectory, endpoint=None, failure=f"context_overflow: {error}")
        index = int(distribution.sample().item()) if distribution is not None else 0
        chosen = actions[index]
        logp = float(distribution.logits[index].item()) if distribution is not None else 0.0
        decisions = trajectory.decisions + (Decision(encoded, index),) if encoded is not None else trajectory.decisions
        trajectory = Trajectory(chosen.target, (*trajectory.path, chosen), trajectory.log_probability + logp, decisions)
    return trajectory


@torch.no_grad()
def beam_search(policy, query, graph, names, hops, width):
    """Top paths by summed log probability; no answer-based reranking or merging."""
    first = initial_trajectory(query, graph)
    if first.failure:
        return [], [first.failure]
    beams, failures = [first], []
    for _ in range(hops):
        expanded = []
        for trajectory in beams:
            try:
                actions, _, distribution = decision(policy, query, graph, names, trajectory, hops)
            except ContextOverflow as error:
                failures.append(f"context_overflow: {error}")
                continue
            logps = distribution.logits.cpu().tolist() if distribution is not None else [0.0]
            # Stable ties follow the deterministic action order, including NO_OP.
            indices = sorted(range(len(actions)), key=lambda i: -logps[i])[:width]
            for i in indices:
                expanded.append(Trajectory(actions[i].target, (*trajectory.path, actions[i]),
                                           trajectory.log_probability + logps[i]))
        beams = sorted(expanded, key=lambda t: -t.log_probability)[:width]
        if not beams:
            break
    return beams, failures


def reinforce(policy, trajectories, answers, optimizer, baseline, entropy_weight):
    """One on-policy update. Recompute/backprop one step at a time, then step once."""
    rewards = [float(t.endpoint in answers) if not t.failure else 0.0 for t in trajectories]
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_entropy, decisions = 0.0, 0.0, 0
    for trajectory, reward in zip(trajectories, rewards):
        for step in trajectory.decisions:
            distribution = policy.distribution(step.encoded)
            entropy = distribution.entropy()
            loss = (-(reward - baseline) * distribution.logits[step.action_index] - entropy_weight * entropy) / len(trajectories)
            loss.backward()
            total_loss += float(loss.detach())
            total_entropy += float(entropy.detach())
            decisions += 1
    if decisions:
        torch.nn.utils.clip_grad_norm_(policy.model.trainable_parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return {"reward": sum(rewards) / len(rewards), "loss": total_loss,
            "entropy_per_decision": total_entropy / decisions if decisions else 0.0,
            "decisions": decisions, "optimizer_step": bool(decisions),
            "failed_rollouts": sum(t.failure is not None for t in trajectories)}
