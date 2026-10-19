"""One graph policy shared by frozen evaluation and on-policy REINFORCE."""
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch

from .checkpoint import Checkpoint, LoadOptions, write_meta
from .model import ContextOverflow
from .suite import digest
from .webqsp_data import Action, observation


class CandidateSampler:
    """Sample edges before encoding; keep a random prefix that fits the actual row budget."""
    def __init__(self, candidate_edges=128, seed=42):
        if candidate_edges < 0:
            raise ValueError("candidate_edges must be nonnegative")
        self.candidate_edges, self.seed = candidate_edges, seed
        self.rng = random.Random(seed)
        self.totals = dict.fromkeys(("prepared_decisions", "legal_actions", "offered_actions", "capped_decisions",
                                   "token_limited_decisions", "noop_only_decisions", "encoded_tokens"), 0)

    def provenance(self):
        return {"method": "uniform_sample_then_fitting_prefix" if self.candidate_edges else "all_actions",
                "candidate_edges": self.candidate_edges, "no_op_always_available": True, "seed": self.seed}

    def counts(self, previous=None):
        return {key: value - (previous or {}).get(key, 0) for key, value in self.totals.items()}

    def select(self, actions, encode):
        """`actions[0]` is NO_OP; `encode` applies the normal strict Kev encoder to a whole list."""
        edges = list(actions[1:])
        capped = bool(self.candidate_edges and len(edges) > self.candidate_edges)
        if capped:
            edges = self.rng.sample(edges, self.candidate_edges)
        selected = (actions[0], *edges)
        encoded, token_limited = None, False
        if len(selected) > 1:
            try:
                encoded = encode(selected)
            except ContextOverflow as error:
                if not self.candidate_edges or error.state_tokens is not None:
                    raise
                # Every tested prefix keeps whole options and their indices. Its
                # token length grows monotonically, so binary search is exact.
                # Shuffle only now if the degree cap did not already sample an order.
                if not capped:
                    self.rng.shuffle(edges)
                encoded = encode((actions[0],))  # Propagate an overlong state/instruction.
                lo, hi = 0, len(edges)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    try:
                        trial = encode((actions[0], *edges[:mid]))
                    except ContextOverflow:
                        hi = mid - 1
                    else:
                        lo, encoded = mid, trial
                selected = (actions[0], *edges[:lo])
                token_limited = len(selected) < len(edges) + 1
        self.totals["prepared_decisions"] += 1
        self.totals["legal_actions"] += len(actions)
        self.totals["offered_actions"] += len(selected)
        self.totals["capped_decisions"] += int(capped)
        self.totals["token_limited_decisions"] += int(token_limited)
        self.totals["noop_only_decisions"] += int(len(selected) == 1)
        self.totals["encoded_tokens"] += len(encoded["ids"]) if encoded is not None else 0
        return selected, encoded if len(selected) > 1 else None


class ValueHead(torch.nn.Module):
    """Direct V(s, C) for binary terminal reward; critic gradients never enter Kev."""
    architecture = "linear_sigmoid_v1"

    def __init__(self, hidden_size):
        super().__init__()
        # A new critic must not change the actor's action-sampling RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.projection = torch.nn.Linear(hidden_size, 1, device="cpu", dtype=torch.float32)
        torch.nn.init.zeros_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)  # Initial prediction is 0.5, not a fitted success rate.

    def forward(self, features):
        return self.projection(features.detach().float()).squeeze(-1).sigmoid()

    def checkpoint_state(self):
        return {"architecture": self.architecture, "hidden_size": self.projection.in_features,
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}

    def load_checkpoint_state(self, state):
        if state["architecture"] != self.architecture or state["hidden_size"] != self.projection.in_features:
            raise ValueError("WebQSP value head is incompatible with this architecture/hidden size")
        self.load_state_dict(state["state_dict"], strict=True)


class Policy:
    def __init__(self, checkpoint, device, precision, row_tokens, temperature=None, training=False, use_value_head=False):
        if use_value_head and not training:
            raise ValueError("the WebQSP value head is only needed for training")
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
        self.value_head, self.value_head_loaded = None, False
        if use_value_head:
            self.value_head = ValueHead(self.model.head.q.in_features).to(device)
            saved_value = meta.extra.get("webqsp_value_head")
            if saved_value is not None:
                self.value_head.load_checkpoint_state(saved_value)
                self.value_head_loaded = True

    def encode(self, record):
        return self.model.encode(self.tok, record, max_state=self.row_tokens, max_branch=self.row_tokens, strict=True)

    def _forward(self, encoded, return_features=False):
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bf16" else nullcontext()
        with autocast:
            return self.model(encoded, return_features=return_features)[0]

    def _distribution(self, logits):
        logits = logits.float() / self.temperature
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite WebQSP action logits")
        return torch.distributions.Categorical(logits=logits)

    def distribution(self, encoded):
        return self._distribution(self._forward(encoded))

    @torch.no_grad()
    def distribution_and_value(self, encoded):
        if self.value_head is None:
            raise ValueError("value-baseline rollout requires an initialized value head")
        logits, features = self._forward(encoded, return_features=True)
        value = float(self.value_head(features))
        if not math.isfinite(value):
            raise FloatingPointError("non-finite WebQSP value prediction")
        # Copy just this vector, not a view retaining the full row's hidden states.
        saved_features = features.detach().to(device="cpu", dtype=torch.float32, copy=True)
        return self._distribution(logits), value, saved_features

    def provenance(self):
        return {"requested": self.checkpoint.requested, "resolved": self.checkpoint.path,
                "weights_sha256": self.checkpoint.weights_sha256(),
                "head_sha256": digest(self.checkpoint.file("head.pt")),
                "base": self.checkpoint.meta.base, "base_revision": self.checkpoint.meta.base_revision,
                "temperature": self.temperature, "precision": self.precision, "master_weights": "fp32",
                "adapter_merged": False, "dropout": 0.0,
                "value_head": {"architecture": self.value_head.architecture,
                               "loaded_from_checkpoint": self.value_head_loaded,
                               "features_detached": True} if self.value_head is not None else None}

    def save(self, directory, provenance):
        directory.mkdir(parents=True, exist_ok=False)
        self.model.lm.save_pretrained(directory)
        self.tok.save_pretrained(directory)
        # Old held-out calibration is no longer evidence about the updated policy.
        # Preserve its numeric temperature as an explicit policy hyperparameter.
        extra = {"webqsp": provenance, "calibration_status": "not calibrated after WebQSP RL"}
        if self.value_head is not None:
            extra["webqsp_value_head"] = self.value_head.checkpoint_state()
        meta = replace(self.checkpoint.meta,
                       head={k: v.detach().cpu() for k, v in self.model.head.state_dict().items()},
                       temperature=self.temperature, holdout=[],
                       extra=extra)
        write_meta(directory, meta)
        # Subsequent dev reports must fingerprint the updated weights, rather
        # than labelling an in-memory trained policy with its parent's hashes.
        self.checkpoint = Checkpoint(directory)


@dataclass
class Decision:
    encoded: dict
    action_index: int
    baseline: float | None = None
    value_features: torch.Tensor | None = None


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


def decision(policy, query, graph, names, trajectory, hops, sampler=None, with_value=False):
    actions = graph.actions(trajectory.endpoint)
    def encode(candidates):
        return policy.encode(observation(query, trajectory.endpoint, trajectory.path, candidates, names, hops))
    if sampler is not None:
        actions, encoded = sampler.select(actions, encode)
    else:
        encoded = encode(actions) if len(actions) > 1 else None
    # A forced NO_OP has log probability and entropy zero.
    if encoded is None:
        return actions, encoded, None, None, None
    if with_value:
        distribution, baseline, features = policy.distribution_and_value(encoded)
        return actions, encoded, distribution, baseline, features
    return actions, encoded, policy.distribution(encoded), None, None


@torch.no_grad()
def rollout(policy, query, graph, names, hops, sampler=None, with_value=False):
    trajectory = initial_trajectory(query, graph)
    if trajectory.failure:
        return trajectory
    for _ in range(hops):
        try:
            actions, encoded, distribution, baseline, features = decision(
                policy, query, graph, names, trajectory, hops, sampler, with_value)
        except ContextOverflow as error:
            # No state/option text truncation or gold-based pruning. Failure is an
            # absorbing terminal state, with reward zero even at a gold node.
            return replace(trajectory, endpoint=None, failure=f"context_overflow: {error}")
        index = int(distribution.sample().item()) if distribution is not None else 0
        chosen = actions[index]
        logp = float(distribution.logits[index].item()) if distribution is not None else 0.0
        decisions = (trajectory.decisions + (Decision(encoded, index, baseline, features),)
                     if encoded is not None else trajectory.decisions)
        trajectory = Trajectory(chosen.target, (*trajectory.path, chosen), trajectory.log_probability + logp, decisions)
    return trajectory


@torch.no_grad()
def beam_search(policy, query, graph, names, hops, width, sampler=None):
    """Top paths by summed log probability; no answer-based reranking or merging."""
    first = initial_trajectory(query, graph)
    if first.failure:
        return [], [first.failure]
    beams, failures = [first], []
    for _ in range(hops):
        expanded = []
        for trajectory in beams:
            try:
                actions, _, distribution, _, _ = decision(policy, query, graph, names, trajectory, hops, sampler)
            except ContextOverflow as error:
                failures.append(f"context_overflow: {error}")
                continue
            logps = distribution.logits.cpu().tolist() if distribution is not None else [0.0]
            # Stable ties follow the offered action order, including NO_OP.
            indices = sorted(range(len(actions)), key=lambda i: -logps[i])[:width]
            for i in indices:
                expanded.append(Trajectory(actions[i].target, (*trajectory.path, actions[i]),
                                           trajectory.log_probability + logps[i]))
        beams = sorted(expanded, key=lambda t: -t.log_probability)[:width]
        if not beams:
            break
    return beams, failures


def reinforce(policy, trajectories, answers, optimizer, baseline, entropy_weight, value_optimizer=None):
    """One on-policy update. Recompute/backprop one step at a time, then step once."""
    rewards = [float(t.endpoint in answers) if not t.failure else 0.0 for t in trajectories]
    use_value = value_optimizer is not None
    if use_value and any(step.baseline is None or step.value_features is None
                         for t in trajectories for step in t.decisions):
        raise ValueError("value-baseline update requires predictions/features saved during rollout")
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_entropy, decisions = 0.0, 0.0, 0
    value_features, targets = [], []
    total_baseline, total_advantage, value_error = 0.0, 0.0, 0.0
    for trajectory, reward in zip(trajectories, rewards):
        for step in trajectory.decisions:
            # Use the prediction made BEFORE this action/reward, never a value
            # already fitted on this group's own outcome.
            step_baseline = step.baseline if use_value else baseline
            advantage = reward - step_baseline
            distribution = policy.distribution(step.encoded)
            entropy = distribution.entropy()
            loss = (-advantage * distribution.logits[step.action_index] - entropy_weight * entropy) / len(trajectories)
            loss.backward()
            total_loss += float(loss.detach())
            total_entropy += float(entropy.detach())
            total_baseline += step_baseline
            total_advantage += advantage
            if use_value:
                value_features.append(step.value_features)
                targets.append(reward)
                value_error += advantage ** 2
            decisions += 1
    if decisions:
        torch.nn.utils.clip_grad_norm_(policy.model.trainable_parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    value_loss = None
    if value_features:
        # Only the small head is trained here. Saved detached vectors avoid a
        # second backbone pass and exclude critic gradients from LoRA/the actor.
        value_optimizer.zero_grad(set_to_none=True)
        device = next(policy.value_head.parameters()).device
        predicted = policy.value_head(torch.stack(value_features).to(device))
        target = torch.tensor(targets, dtype=predicted.dtype, device=device)
        loss = torch.nn.functional.mse_loss(predicted, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.value_head.parameters(), 1.0, error_if_nonfinite=True)
        value_optimizer.step()
        value_loss = float(loss.detach())
    return {"reward": sum(rewards) / len(rewards), "loss": total_loss,
            "entropy_per_decision": total_entropy / decisions if decisions else 0.0,
            "baseline_mode": "value" if use_value else "ema",
            "baseline_mean": total_baseline / decisions if decisions else None,
            "advantage_mean": total_advantage / decisions if decisions else None,
            "value_loss": value_loss, "value_mse_before_update": value_error / decisions if use_value and decisions else None,
            "value_optimizer_step": bool(value_features),
            "decisions": decisions, "optimizer_step": bool(decisions),
            "failed_rollouts": sum(t.failure is not None for t in trajectories)}
