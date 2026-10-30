"""Exhaustive window nominations and group-relative updates for the Kev actor.

GRPO groups are complete trajectories of ONE question. Every stochastic
nomination and executed move receives that trajectory's terminal advantage.
"""
import math
from itertools import groupby

import torch

from .model import ContextOverflow
from .webqsp_data import observation
from .webqsp_policy import Decision


def conditional_distribution(distribution, excluded):
    """Reuse one window's logits, masking earlier draws without replacement."""
    if not excluded:
        return distribution
    mask = torch.zeros_like(distribution.logits, dtype=torch.bool)
    mask[list(excluded)] = True
    return torch.distributions.Categorical(logits=distribution.logits.masked_fill(mask, -torch.inf))


class WindowSelector:
    def __init__(self, candidate_edges=128, nominees_per_window=2):
        if not 1 <= nominees_per_window < candidate_edges:
            raise ValueError("windowed selection requires 1 <= nominees_per_window < candidate_edges")
        self.candidate_edges, self.nominees_per_window = candidate_edges, nominees_per_window
        self.totals = dict.fromkeys(("prepared_decisions", "legal_actions", "offered_actions", "windows",
                                    "inspected_edges", "token_limited_windows", "reduction_rounds",
                                    "nomination_forwards", "final_forwards", "encoded_tokens", "noop_only_decisions"), 0)

    def provenance(self):
        return {"method": "exhaustive_windows", "candidate_edges": self.candidate_edges,
                "nominees_per_window": self.nominees_per_window, "order": "graph_order",
                "training_nominations": "sample_without_replacement", "evaluation_nominations": "stable_top_k",
                "no_op_always_available": True, "no_op_stage": "final_only",
                "oversized_shortlist": "repeat_window_selection", "instruction": "shared_with_final_move"}

    def counts(self, previous=None):
        return {key: value - (previous or {}).get(key, 0) for key, value in self.totals.items()}

    def windows(self, edges, noop, encode):
        """Cover every edge once per round; reserve room for the final NO_OP.

        Binary search uses actual strict encodings. Nothing is truncated or
        silently dropped, even an individual option that cannot fit.
        """
        offset = 0
        while offset < len(edges):
            count = min(self.candidate_edges, len(edges) - offset)
            window = edges[offset:offset + count]
            try:
                encode((noop, *window))
            except ContextOverflow as error:
                if error.state_tokens is not None:
                    raise
                # A single whole edge plus NO_OP must be representable downstream.
                encode((noop, window[0]))
                lo, hi = 1, count - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    try:
                        encode((noop, *window[:mid]))
                    except ContextOverflow:
                        hi = mid - 1
                    else:
                        lo = mid
                window = window[:lo]
                self.totals["token_limited_windows"] += 1
            # NO_OP is not a nominee. Re-encode indices exactly as Kev will see them.
            encoded = encode(window)
            self.totals["windows"] += 1
            offset += len(window)
            yield window, encoded

    def select(self, actions, encode, policy, decisions=None):
        """Append sampled nominations to `decisions`; None selects deterministic top-k.

        The accumulator also retains choices made before a later context failure,
        so a failed rollout still trains every stochastic choice it made.
        """
        noop, edges = actions[0], tuple(actions[1:])
        self.totals["legal_actions"] += len(actions)
        if not edges:
            self.totals["prepared_decisions"] += 1
            self.totals["offered_actions"] += 1
            self.totals["noop_only_decisions"] += 1
            return (noop,), None
        first_round = True
        while True:
            nominees = []
            self.totals["reduction_rounds"] += 1
            for window, encoded in self.windows(edges, noop, encode):
                if first_round:
                    self.totals["inspected_edges"] += len(window)
                if len(window) <= self.nominees_per_window:
                    # Keeping the whole window is deterministic; it has no policy term.
                    nominees.extend(window)
                    continue
                distribution = policy.distribution(encoded)
                self.totals["nomination_forwards"] += 1
                self.totals["encoded_tokens"] += len(encoded["ids"])
                if decisions is None:
                    logps = distribution.logits.cpu().tolist()
                    selected = sorted(range(len(window)), key=lambda i: -logps[i])[:self.nominees_per_window]
                else:
                    selected = []
                    for _ in range(self.nominees_per_window):
                        conditional = conditional_distribution(distribution, tuple(selected))
                        index = int(conditional.sample().item())
                        decisions.append(Decision(encoded, index,
                                                  old_log_probability=float(conditional.logits[index]),
                                                  excluded=tuple(selected), stage="nomination"))
                        selected.append(index)
                # Keep graph order in the shortlist; draw order lives in each mask.
                nominees.extend(window[i] for i in sorted(selected))
            final = (noop, *nominees)
            if len(nominees) <= self.candidate_edges:
                try:
                    encoded = encode(final)
                except ContextOverflow as error:
                    if error.state_tokens is not None:
                        raise
                else:
                    self.totals["prepared_decisions"] += 1
                    self.totals["offered_actions"] += len(final)
                    self.totals["final_forwards"] += 1
                    self.totals["encoded_tokens"] += len(encoded["ids"])
                    return final, encoded
            if len(nominees) >= len(edges):
                raise ContextOverflow("window nominees cannot fit or shrink within row-tokens; increase row-tokens "
                                      "or reduce nominees-per-window")
            edges, first_round = tuple(nominees), False


def window_decision(policy, query, graph, names, trajectory, hops, sampler, with_value=False, decisions=None):
    if with_value:
        raise ValueError("windowed training uses GRPO without a value head")

    def encode(actions):
        return policy.encode(observation(query, trajectory.endpoint, trajectory.path, actions, names, hops))

    actions, encoded = sampler.select(graph.actions(trajectory.endpoint), encode, policy, decisions)
    distribution = policy.distribution(encoded) if encoded is not None else None
    return actions, encoded, distribution, None, None


def grpo(policy, trajectories, answers, optimizer, clip_epsilon=0.2, iterations=1, entropy_weight=0.0):
    """Clipped action-level GRPO, no critic or reference model (KL coefficient 0).

    Sum decisions, average trajectories. Recompute each window once per update,
    accumulate gradients with one encoding live, then update the shared actor.
    Old probabilities and masks remain fixed across the requested update passes.
    """
    if len(trajectories) < 2:
        raise ValueError("GRPO needs at least two trajectories of the same question")
    if not 0 < clip_epsilon < 1 or iterations < 1 or not math.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError("require 0 < clip_epsilon < 1, iterations >= 1 and finite entropy_weight >= 0")
    rewards = [float(t.endpoint in answers) if not t.failure else 0.0 for t in trajectories]
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    advantages = [(r - mean) / (std + 1e-8) for r in rewards]
    steps = [step for t in trajectories for step in t.decisions]
    for step in steps:
        if step.old_log_probability is None or not math.isfinite(step.old_log_probability):
            raise ValueError("GRPO requires finite old log probabilities saved during rollout")
        if step.action_index in step.excluded:
            raise ValueError("a selected action cannot be masked out")
    stats = {"reward": mean, "reward_std": std, "group_size": len(trajectories),
             "advantages": advantages, "zero_variance_group": std == 0,
             "decisions": len(steps), "nomination_decisions": sum(s.stage == "nomination" for s in steps),
             "move_decisions": sum(s.stage == "move" for s in steps),
             "failed_rollouts": sum(t.failure is not None for t in trajectories),
             "loss": 0.0, "entropy_per_decision": 0.0, "clip_fraction": 0.0,
             "approx_old_policy_kl": 0.0, "max_initial_logp_error": 0.0, "grad_norm": 0.0,
             "optimizer_steps": 0, "optimizer_step": False, "baseline_mode": "group"}
    optimizer.zero_grad(set_to_none=True)
    # Skip the whole update, including entropy and optimizer moments. Equal rewards
    # provide no relative task signal, even when every trajectory succeeds.
    if std == 0 or not steps:
        return stats
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        for trajectory, advantage in zip(trajectories, advantages):
            # Two nominations share one exact encoding, but distinct masks/logps.
            for _, choices in groupby(trajectory.decisions, key=lambda s: id(s.encoded)):
                choices = list(choices)
                distribution = policy.distribution(choices[0].encoded)
                loss = distribution.logits.new_zeros(())
                for step in choices:
                    conditional = conditional_distribution(distribution, step.excluded)
                    logp = conditional.logits[step.action_index]
                    log_ratio = logp - step.old_log_probability
                    ratio = log_ratio.exp()
                    if not torch.isfinite(ratio):
                        raise FloatingPointError("non-finite WebQSP GRPO probability ratio")
                    surrogate = torch.minimum(ratio * advantage,
                                              ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
                    entropy = conditional.entropy()
                    loss = loss + (-surrogate - entropy_weight * entropy) / len(trajectories)
                    stats["entropy_per_decision"] += float(entropy.detach()) / (len(steps) * iterations)
                    stats["clip_fraction"] += float((ratio.detach() - 1).abs() > clip_epsilon) / (len(steps) * iterations)
                    stats["approx_old_policy_kl"] += float((ratio - 1 - log_ratio).detach()) / (len(steps) * iterations)
                    if iteration == 0:
                        stats["max_initial_logp_error"] = max(stats["max_initial_logp_error"], float(log_ratio.detach().abs()))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite WebQSP GRPO loss")
                loss.backward()
                stats["loss"] += float(loss.detach()) / iterations
        norm = torch.nn.utils.clip_grad_norm_(policy.model.trainable_parameters(), 1.0, error_if_nonfinite=True)
        stats["grad_norm"] += float(norm) / iterations
        optimizer.step()
        stats["optimizer_steps"] += 1
    stats["optimizer_step"] = True
    return stats
