# First windowed GRPO experiment

This experiment trains the same Kev LoRA adapter and pointer head for window nominations and the final graph move. Every call uses the existing selection instruction, question, current node and path. Only the candidate list changes. There is no critic, external LLM evaluator, reward model or reference-policy copy; the reference KL coefficient is zero.

The existing sampled-candidate REINFORCE path remains the default. `--algorithm grpo` opts into GRPO and defaults to windowed selection. This is a custom action-level GRPO loop for Kev's pointer probabilities, not TRL's text-generation trainer.

## Run the experiment

Use the data and A100 environment setup in [webqsp.md](webqsp.md). From the repository root:

```bash
WEBQSP_DATA="$PWD/../webqsp-data"
WEBQSP_GRPO_ARGS=(
  --nsm-dir "$WEBQSP_DATA/nsm/webqsp"
  --names "$WEBQSP_DATA/entities_names.json"
  --device cuda --precision bf16
  --hops 3 --beam-width 8 --row-tokens 8192 --seed 42
  --candidate-mode windowed --candidate-edges 128 --nominees-per-window 2
)

# Frozen control with the SAME windowed evaluation policy.
.venv/bin/python -m kev.webqsp evaluate \
  --checkpoint jaredpalmer/kev-4b@v1.0 \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.train.json" --split dev \
  "${WEBQSP_GRPO_ARGS[@]}" --out runs/webqsp-windowed-frozen-dev

# Four full trajectories per question; one actor update per informative group.
.venv/bin/python -m kev.webqsp train \
  --algorithm grpo --checkpoint jaredpalmer/kev-4b@v1.0 \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.train.json" \
  "${WEBQSP_GRPO_ARGS[@]}" \
  --epochs 3 --rollouts 4 --lr 5e-6 --entropy 0 \
  --clip-epsilon 0.2 --grpo-iterations 1 \
  --out runs/webqsp-grpo
```

For an initial bounded smoke run, add `--max-groups 32` and use a separate output directory such as `runs/webqsp-grpo-smoke`. This bounds attempted question groups even if every reward is zero. `--max-updates` instead caps actual actor updates; it may take many groups to reach that limit. Neither command overwrites an existing directory.

Each epoch saves the actor and evaluates the disjoint development split. `best.json` selects the earliest checkpoint with the highest dev F1. Fix the settings and candidate on development before reading test results:

```bash
WEBQSP_BEST=$(.venv/bin/python -c 'import json; print(json.load(open("runs/webqsp-grpo/best.json", encoding="utf-8"))["checkpoint"])')
.venv/bin/python -m kev.webqsp evaluate \
  --checkpoint "$WEBQSP_BEST" \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.test.json" --split test \
  "${WEBQSP_GRPO_ARGS[@]}" --out runs/webqsp-grpo-test
```

Evaluation inherits `candidate-mode`, `candidate-edges` and `nominees-per-window` from a trained checkpoint when those flags are omitted. Explicit flags override them. Keep the horizon, row budget, beam width, precision and seed matched too; those flags are still supplied normally. The numeric checkpoint temperature stays fixed and is not claimed to be calibrated after RL.

CPU unit and toy-actor integration checks have been run. No real WebQSP training, released-checkpoint GPU validation, A100 memory measurement or accuracy comparison was run for this implementation.

## Candidate selection

At each graph hop:

1. Take every legal outgoing edge in the graph's deterministic order, including inverse edges when configured. Partition into windows of **at most 128 edges**, rather than a fixed count of 128 windows. Actual token lengths can make windows smaller. Whole state and option text are retained; packing reserves space for one final `NO_OP`.
2. Use one Kev forward pass per window with more than two edges. During training, sample two distinct nominees without replacement from its logits. The second categorical distribution masks the first draw. A window of one or two edges passes through unchanged and contributes no stochastic selection term.
3. Merge nominees in their original graph order. If the shortlist exceeds the edge cap or token budget, repeat the same window procedure on it. All stochastic draws from every reduction round participate in training.
4. Offer the merged shortlist with **one `NO_OP`**, and sample the graph move. NO_OP consumes one hop; it never competes for a nomination. A node with no edges executes a forced NO_OP without a forward pass or policy term.

For a window distribution `p`, ordered distinct draws `u, v` have probability `p(u) * p(v) / (1 - p(u))`. The runner saves the original encoding, each selected index, the earlier-draw exclusion mask and the corresponding old conditional log probability. It never rebuilds a candidate set while computing the loss. Keeping nominations in the final input's original order removes incidental draw-order changes to that input; draw order still determines the conditional likelihood of the sampled proposal.

An individual edge that cannot fit with NO_OP causes an explicit context failure instead of being silently omitted. If token-limited windows are all too small to reduce an oversized shortlist, the runner fails that path rather than looping or discarding candidates. Increase the row budget or reduce `nominees-per-window` after inspecting development diagnostics. Choices already made before a later failure remain in the rollout and receive terminal reward zero.

Evaluation uses stable top-two nominations in every window and ordinary beam expansion over final moves; `--beam-width 1` is greedy. It consumes no nomination RNG. Beam path scores are sums of final-move log probabilities **conditional on these deterministic shortlists**, not marginal probabilities under the stochastic training proposal policy. Training trajectory log probabilities include every internal draw and move. Evaluation retains the same strict packing, recursive reduction and failure rules.

## Objective and update

A group consists of `G=4` independent complete trajectories of the **same question and start graph**. Windows are internal decisions, not separate GRPO group members. Weights remain fixed while the whole group is sampled sequentially without gradients.

Each trajectory receives terminal reward `R_i=1` if its final endpoint belongs to the annotated answer union, otherwise zero. Context failures also receive zero. Answer annotations are used only after traversal; no gold path, answer or relevance label is shown to either selector.

Use the population standard deviation within the group:

$$A_i = \frac{R_i-\bar R}{\sqrt{\frac{1}{G}\sum_j(R_j-\bar R)^2}+10^{-8}}.$$

Every stochastic nomination and final move in trajectory `i` receives the same `A_i`, including nominations from windows whose candidates were not ultimately selected. For each recorded decision `d`, replay its exact context and mask and compute

$$r_{i,d}(\theta)=\exp(\log\pi_\theta(a_{i,d}\mid x_{i,d},m_{i,d})-\log\pi_{\mathrm{old}}(a_{i,d}\mid x_{i,d},m_{i,d})).$$

The minimized loss is

$$L=-\frac{1}{G}\sum_i\sum_d\left[\min(r_{i,d}A_i,\operatorname{clip}(r_{i,d},1-\epsilon,1+\epsilon)A_i)+\lambda H(\pi_\theta(\cdot\mid x_{i,d},m_{i,d}))\right].$$

The defaults are `epsilon=0.2` and entropy weight `lambda=0`. The loss sums internal decisions and averages trajectories; it does not divide each trajectory by its number of windows. More windows therefore mean more gradient terms as well as more computation. This preserves the sum of log probabilities of the stochastic policy at the initial on-policy update, rather than implicitly downweighting trajectories that visit high-degree nodes.

The replay recomputes one window at a time, uses that forward for both masked nomination terms, and backpropagates before moving to the next encoding. Gradients from both stages accumulate into the same LoRA and pointer-head parameters. The actor uses AdamW with zero weight decay and gradient-norm clipping at 1. Dropout remains disabled and policy temperature is applied exactly once, as in the existing runner.

`--grpo-iterations 1` is deliberately the initial experiment: before its single update, new and old probabilities should agree, so ratio clipping is initially inactive. `--grpo-iterations 2` or higher enables multiple clipped update passes over that collected group, keeping its old probabilities and advantages fixed. All trajectories are discarded before sampling the next group. `--max-updates` also caps these extra passes exactly. The diagnostic `approx_old_policy_kl` compares the current actor to the rollout actor; it is not a reference-model penalty.

Groups with identical rewards (all zero or all one) skip the entire optimizer update, including entropy and optimizer moments. These groups provide no relative task signal. A run with no informative groups still saves and evaluates its unchanged actor and reports zero updates. No critic is initialized or saved. Checkpoint continuation restores actor weights but starts fresh optimizer and RNG state, as in the REINFORCE runner.

## Diagnostics and comparison

`train.jsonl` includes reward mean and population standard deviation, group advantages, nomination/move decision counts, zero-variance-group fraction, actual optimizer updates, clip fraction, old-policy KL estimate, initial replay log-probability error, gradient norm, rollout/group seconds and failure reasons. Replay metrics are zero when an update is skipped; they are not entropy or KL measurements for that group. A large zero-variance fraction is evidence that terminal reward is too sparse or saturated for this grouping, not evidence of successful training.

The `sampling` counters in training and evaluation include:

| Counter | Meaning |
| --- | --- |
| `legal_actions` | All legal edges plus NO_OP at attempted hops |
| `inspected_edges` | Edges covered by the first window pass, excluding NO_OP and later reduction rounds |
| `windows`, `reduction_rounds` | All admitted windows and passes, including additional shortlist reduction |
| `token_limited_windows` | Windows reduced by strict token admission |
| `offered_actions` | Final shortlist sizes including NO_OP at successfully prepared hops |
| `nomination_forwards`, `final_forwards` | Rollout/evaluation actor calls, excluding gradient replay |
| `encoded_tokens` | Tokens in those actor calls, excluding packing probes and deterministic pass-through windows |

Counts from earlier windows survive a later failure. Reachable-answer diagnostics still inspect the full graph after traversal; they are an upper bound before shortlist selection. Check cost and failures as well as dev F1 and hits@1.

Compare the trained actor with the frozen **windowed** control above to measure the effect of adaptation. The older sampled-candidate control remains available through `--candidate-mode sampled`; it answers a separate selection-policy comparison. GRPO also accepts sampled mode to isolate the optimizer change without introducing windows. Use separate output directories and matched initial checkpoints, data, horizon, temperature, token budget, seed and kernel environment.

Run the focused verification without checkpoint downloads:

```bash
.venv/bin/python -m pytest tests/test_webqsp.py tests/test_webqsp_grpo.py tests/test_conventions.py -q
```

The tests check complete candidate coverage, recursive reduction, small/forced windows, failure bookkeeping, conditional likelihoods, an analytical gradient through both stages, unchanged weights throughout group replay, both sides of clipping, repeated updates, all-zero/all-one rewards, CLI validation, evaluation inheritance and checkpoint metadata. A locally generated tiny Qwen3.5 model also exercises the real Policy loader, LoRA/pointer update, frozen-backbone check and checkpoint round trip. CI includes both WebQSP test modules.

Verification for this implementation (Python 3.13, CPU torch 2.8.0, transformers 5.17.0): **70 focused tests passed**. Six seeded EMA/value-baseline comparisons against base commit `39981f7` produced identical actor/critic weights, paths, rewards, RNG state and sampler counts. The broader CI Python suite reported 635 passed, 26 skipped and six failures: `test_local_predictor_scores_long_rows_through_the_shared_prefix` also fails on unchanged `main` because its mock does not accept `return_features`; the other five are two-process distributed resume/FSDP/snapshot checks in this runtime, where Gloo TCP setup is denied (`Operation not permitted`). These are not a passing full-suite or GPU-parity claim.
