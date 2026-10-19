# WebQSP graph traversal with Kev

`kev.webqsp` runs the released Qwen3.5 Kev policy on pre-extracted question graphs and optionally adapts its LoRA adapter and pointer head with REINFORCE. Evaluation alone does not train or change the checkpoint. The default checkpoint is `jaredpalmer/kev-4b@v1.0`.

This is a MINERVA-style terminal-reward traversal experiment, not a reproduction of MINERVA's embedding/LSTM architecture. Kev scores outgoing graph moves from the question and path text. Outputs are endpoint IDs; there is no answer-generating LLM or executable SPARQL stage.

The implementation has been source-reviewed but **has not been executed, benchmarked, or trained**. The commands below are for you to run. A100 memory use, runtime, and QA accuracy have not been measured.

This branch implements **version 2: sampled candidate sets with a direct value head V(s, C)**, built on version 1's identical sampler. Training defaults to `--baseline value`; `--baseline ema` selects version 1's moving-average reward baseline. The default offers at most 128 graph edges plus `NO_OP`, within the configured token budget. `--candidate-edges 0` restores the original all-actions, strict-overflow protocol. Compare frozen and trained checkpoints with the same candidate settings.

## Data

Use these three sources together:

| Input | Source | Purpose |
| --- | --- | --- |
| `WebQSP.train.json`, `WebQSP.test.json` | [Original Microsoft release](https://www.microsoft.com/en-us/download/details.aspx?id=52763) | Questions, gold topic entity, and answer IDs |
| NSM `webqsp.tgz` | [WSDM2021_NSM](https://github.com/RichardHGL/WSDM2021_NSM), [released data folder](https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv) | Question subgraphs, global entity/relation vocabularies, train/dev assignment |
| `entities_names.json` | [GNN-RAG's instructions](https://github.com/cmavro/GNN-RAG/blob/28d31bc0db3d54f0930e6125f1c98b5b1ace1ba4/llm/README.md), [released data folder](https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp) | Separate `MID: name` dictionary for observation text |

The NSM archive contains `train_simple.json`, `dev_simple.json`, `test_simple.json`, `entities.txt`, and `relations.txt`. Despite their `.json` suffix, the three question files are JSONL. Triple indices refer to the **global** vocabulary files, not to positions within the per-question entity list. The published split is 2,848 train / 250 dev / 1,639 test questions.

Keep the archive's entity and relation order. Do not build a name map from Microsoft or NSM answer annotations. The runner ignores NSM's answer fields and reads labels only from Microsoft. Missing names, including unnamed mediator/CVT nodes, remain visible as IDs; two entities with the same name retain different IDs.

From the repository root, create a data directory outside the checkout. Put your downloaded `WebQSP.zip` in that directory, then run:

```bash
WEBQSP_DATA="$PWD/../webqsp-data"
mkdir -p "$WEBQSP_DATA/nsm"
# Copy your downloaded WebQSP.zip to "$WEBQSP_DATA/WebQSP.zip" first.
unzip "$WEBQSP_DATA/WebQSP.zip" -d "$WEBQSP_DATA"

curl -fL 'https://drive.usercontent.google.com/download?id=1KcIVAi4nf2uyflMOz5OSr54FOL2s2tAi&export=download&confirm=t' \
  -o "$WEBQSP_DATA/webqsp.tgz"
curl -fL 'https://drive.usercontent.google.com/download?id=1H7BqbVsQXr0bGuBsLUphb_Cb0Jf0rSoN&export=download&confirm=t' \
  -o "$WEBQSP_DATA/entities_names.json"
tar -xzf "$WEBQSP_DATA/webqsp.tgz" -C "$WEBQSP_DATA/nsm"
```

If Drive returns a confirmation page instead of a file, use the linked data folders to download the same filenames. The artifacts inspected while writing this implementation had these SHA-256 hashes; verify your copies with `sha256sum`:

```text
95cb9cd2f6b4e1116ba4bec348f08a7f5ca2a5df4f223ea5af3f9b25a971b0ee  WebQSP.zip
6ee4f62792de62d863eea8cdbd92fa64469d0a597217753f080558282c9534bd  webqsp.tgz
93ce094acccb06b1906db8a4534175481efe55bb9fa8ed711fedd1f8b37d2f98  entities_names.json
```

The names dictionary supplies presentation labels only. All transitions, rewards, deduplication, and evaluation use original IDs or literal strings. No name-to-ID reverse mapping is constructed.

## Single A100 80GB setup

```bash
uv sync --python 3.12
# CUDA acceleration for the Qwen3.5 DeltaNet layers; keep this version the same
# for the frozen baseline, training, and trained evaluation.
uv pip install --python .venv/bin/python 'flash-linear-attention==0.5.2'
```

Use the same CUDA/PyTorch/transformers/kernel environment for both arms. The runner records it. The defaults use fp32 backbone/adapter master weights, bf16 forward autocast, an unmerged LoRA adapter, gradient checkpointing, and one decision's gradient graph at a time. There is no quantization, distributed launch, or serving process. Four rollouts per question are accumulated before one optimizer update; they are not four simultaneous GPU batches.

Set the common arguments in the same Bash session:

```bash
WEBQSP_DATA="$PWD/../webqsp-data"
WEBQSP_ARGS=(
  --nsm-dir "$WEBQSP_DATA/nsm/webqsp"
  --names "$WEBQSP_DATA/entities_names.json"
  --device cuda --precision bf16
  --hops 3 --beam-width 8 --row-tokens 8192 --candidate-edges 128 --seed 42
)
```

## Frozen baseline

First inspect development results and graph/context coverage:

```bash
.venv/bin/python -m kev.webqsp evaluate \
  --checkpoint jaredpalmer/kev-4b@v1.0 \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.train.json" --split dev \
  "${WEBQSP_ARGS[@]}" --out runs/webqsp-frozen-dev
```

After fixing the graph/search settings, evaluate the frozen checkpoint on the full original test set:

```bash
.venv/bin/python -m kev.webqsp evaluate \
  --checkpoint jaredpalmer/kev-4b@v1.0 \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.test.json" --split test \
  "${WEBQSP_ARGS[@]}" --out runs/webqsp-frozen-test
```

Every evaluation writes `report.json`, `rows.jsonl` with path traces and failure reasons, and `predictions.json` in the input format expected by Microsoft's supplied `eval/eval.py`. That original script uses Python 2 syntax; the runner implements its quality gate and best-parse macro P/R/F1 directly in Python 3, excluding the original numerical `1e-40` trick. No conversion of entity IDs to answer names is needed.

## REINFORCE adaptation

```bash
.venv/bin/python -m kev.webqsp train \
  --checkpoint jaredpalmer/kev-4b@v1.0 \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.train.json" \
  "${WEBQSP_ARGS[@]}" \
  --epochs 3 --rollouts 4 --lr 5e-6 --entropy 0.01 \
  --baseline value --value-lr 1e-4 \
  --out runs/webqsp-rl-v2
```

This trains only on NSM's train IDs within Microsoft's training file. After each epoch it saves `epoch-N/checkpoint`, evaluates on the disjoint NSM dev IDs, and writes the highest-dev-F1 checkpoint to `best.json` (earlier epoch wins a tie). It never evaluates test during training. Use development results to choose hyperparameters and the checkpoint before reading trained test results.

The loss averages over the sampled trajectories:

$$L_\pi = -\frac{1}{K}\sum_{k=1}^{K}\sum_t\left[(R_k-b_{k,t})\log\pi(a_{k,t}\mid s_{k,t},q,C_{k,t}) + \beta H(\pi(\cdot\mid s_{k,t},q,C_{k,t}))\right].$$

`R` is 1 only if the final endpoint is in the union of the question's annotated answer sets, otherwise 0. There are no intermediate rewards or gold path labels, and no discounting: every decision in a rollout has the same terminal-return target. With `--baseline value`, `b` is the value prediction saved for that decision during rollout. With `--baseline ema`, it is the exponential moving reward baseline from previous questions, updated after the current group's gradient is computed. The trainer samples with no gradient, recomputes each selected action with gradients, accumulates all steps/rollouts, clips the gradient norm to 1, then updates once. It does not reuse trajectories after an optimizer update.

`C` is the offered candidate list at that hop. Each rollout keeps the exact encoding and selected index for gradient recomputation; it never draws another subset while computing that decision's loss. The objective is expected terminal reward under this sampler. The loss averages over rollouts, with no additional division by candidate count and no importance correction toward the full-action policy.

LoRA dropout and attention dropout are disabled in both passes. Temperature is applied once in sampling, gradient computation, and evaluation; it defaults to the checkpoint's stored value and is kept fixed. After RL that value is a policy hyperparameter, **not a newly validated confidence calibration**. The saved metadata states this and clears the parent's calibration evidence.

Select the saved checkpoint and evaluate with the same arguments as the frozen baseline:

```bash
WEBQSP_BEST=$(.venv/bin/python -c 'import json; print(json.load(open("runs/webqsp-rl-v2/best.json", encoding="utf-8"))["checkpoint"])')
.venv/bin/python -m kev.webqsp evaluate \
  --checkpoint "$WEBQSP_BEST" \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.test.json" --split test \
  "${WEBQSP_ARGS[@]}" --out runs/webqsp-rl-v2-test
```

Output directories must be new. `--max-updates N` ends training at that number of actor optimizer updates and then saves/evaluates development. Checkpoints contain actor weights, value-head weights when enabled, and provenance; exact optimizer/RNG resume is not implemented. A later `--checkpoint` continuation restores the actor and, in value mode, an available saved critic, but starts new optimizers/RNG streams and resets the reward EMA. Evaluation uses only the actor.

## Direct value baseline

For the sampled, ordered candidate list `C`, the same Kev forward pass supplies action logits and the hidden vector at `<decide>`. That token can attend to the question, traversal state, history and all offered options. The critic estimates success under the current policy and future candidate sampling:

$$V_\phi(s,C)=\operatorname{sigmoid}(w^\top\operatorname{stopgrad}(h_{\text{decide}})+b).$$

The head is a single fp32 linear projection with a sigmoid, giving values between 0 and 1 for the binary terminal reward. Zero initialization produces 0.5 before any fitting; this is an initialization, not an estimated success rate. Creating the head preserves the actor's random-number stream. The actor's LoRA and pointer head retain their existing optimizer and learning rate. The critic has its own AdamW optimizer, no weight decay, gradient clipping at 1, and `--value-lr 1e-4` as a starting value to assess on development data.

For each question's rollout group:

1. Sample candidate sets and actions without gradients. Before each action is sampled, save its scalar value prediction and a detached CPU copy of the `<decide>` vector, along with the exact action encoding/index.
2. Compute terminal rewards, then update the actor using the saved advantage `R - V(s,C)`. The baseline is a constant in this loss; it is not recomputed after fitting the current reward.
3. Fit only the small value head to the saved vectors and terminal returns, after the actor update. The critic loss is the mean squared error across the group's recorded decisions, `mean((V - R)^2)`. No extra backbone forward pass is needed, and critic gradients cannot change the backbone, LoRA adapter or pointer head.

The actor loss averages over trajectories and sums their decisions; the critic loss averages over recorded decisions. These normalizations serve separate optimizers. Forced `NO_OP` moves have no action choice and generate no actor/critic training sample. A failed rollout still trains its earlier recorded decisions with terminal reward 0. A group with no recorded decisions updates neither optimizer.

This implements the direct value baseline only: there is no Q head, TD bootstrapping, GAE, action-dependent baseline or `B(s,C)` correction. The baseline can reduce gradient variance; it does not add reward signals when successful paths are rare.

In `train.jsonl`, `baseline_mode` identifies the method; `baseline_mean` and `advantage_mean` average the values actually used over decisions. `value_mse_before_update` compares the saved rollout predictions with returns; `value_loss` is the critic's pre-step training MSE on those same vectors (normally the same up to numerical batching differences). `value_optimizer_step` records whether the critic updated. `reward_ema` remains an earlier-reward reference in value mode and is not used in its actor loss. Fields without a sample are `null`.

The critic is saved in the `webqsp_value_head` metadata inside `head.pt`, using the normal checkpoint helpers. Inference remains compatible with the actor-only loader. A value-mode continuation restores this head when present; an original or version-1 checkpoint starts a fresh head. Training with `--baseline ema` omits the critic from subsequent checkpoints, so a stale critic is not carried along with a newly changed actor.

For the version-1 comparison on this branch, use the same command and released starting checkpoint, replace `--baseline value --value-lr 1e-4` with `--baseline ema --baseline-decay 0.9`, and choose a new output directory such as `runs/webqsp-rl-v1`. Start both versions from the same initial actor checkpoint rather than training version 2 on version 1's result. Keep the candidate cap, token budget, seed, rollout count, actor learning rate, horizon and evaluation settings matched.

## Candidate sampling

`--candidate-edges N` caps edge moves, excluding the always-retained `NO_OP`:

1. Above the cap, draw N edges uniformly without replacement, keeping their random draw order. Otherwise retain the full legal list in its original order.
2. Encode the complete state, instruction and offered options with the strict Kev encoder.
3. If the row is too long, use a random edge order (shuffle the full list if it was not already sampled) and keep the longest prefix that fits, with `NO_OP` first. A binary search checks actual encodings; no option or state text is cut.
4. If only `NO_OP` fits, execute it as a forced move with no policy gradient. If even the minimal prompt does not fit, retain the existing overflow failure behavior.

The token-budget step produces variable-size subsets. It samples prefixes, not uniformly from every possible budget-feasible subset, and does not search for a better packing after a long early option. `--candidate-edges 0` disables both the cap and budget-based candidate removal: every legal action must fit, as in the original runner. Sampling never reads answer annotations, and the underlying graph remains complete for reachability diagnostics.

Training and beam evaluation call the same candidate constructor. Each training run has a continuing candidate RNG seeded by `--seed`, separate from question shuffling and PyTorch's action sampling. Every evaluation starts a fresh candidate RNG from that seed, so development evaluation does not consume training randomness. Repeated evaluation is reproducible for a fixed policy and environment; different policies can visit different paths and consume the stream differently. Use the same cap, row budget and seed when comparing checkpoints.

`train.jsonl` and evaluation `rows.jsonl` contain per-question `sampling` counts. `report.json` also contains their totals and the `candidate_sampling` configuration. Counts cover successfully prepared decisions: `legal_actions` and `offered_actions` include `NO_OP`, `capped_decisions` counts degree-cap sampling, `token_limited_decisions` counts further budget reductions, `noop_only_decisions` counts forced moves, and `encoded_tokens` sums the lengths actually admitted. A naturally forced `NO_OP` needs no encoding and contributes zero tokens. Failed prompt construction is recorded separately in the existing overflow/failure fields. Reachable-answer coverage still refers to the full graph, before candidate sampling.

## Protocol and interpretation

- **Inputs:** raw question, the first annotated nonempty topic entity (gold entity linking), current entity, traversed path, and legal actions. Answer lists, inferential chains, SPARQL, constraints, and answer names from supervision never enter observations. The selected topic is independent of answers and graph coverage.
- **Actions:** the legal pool contains every outgoing `(relation, destination)` pair, inverse edges by default, plus NO_OP. Duplicate triples are removed; the full pool has deterministic order. Candidate sampling selects the list actually presented to Kev. NO_OP stays in place and consumes a step. Each mediator/CVT edge counts as one raw hop; a two-relation semantic question can need more than two raw hops. `--no-inverse-edges` is an explicit alternative protocol.
- **Search:** fixed-horizon beam search with cumulative log probabilities. Beam width 1 is greedy. The reported answer set is the unique endpoints of the retained paths; top-1 is the endpoint of the highest-probability path. Duplicate endpoints may occupy beam slots. Endpoints are not filtered or reranked using gold answers, answer types, or a learned answer model.
- **Metrics:** best-parse macro precision, recall and F1 use Microsoft's original answers. `hits_at_1` and `hits_at_k` are endpoint membership against the union of annotated answers; they are additional retrieval metrics, not answer-set F1. Multiple valid parses are scored separately for F1, rather than unioning their denominators.
- **Coverage:** reports include missing graphs/starts, hop-limited reachable-answer coverage, and name coverage over entity occurrences. These diagnostics may use gold answers only after policy execution; the policy cannot read them. Graph reachability is an upper bound before context limits, not a model score. Test questions missing an NSM graph remain in the denominator with empty predictions.
- **Context:** the offered candidates must fit one strict row. The internal encoder supports variable action counts; this offline runner does not use the HTTP API's 255-choice schema. Sampling happens before the model call, and whole options are removed only under the explicit sampler above. An unresolvable context overflow terminates that rollout with reward 0; during beam search the affected path fails. Reports count affected questions, and training logs failed trajectories. If a start state overflows, it yields no gradient. Check these counts on development before committing to a full training run. Longer inputs can be expensive and are outside the released model's original short-context training regime.
- **Scope:** pre-extracted NSM graphs impose a retrieval ceiling. This is gold-topic, subgraph-constrained traversal, so results are not directly comparable to live whole-Freebase traversal or full constraint-aware semantic parsing. Multi-answer questions are limited by the number and diversity of retained endpoints. Nothing here claims an accuracy improvement before you run the comparison.

`provenance.json` records arguments, data hashes, code hashes and initial weights. Evaluation reports fingerprint the weights actually evaluated, including saved RL checkpoints. Keep these artifacts alongside results.

## Optional verification

The focused checks cover split isolation, missing-graph denominators, label separation, mediator/inverse/NO_OP traversal, best-parse metrics, seeded candidate sampling, strict token budgeting, training/evaluation candidate consistency, overflow handling, analytical actor updates, value-gradient isolation, saved baselines, feature readout in packed/causal/shared-prefix forms, and value-head checkpoint round trips. They require no model weights:

```bash
.venv/bin/python -m pytest tests/test_webqsp.py tests/test_conventions.py -q
```

No tests or model runs were executed while preparing this change.

References: [MINERVA](https://arxiv.org/abs/1711.05851), [NSM preprocessing](https://github.com/RichardHGL/WSDM2021_NSM/tree/main/preprocessing/Freebase), [WebQSP format and scoring](https://ad-research.cs.uni-freiburg.de/benchmarks/webqsp/doc/WebQSP.pdf), and [GNN-RAG](https://github.com/cmavro/GNN-RAG).
