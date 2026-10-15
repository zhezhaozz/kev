# WebQSP graph traversal with Kev

`kev.webqsp` runs the released Qwen3.5 Kev policy on pre-extracted question graphs and optionally adapts its LoRA adapter and pointer head with REINFORCE. Evaluation alone does not train or change the checkpoint. The default checkpoint is `jaredpalmer/kev-4b@v1.0`.

This is a MINERVA-style terminal-reward traversal experiment, not a reproduction of MINERVA's embedding/LSTM architecture. Kev scores outgoing graph moves from the question and path text. Outputs are endpoint IDs; there is no answer-generating LLM or executable SPARQL stage.

The implementation has been source-reviewed but **has not been executed, benchmarked, or trained**. The commands below are for you to run. A100 memory use, runtime, and QA accuracy have not been measured.

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
  --hops 3 --beam-width 8 --row-tokens 8192 --seed 42
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
  --epochs 3 --rollouts 4 --lr 5e-6 --entropy 0.01 --baseline-decay 0.9 \
  --out runs/webqsp-rl
```

This trains only on NSM's train IDs within Microsoft's training file. After each epoch it saves `epoch-N/checkpoint`, evaluates on the disjoint NSM dev IDs, and writes the highest-dev-F1 checkpoint to `best.json` (earlier epoch wins a tie). It never evaluates test during training. Use development results to choose hyperparameters and the checkpoint before reading trained test results.

The loss averages over the sampled trajectories:

$$L = -\frac{1}{K}\sum_{k=1}^{K}\left[(R_k-b)\sum_t \log\pi(a_{k,t}\mid s_{k,t},q) + \beta\sum_t H(\pi(\cdot\mid s_{k,t},q))\right].$$

`R` is 1 only if the final endpoint is in the union of the question's annotated answer sets, otherwise 0. There are no intermediate rewards or gold path labels. `b` is an exponential moving reward baseline from previous questions, updated after the current group's gradient is computed. The trainer samples with no gradient, recomputes each selected action with gradients, accumulates all steps/rollouts, clips the gradient norm to 1, then updates once. It does not reuse trajectories after an optimizer update.

LoRA dropout and attention dropout are disabled in both passes. Temperature is applied once in sampling, gradient computation, and evaluation; it defaults to the checkpoint's stored value and is kept fixed. After RL that value is a policy hyperparameter, **not a newly validated confidence calibration**. The saved metadata states this and clears the parent's calibration evidence.

Select the saved checkpoint and evaluate with the same arguments as the frozen baseline:

```bash
WEBQSP_BEST=$(.venv/bin/python -c 'import json; print(json.load(open("runs/webqsp-rl/best.json", encoding="utf-8"))["checkpoint"])')
.venv/bin/python -m kev.webqsp evaluate \
  --checkpoint "$WEBQSP_BEST" \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.test.json" --split test \
  "${WEBQSP_ARGS[@]}" --out runs/webqsp-rl-test
```

Output directories must be new. `--max-updates N` ends training at that number of optimizer updates and then saves/evaluates development. Checkpoints contain inference weights and provenance; exact optimizer/RNG resume is not implemented. A later `--checkpoint` continuation is a new run with a new optimizer and baseline.

## Protocol and interpretation

- **Inputs:** raw question, the first annotated nonempty topic entity (gold entity linking), current entity, traversed path, and legal actions. Answer lists, inferential chains, SPARQL, constraints, and answer names from supervision never enter observations. The selected topic is independent of answers and graph coverage.
- **Actions:** every outgoing `(relation, destination)` pair, inverse edges by default, plus NO_OP. Duplicate triples are removed; ordering is deterministic. NO_OP stays in place and consumes a step. Each mediator/CVT edge counts as one raw hop; a two-relation semantic question can need more than two raw hops. `--no-inverse-edges` is an explicit alternative protocol.
- **Search:** fixed-horizon beam search with cumulative log probabilities. Beam width 1 is greedy. The reported answer set is the unique endpoints of the retained paths; top-1 is the endpoint of the highest-probability path. Duplicate endpoints may occupy beam slots. Endpoints are not filtered or reranked using gold answers, answer types, or a learned answer model.
- **Metrics:** best-parse macro precision, recall and F1 use Microsoft's original answers. `hits_at_1` and `hits_at_k` are endpoint membership against the union of annotated answers; they are additional retrieval metrics, not answer-set F1. Multiple valid parses are scored separately for F1, rather than unioning their denominators.
- **Coverage:** reports include missing graphs/starts, hop-limited reachable-answer coverage, and name coverage over entity occurrences. These diagnostics may use gold answers only after policy execution; the policy cannot read them. Graph reachability is an upper bound before context limits, not a model score. Test questions missing an NSM graph remain in the denominator with empty predictions.
- **Context:** all actions must fit one strict row. The internal encoder supports variable action counts; this offline runner does not use the HTTP API's 255-choice schema. No candidate pruning or truncation occurs. A context overflow terminates that rollout with reward 0; during beam search the affected path fails. Reports count affected questions, and training logs failed trajectories. If a start state overflows, it yields no gradient. Check these counts on development before committing to a full training run. Change the row limit/horizon on development if needed, then rerun both arms with identical settings. Longer inputs can be expensive and are outside the released model's original short-context training regime.
- **Scope:** pre-extracted NSM graphs impose a retrieval ceiling. This is gold-topic, subgraph-constrained traversal, so results are not directly comparable to live whole-Freebase traversal or full constraint-aware semantic parsing. Multi-answer questions are limited by the number and diversity of retained endpoints. Nothing here claims an accuracy improvement before you run the comparison.

`provenance.json` records arguments, data hashes, code hashes and initial weights. Evaluation reports fingerprint the weights actually evaluated, including saved RL checkpoints. Keep these artifacts alongside results.

## Optional verification

The focused checks cover split isolation, missing-graph denominators, label separation, mediator/inverse/NO_OP traversal, best-parse metrics, overflow handling, and a small analytical policy-gradient update. They require no model weights:

```bash
.venv/bin/python -m pytest tests/test_webqsp.py tests/test_conventions.py -q
```

No tests or model runs were executed while preparing this change.

References: [MINERVA](https://arxiv.org/abs/1711.05851), [NSM preprocessing](https://github.com/RichardHGL/WSDM2021_NSM/tree/main/preprocessing/Freebase), [WebQSP format and scoring](https://ad-research.cs.uni-freiburg.de/benchmarks/webqsp/doc/WebQSP.pdf), and [GNN-RAG](https://github.com/cmavro/GNN-RAG).
