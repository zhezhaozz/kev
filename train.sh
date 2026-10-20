WEBQSP_DATA="data"
WEBQSP_ARGS=(
  --nsm-dir "$WEBQSP_DATA/nsm/webqsp"
  --names "$WEBQSP_DATA/entities_names.json"
  --device cuda --precision bf16
  --hops 3 --beam-width 8 --row-tokens 8192 --seed 42
)

.venv/bin/python -m kev.webqsp train \
  --checkpoint jaredpalmer/kev-4b@v1.0 \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.train.json" \
  "${WEBQSP_ARGS[@]}" \
  --epochs 3 --rollouts 4 --lr 5e-6 --entropy 0.01 --baseline-decay 0.9 \
  --out "$WEBQSP_DATA/runs/webqsp-rl-v2"