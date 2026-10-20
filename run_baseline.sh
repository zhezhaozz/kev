WEBQSP_DATA="data"
WEBQSP_ARGS=(
  --nsm-dir "$WEBQSP_DATA/nsm/webqsp"
  --names "$WEBQSP_DATA/entities_names.json"
  --device cuda --precision bf16
  --hops 3 --beam-width 8 --row-tokens 8192 --seed 42
)

#.venv/bin/python -m kev.webqsp evaluate \
#  --checkpoint jaredpalmer/kev-4b@v1.0 \
#  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.test.json" --split test \
#  "${WEBQSP_ARGS[@]}" --out "$WEBQSP_DATA/runs/webqsp-frozen-test"

.venv/bin/python -m kev.webqsp evaluate \
  --checkpoint jaredpalmer/kev-4b@v1.0 \
  --questions "$WEBQSP_DATA/WebQSP/data/WebQSP.train.json" --split dev\
  "${WEBQSP_ARGS[@]}" --out "$WEBQSP_DATA/runs/webqsp-frozen-dev-sampled128"