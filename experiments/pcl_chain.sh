#!/bin/bash
SPICENN_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
# Chunked gen_pcL2d training: each chunk fits the ~90s window, resumes the weight caps (ICWF) from the prev
# chunk so the in-spice output learning accumulates many epochs. Frozen random Gilbert hidden (FRZHID).
cd "$SPICENN_ROOT"
: "${B:?}"; N=${N:-6}; SV=sav_${RUNTAG:-ch}.txt; rm -f "$SV"
for r in $(seq 1 $N); do
  IC=""; [ -s "$SV" ] && IC="ICWF=$SV"
  out=$(env $B $IC SAVEW="$SV" RUNTAG=${RUNTAG:-ch} timeout 85 python3 "$SPICENN_ROOT/experiments/gen_pcL2d.py" 2>&1 | grep "gen_pcL2d")
  echo "chunk $r: $out"
done
echo PCL_CHAIN_DONE
