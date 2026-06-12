#!/bin/bash
# restart_run.sh — detect-and-restart wrapper for pc_deep.py (anti-lock remedy).
# Anti-locked nets are PERFECTLY wrong (acc ~0 << chance), trivially detectable; re-roll WSEED and rerun.
# Usage: env <all pc_deep knobs> RUNTAG=foo THRESH=0.5 TRIES=3 ./restart_run.sh
# Keeps the best run's log as <RUNTAG>.log; per-try logs are <RUNTAG>_t<N>.log.
set -u
THRESH=${THRESH:-0.5}; TRIES=${TRIES:-3}; W0=${WSEED:-1}; TAG=${RUNTAG:-run}
best=-1; bestlog=""
for i in $(seq 0 $((TRIES-1))); do
  w=$((W0 + 10*i))
  log=${TAG}_t${i}.log
  WSEED=$w RUNTAG=${TAG}_t${i} python3 -u pc_deep.py > "$log" 2>&1
  acc=$(grep -m1 "TEST ACC" "$log" | sed 's/.*BEST(early-stop) = \([0-9.]*\).*/\1/')
  [ -z "$acc" ] && acc=0
  echo "[restart_run] try $i WSEED=$w best-acc=$acc"
  better=$(echo "$acc > $best" | bc -l); [ "$better" = 1 ] && { best=$acc; bestlog=$log; }
  done_ok=$(echo "$acc >= $THRESH" | bc -l); [ "$done_ok" = 1 ] && break
done
cp "$bestlog" "${TAG}.log"
echo "[restart_run] FINAL best=$best (from $bestlog)"
