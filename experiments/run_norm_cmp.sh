cd ~/spicenn2
CFG="CONN=patch2x2 D=16 INLO=0.3 INHI=1.7 VREFH=0.5"
run () {  # labels C seed wnoise norm
  NORM=$5 python3 prep_digits.py $1 16 40 >/dev/null 2>&1
  env $CFG C=$2 H=8 DATAFILE=digits_train.npz SEED=$3 BHID=0.6 INITH=0.8 VULK=0.6 OWTW=1 WNOISE=$4 NSEED=$3 python3 gen_mc.py 1500 0.3 5e-4 >/dev/null 2>&1
  timeout 350 ngspice -b mc.cir >/dev/null 2>&1
  env $CFG C=$2 H=8 DATAFILE_TE=digits_test.npz AVGW=15 python3 gen_mc_infer.py >/dev/null 2>&1
  timeout 150 ngspice -b mc_infer.cir >/dev/null 2>&1
  printf "%-6s C=%s seed=%s norm=%-9s " "$1" "$2" "$3" "$5"; C=$2 python3 score_mc.py "" 2>&1
}
echo "=== StandardScaler vs per-image, batch ==="
run 3,8 2 1 0 standard;  run 3,8 2 7 0 standard
run 0,6,9 3 1 1e-4 standard; run 0,6,9 3 7 1e-4 standard
echo "=== minmax (per-feature) for reference ==="
run 3,8 2 7 0 minmax
echo DONE
