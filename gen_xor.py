#!/usr/bin/env python3
# Two-layer ReLU network learning XOR, fully-analog, continuous-time.
#   inputs [a1,a2,bias=1] -> H hidden ReLU units -> 1 linear output
# Forward, backward (transpose read), and asymmetric weight update are ALL
# analog elements evolving in time. Controller = only a1(t),a2(t),tgt(t).
import random
random.seed(7)

H     = 6            # hidden units (overparameterised for reliable XOR)
Ts    = 1.0
dt    = 0.03*Ts
Nstp  = 2500
eta   = 0.40
Bsym  = 1e6          # asymmetry OFF first (huge spring scale). Tighten later.
gm    = 1.0
Cs    = 1.0
bb    = 0.05         # softness of the analog ReLU
INIT  = 0.6          # init weight range +/-

# XOR truth table  (inputs 0/1, target 0/1)
pat = [((0,0),0.0), ((0,1),1.0), ((1,0),1.0), ((1,1),0.0)]

def pwl(sel):
    pts=[]
    for k in range(Nstp):
        v=sel(pat[k%4]); t0=k*Ts
        pts.append((t0,v)); pts.append((t0+Ts-dt,v))
    pts.append((Nstp*Ts, pts[-1][1]))
    return " ".join(f"{t:.4f} {v:.4f}" for t,v in pts)

A1=pwl(lambda s:s[0][0]); A2=pwl(lambda s:s[0][1]); TG=pwl(lambda s:s[1])

L=[]
def w(s): L.append(s)

w("* ===== XOR, 2-layer ReLU, fully-analog continuous-time backprop =====")
w(f".param eta={eta} Bsym={Bsym} gm={gm} Cs={Cs} bb={bb}")
w("* ---- controller: data waveforms only ----")
w(f"Va1 a1 0 PWL({A1})")
w(f"Va2 a2 0 PWL({A2})")
w(f"Vtg tg 0 PWL({TG})")
w("Vbias bx 0 dc 1")

# ---- weight state caps + random init ----
ic=[]
for j in range(1,H+1):
    for i in ("1","2","b"):
        n=f"w1_{j}_{i}"
        w(f"C_{n} {n} 0 {{Cs}}"); ic.append((n, round(random.uniform(-INIT,INIT),3)))
for j in range(1,H+1):
    n=f"w2_{j}"
    w(f"C_{n} {n} 0 {{Cs}}"); ic.append((n, round(random.uniform(-INIT,INIT),3)))
w(f"C_w2_b w2_b 0 {{Cs}}"); ic.append(("w2_b", round(random.uniform(-INIT,INIT),3)))

# ---- FORWARD: hidden pre-activations, ReLU, output ----
for j in range(1,H+1):
    w(f"Bz1_{j} z1_{j} 0 V = v(w1_{j}_1)*v(a1)+v(w1_{j}_2)*v(a2)+v(w1_{j}_b)*v(bx)")
    # smooth analog ReLU and its derivative
    w(f"Bh_{j} h_{j} 0 V = 0.5*(v(z1_{j})+sqrt(v(z1_{j})*v(z1_{j})+bb*bb))")
    w(f"Brp_{j} rp_{j} 0 V = 0.5*(1+v(z1_{j})/sqrt(v(z1_{j})*v(z1_{j})+bb*bb))")
ysum="+".join([f"v(w2_{j})*v(h_{j})" for j in range(1,H+1)])+"+v(w2_b)*v(bx)"
w(f"By y 0 V = {ysum}")
w("Bd2 d2 0 V = v(y)-v(tg)")                      # output error

# ---- BACKWARD (transpose read): same W2_j reused; gate by ReLU' ----
for j in range(1,H+1):
    w(f"Bd1_{j} d1_{j} 0 V = v(d2)*v(w2_{j})*v(rp_{j})")

# ---- UPDATES: asymmetric integrators  dW/dt = gm*(s - |s|*W/Bsym) ----
def integ(node, s):
    w(f"B_u_{node} 0 {node} I = gm*( ({s}) - abs({s})*v({node})/Bsym )")
# output layer
for j in range(1,H+1):
    integ(f"w2_{j}", f"-eta*v(d2)*v(h_{j})")
integ("w2_b", "-eta*v(d2)*v(bx)")
# hidden layer
for j in range(1,H+1):
    integ(f"w1_{j}_1", f"-eta*v(d1_{j})*v(a1)")
    integ(f"w1_{j}_2", f"-eta*v(d1_{j})*v(a2)")
    integ(f"w1_{j}_b", f"-eta*v(d1_{j})*v(bx)")

# ---- initial conditions ----
w(".ic " + " ".join([f"v({n})={v}" for n,v in ic]))

w(".control")
w(f"  tran {0.05*Ts} {Nstp*Ts} uic")
w("  let loss = v(d2)*v(d2)")
w("  wrdata xor_trace.txt loss v(y) v(a1) v(a2) v(tg)")
w("  echo done")
w(".endc")
w(".end")

open("xor_net.cir","w").write("\n".join(L)+"\n")
print(f"wrote xor_net.cir  (H={H}, {Nstp} presentations, Bsym={Bsym})")
