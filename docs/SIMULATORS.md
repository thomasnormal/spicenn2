# Three-simulator cross-validation: ngspice, Spectre, Xyce

Our in-circuit trainer results are **not ngspice-specific numerical artifacts** — the same deck
gives the same answer on the production Cadence simulator (Spectre) and on Xyce (Sandia),
built from source on this machine.

## Result (XOR frozen-weights inference)
| XOR | ngspice | Spectre | Xyce |
|---|---|---|---|
| (0,0) | 0.23 | 0.235 | 0.235 |
| (0,1) | 1.07 | 1.070 | 1.071 |
| (1,0) | 0.60 | 0.605 | 0.605 |
| (1,1) | 0.36 | 0.357 | 0.357 |

max |Xyce−ngspice| = 0.005 V, |Xyce−Spectre| = 0.001 V. All three: 4/4 corners.

## Converters
- `to_spectre.py deck.cir out.scs` — ngspice -> Spectre (SPICE-lang mode; inlines `{param}`
  braces; replaces `.control` with `.tran`+`.save`). Run:
  `spectre out.scs +mt=16 +aps=moderate -format nutascii -raw out.raw =log spectre.log`
  (note: `-mt` is invalid with `+aps` in 23.1; use `+mt`.) Output is nutascii; parse the
  Variables block (split on the LAST `Variables:`) then reshape Values to (npoints, nvars+1).
- `to_xyce.py deck.cir out.cir` — ngspice -> Xyce (`.control` -> `.TRAN`+`.PRINT TRAN`).
  Output `.prn` is a clean column table (header row + data), no `[1::2]`.

## Gotchas found
- **Xyce convergence**: our inference deck is a *cap-less* resistive-MOS network, which is stiff
  for Xyce's transient Newton — it fails at PWL input transitions with default settings.
  Fix that worked: `.options DEVICE gmin=1e-8 voltagelimiterflag=1` +
  `.options TIMEINT reltol=1e-3 abstol=1e-6 delmax=2e-3` + a fine max timestep. Drop `UIC` for
  cap-less decks (let Xyce do a DC operating point first). `method=2` is invalid (use trap/gear).
- **Spectre** runs SPICE decks directly in `simulator lang=spice` mode (0 errors), but chokes on
  ngspice `{param}` brace expressions -> inline the `.param` values.

## Building Xyce 7.10 from source here (no root, RHEL yum blocked)
Chain (all installed to `~/xyce_deps`, ~96-core parallel build, ~30 min total):
1. **OpenBLAS** (BLAS+LAPACK were absent): clone + `make -j96` + install.
2. **SuiteSparse AMD**: clone, build with `~/Xyce/cmake/trilinos/AMD` helper.
3. **Trilinos 14.4** (Xyce 7.10 wants >=14.4, not the SuperBuild's stale 12.12.1): configure
   with `-C ~/Xyce/cmake/trilinos/trilinos-base.cmake` + OpenBLAS + AMD, `-D Trilinos_ENABLE_Fortran=OFF`
   (no system gfortran), serial, shared libs. `make -j96 install`.
4. **Xyce**: cmake with `Trilinos_ROOT=~/xyce_deps`; `make -j96 Xyce`.
   - **Fortran-symbol fix**: with Fortran OFF, EpetraExt's BTF reordering left `mattrans_`,
     `genbtf_` undefined. Compiled `Trilinos/packages/epetraext/src/btf/pothen/*.f` with a
     gfortran bundled in the Cadence toolchain (`/opt/cadence/.../cdsgcc/gcc/9.3/.../gfortran`,
     `-std=legacy`) into `~/xyce_deps/lib/libgenbtf.so` (self-contained via rpath) and added
     `-lgenbtf` to `CMAKE_EXE_LINKER_FLAGS`. Do NOT add the gcc-9.3 lib dir to `-L` — it shadows
     the system libstdc++ and breaks GCC-11 symbols (`std::__throw_bad_array_new_length`).

Run: `LD_LIBRARY_PATH=~/xyce_deps/lib ~/Xyce/build/src/Xyce deck.cir`
