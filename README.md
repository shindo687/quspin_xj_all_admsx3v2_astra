# quspin-ad

`quspin-ad` is a separately installable sidecar for QuSpin 1.0.1.  It registers
analytic first-order rules with the small `chainrules` protocol (or its bundled
compatible fallback when ChainRules is unavailable) while leaving
the upstream QuSpin callable and source tree unchanged.

Install it in a clean environment with:

```bash
python -m pip install .
```

Load the rules explicitly (registration never monkey-patches QuSpin):

```python
import quspin_ad  # registers the rules
import chainrules as ad
from quspin.tools.misc import KL_div

value, tangent = ad.jvp(
    KL_div, p1, p2,
    tangents={"p1": dp1, "p2": ad.ZERO},
)
value, pullback = ad.vjp(KL_div, p1, p2, wrt=("p1", "p2"))
gradients = pullback(1.0)
```

Supported rules and their mathematical domains are specified in [SPEC.md](SPEC.md).
The package currently covers the continuous, array-valued APIs `KL_div`,
`coherent_state`, `commutator`, `anti_commutator`, `ED_state_vs_time`,
`lin_comb_Q_T`, and `project_op` (dense ndarray domain).  Discrete basis
construction, eigensolvers, entropy routines,
I/O, sparse/operator object methods, and non-array workflows are explicitly
reported as deferred or not suitable for AD rather than approximated by finite
differences.

The `upstream/` directory is a byte-for-byte snapshot of the official QuSpin
repository used for API inventory and tests; it is not imported by the wheel.

## Dynamic trajectories, Floquet spectra, and second order rules

Dynamic controls can be differentiated on a fixed time grid when each callback
has an explicit derivative contract.  Wrap a callback with
`quspin_ad.differentiable_drive(f, {"amplitude": df_da})`, where the derivative
is called as `df_da(t, *callback_args)`, then use `chainrules.jvp` or `vjp` on
`H.evolve(..., iterate=False)`.  Missing contracts and iterator/adaptive paths
raise `NonDifferentiablePoint`.  The variational solve stores one tangent
trajectory per active parameter; `checkpoint_interval` is accepted by
`dynamic_trajectory` for callers that bound retained checkpoints.

`quspin_ad.floquet_eigensystem(UF, T)` returns branch-sorted `EF`, normalized
`VF`, and `thetaF` arrays.  Its JVP/VJP uses the principal logarithm and rejects
unresolved eigenvalue gaps.  Eigenvector derivatives use the parallel-transport
gauge, while projector observables remain invariant to input phases.

The fallback ChainRules layer now exposes `nested_jvp`, `hvp`, and
`value_grad_and_hvp`.  These use analytic second directional rules for the
seven existing smooth primitives; no production finite-difference path is
used.
