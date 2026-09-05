"""ChainRules rules for small, continuous QuSpin API functions.

The wrappers below deliberately do not duplicate QuSpin's numerical
implementations.  Each wrapper calls its upstream function for the primal
value, and the registered rules contain only the corresponding linear map or
adjoint map.  Matrix and state dimensions are fixed while differentiating;
basis construction, sparse operator assembly, eigensolver choices and other
discrete operations are outside this module's support domain.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

try:  # Prefer the standalone ChainRules package when available.
    import chainrules as ad
except ModuleNotFoundError:  # pragma: no cover - exercised in clean offline venvs
    from . import _chainrules as ad


def _native(path: str) -> Callable[..., Any]:
    """Resolve an upstream callable lazily.

    Lazy resolution keeps ``quspin_ad`` importable while a user is preparing a
    fresh environment.  No fallback implementation is provided: calling a
    wrapper without QuSpin installed raises the normal import error.
    """
    module_name, name = path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)


def _unsupported(
    function: Callable[..., Any], names: Iterable[str], supported: Iterable[str]
) -> None:
    bad = set(names) - set(supported)
    if bad:
        raise ad.UnsupportedWrt(function, bad, supported=supported)


def _active(tangents: Mapping[str, object], name: str) -> object:
    return tangents.get(name, ad.ZERO)


def _array(value: object, *, name: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - numpy controls the error
        raise TypeError(f"{name} must be array-like") from exc


def _same_shape(value: object, reference: np.ndarray, *, name: str) -> np.ndarray:
    array = _array(value, name=name)
    if array.shape != reference.shape:
        raise ValueError(f"{name} shape {array.shape} does not match {reference.shape}")
    return array


def _input_gradient(value: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Project a real-linear gradient back to the primal input dtype.

    QuSpin accepts real arrays for several callables whose outputs may be
    complex.  Under ChainRules' real inner product, a real input has a real
    cotangent; discard the imaginary component introduced by a complex output
    cotangent in that case.
    """
    return np.real(gradient) if not np.iscomplexobj(value) else gradient


def KL_div(p1: object, p2: object) -> Any:
    """Call :func:`quspin.tools.misc.KL_div` (primal only)."""
    return _native("quspin.tools.misc.KL_div")(p1, p2)


@ad.rules.jvp_for(KL_div)
def _kl_jvp(
    tangents: Mapping[str, object], p1: object, p2: object
) -> tuple[Any, object]:
    value = KL_div(p1, p2)
    _unsupported(KL_div, tangents, ("p1", "p2"))
    dp1 = _active(tangents, "p1")
    dp2 = _active(tangents, "p2")
    if dp1 is ad.ZERO and dp2 is ad.ZERO:
        return value, ad.ZERO
    x = _array(p1, name="p1")
    y = _array(p2, name="p2")
    tangent = 0.0
    if dp1 is not ad.ZERO:
        tangent = tangent + np.sum(
            (np.log(x / y) + 1.0) * _same_shape(dp1, x, name="dp1")
        )
    if dp2 is not ad.ZERO:
        tangent = tangent - np.sum((x / y) * _same_shape(dp2, y, name="dp2"))
    return value, tangent


@ad.rules.vjp_for(KL_div)
def _kl_vjp(
    wrt: tuple[str, ...], p1: object, p2: object
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    _unsupported(KL_div, wrt, ("p1", "p2"))
    value = KL_div(p1, p2)
    x = _array(p1, name="p1")
    y = _array(p2, name="p2")
    g1 = np.log(x / y) + 1.0
    g2 = -(x / y)

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        result: dict[str, object] = {}
        if "p1" in wrt:
            result["p1"] = _input_gradient(x, np.asarray(cotangent) * g1)
        if "p2" in wrt:
            result["p2"] = _input_gradient(y, np.asarray(cotangent) * g2)
        return result

    return value, pullback


def coherent_state(a: object, n: int, dtype: object = np.float64) -> Any:
    """Call :func:`quspin.basis.coherent_state` (primal only)."""
    return _native("quspin.basis.coherent_state")(a, n, dtype=dtype)


def _coherent_linearization(value: np.ndarray, a: object, da: object) -> np.ndarray:
    aa = np.asarray(a)
    if aa.ndim != 0:
        raise TypeError("coherent_state AD currently requires scalar a")
    if aa == 0 or not np.all(np.isfinite(value)):
        raise ad.NonDifferentiablePoint(
            "coherent_state has no stable rule at a=0 or non-finite amplitude"
        )
    k = np.arange(value.size, dtype=np.result_type(value.dtype, np.float64))
    # Real-linear convention: |a|^2 contributes -Re(conj(a) da), while a^k
    # contributes k da/a.  This also specializes correctly to real ``a``.
    daa = np.asarray(da)
    if daa.ndim != 0:
        raise TypeError("coherent_state AD requires a scalar tangent da")
    logarithmic = -np.real(np.conj(aa) * daa) + k * daa / aa
    return value * logarithmic


@ad.rules.jvp_for(coherent_state)
def _coherent_jvp(
    tangents: Mapping[str, object], a: object, n: int, dtype: object = np.float64
) -> tuple[Any, object]:
    value = coherent_state(a, n, dtype=dtype)
    _unsupported(coherent_state, tangents, ("a",))
    da = _active(tangents, "a")
    if da is ad.ZERO:
        return value, ad.ZERO
    return value, _coherent_linearization(np.asarray(value), a, da)


@ad.rules.vjp_for(coherent_state)
def _coherent_vjp(
    wrt: tuple[str, ...], a: object, n: int, dtype: object = np.float64
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    _unsupported(coherent_state, wrt, ("a",))
    value = coherent_state(a, n, dtype=dtype)
    aa = np.asarray(a)
    if aa.ndim != 0:
        raise TypeError("coherent_state AD currently requires scalar a")
    if aa == 0 or not np.all(np.isfinite(value)):
        raise ad.NonDifferentiablePoint(
            "coherent_state has no stable rule at a=0 or non-finite amplitude"
        )
    k = np.arange(np.asarray(value).size, dtype=np.result_type(value, np.float64))

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        c = _array(cotangent, name="cotangent").reshape(-1)
        v = np.asarray(value).reshape(-1)
        if c.shape != v.shape:
            raise ValueError("coherent_state cotangent must match the state shape")
        q = np.sum(np.conj(c) * v)
        r = np.sum(np.conj(c) * v * k)
        # g is defined by Re(conj(g) da) = Re(sum(conj(c) dstate)).
        g = -np.real(q) * aa + np.conj(r / aa)
        g = np.real(g) if not np.iscomplexobj(aa) else g
        return {"a": g}

    return value, pullback


def commutator(H1: object, H2: object) -> Any:
    """Call :func:`quspin.operators.commutator` (primal only)."""
    return _native("quspin.operators.commutator")(H1, H2)


def anti_commutator(H1: object, H2: object) -> Any:
    """Call :func:`quspin.operators.anti_commutator` (primal only)."""
    return _native("quspin.operators.anti_commutator")(H1, H2)


def _matrix(value: object, *, name: str) -> np.ndarray:
    arr = _array(value, name=name)
    if arr.ndim != 2:
        raise TypeError(f"{name} AD domain is a rank-2 dense ndarray")
    return arr


def _binary_jvp(
    fn: Callable[..., Any],
    tangents: Mapping[str, object],
    H1: object,
    H2: object,
    plus: bool,
) -> tuple[Any, object]:
    value = fn(H1, H2)
    _unsupported(fn, tangents, ("H1", "H2"))
    d1 = _active(tangents, "H1")
    d2 = _active(tangents, "H2")
    if d1 is ad.ZERO and d2 is ad.ZERO:
        return value, ad.ZERO
    a = _matrix(H1, name="H1")
    b = _matrix(H2, name="H2")
    tangent_dtype = np.result_type(np.asarray(value), a, b)
    if d1 is not ad.ZERO:
        tangent_dtype = np.result_type(tangent_dtype, d1)
    if d2 is not ad.ZERO:
        tangent_dtype = np.result_type(tangent_dtype, d2)
    tangent = np.zeros_like(np.asarray(value), dtype=tangent_dtype)
    if d1 is not ad.ZERO:
        da = _matrix(d1, name="dH1")
        if da.shape != a.shape:
            raise ValueError("dH1 shape must match H1")
        tangent = tangent + da @ b + (b @ da if plus else -(b @ da))
    if d2 is not ad.ZERO:
        db = _matrix(d2, name="dH2")
        if db.shape != b.shape:
            raise ValueError("dH2 shape must match H2")
        tangent = tangent + a @ db + (db @ a if plus else -(db @ a))
    return value, tangent


@ad.rules.jvp_for(commutator)
def _comm_jvp(
    tangents: Mapping[str, object], H1: object, H2: object
) -> tuple[Any, object]:
    return _binary_jvp(commutator, tangents, H1, H2, False)


@ad.rules.jvp_for(anti_commutator)
def _anti_jvp(
    tangents: Mapping[str, object], H1: object, H2: object
) -> tuple[Any, object]:
    return _binary_jvp(anti_commutator, tangents, H1, H2, True)


def _binary_vjp(
    fn: Callable[..., Any],
    wrt: tuple[str, ...],
    H1: object,
    H2: object,
    plus: bool,
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    _unsupported(fn, wrt, ("H1", "H2"))
    value = fn(H1, H2)
    a = _matrix(H1, name="H1")
    b = _matrix(H2, name="H2")

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        g = _matrix(cotangent, name="cotangent")
        result: dict[str, object] = {}
        if "H1" in wrt:
            g_h1 = g @ b.conj().T + (b.conj().T @ g if plus else -(b.conj().T @ g))
            result["H1"] = _input_gradient(a, g_h1)
        if "H2" in wrt:
            g_h2 = a.conj().T @ g + (g @ a.conj().T if plus else -(g @ a.conj().T))
            result["H2"] = _input_gradient(b, g_h2)
        return result

    return value, pullback


@ad.rules.vjp_for(commutator)
def _comm_vjp(
    wrt: tuple[str, ...], H1: object, H2: object
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    return _binary_vjp(commutator, wrt, H1, H2, False)


@ad.rules.vjp_for(anti_commutator)
def _anti_vjp(
    wrt: tuple[str, ...], H1: object, H2: object
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    return _binary_vjp(anti_commutator, wrt, H1, H2, True)


def ED_state_vs_time(
    psi: object, E: object, V: object, times: object, iterate: bool = False
) -> Any:
    """Call QuSpin's exact-diagonalization time evolution routine."""
    return _native("quspin.tools.evolution.ED_state_vs_time")(
        psi, E, V, times, iterate=iterate
    )


def _ed_forward(
    psi: object, E: object, V: object, times: object
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = ED_state_vs_time(psi, E, V, times, iterate=False)
    p = _array(psi, name="psi")
    e = _array(E, name="E")
    mat = _matrix(V, name="V")
    t = _array(times, name="times")
    if (
        p.ndim != 1
        or e.ndim != 1
        or t.ndim != 1
        or p.size != e.size
        or mat.shape != (e.size, e.size)
    ):
        raise TypeError("ED_state_vs_time AD requires 1-D psi, E, times and square V")
    if np.iscomplexobj(e) or np.iscomplexobj(t):
        raise TypeError("ED_state_vs_time AD requires real E and times")
    phase = np.exp(-1j * t[:, None] * e[None, :])
    coeff = mat.conj().T @ p
    return value, phase, coeff, mat, t


@ad.rules.jvp_for(ED_state_vs_time)
def _ed_jvp(
    tangents: Mapping[str, object],
    psi: object,
    E: object,
    V: object,
    times: object,
    iterate: bool = False,
) -> tuple[Any, object]:
    if iterate:
        raise ad.NonDifferentiablePoint("ED_state_vs_time AD requires iterate=False")
    value, phase, coeff, mat, t = _ed_forward(psi, E, V, times)
    _unsupported(ED_state_vs_time, tangents, ("psi", "E", "times"))
    dpsi = _active(tangents, "psi")
    dE = _active(tangents, "E")
    dt = _active(tangents, "times")
    if dpsi is ad.ZERO and dE is ad.ZERO and dt is ad.ZERO:
        return value, ad.ZERO
    p = _array(psi, name="psi")
    e = _array(E, name="E")
    dc = np.zeros_like(coeff, dtype=np.result_type(coeff, np.complex128))
    if dpsi is not ad.ZERO:
        dc = dc + mat.conj().T @ _same_shape(dpsi, p, name="dpsi")
    de = (
        np.zeros_like(np.asarray(E), dtype=np.result_type(E, np.float64))
        if dE is ad.ZERO
        else _same_shape(dE, e, name="dE")
    )
    dtime = (
        np.zeros_like(t, dtype=np.result_type(t, np.float64))
        if dt is ad.ZERO
        else _same_shape(dt, t, name="dtimes")
    )
    dphase = phase * (
        -1j * (dtime[:, None] * _array(E, name="E")[None, :] + t[:, None] * de[None, :])
    )
    # QuSpin returns states in the ``(Hilbert, time)`` orientation for the
    # non-iterator pure-state path (``V.dot(psi_t.T)`` in upstream source).
    # Preserve that exact primal shape here; callers should not need to know
    # that the phase factors are assembled in ``(time, eigenstate)`` order.
    return value, mat @ (dphase * coeff[None, :] + phase * dc[None, :]).T


@ad.rules.vjp_for(ED_state_vs_time)
def _ed_vjp(
    wrt: tuple[str, ...],
    psi: object,
    E: object,
    V: object,
    times: object,
    iterate: bool = False,
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    if iterate:
        raise ad.NonDifferentiablePoint("ED_state_vs_time AD requires iterate=False")
    _unsupported(ED_state_vs_time, wrt, ("psi", "E", "times"))
    value, phase, coeff, mat, t = _ed_forward(psi, E, V, times)
    e = _array(E, name="E")

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        g = _array(cotangent, name="cotangent")
        if g.shape != np.asarray(value).shape:
            raise ValueError("ED_state_vs_time cotangent must match output shape")
        # Y = V @ (phase * coeff).T, so first pull back through the final
        # matrix product and transpose back to phase's (time, eigenstate)
        # orientation.
        g_a = mat.conj().T @ g
        g_s = g_a.T
        g_coeff = np.sum(np.conj(phase) * g_s, axis=0)
        result: dict[str, object] = {}
        if "psi" in wrt:
            result["psi"] = _input_gradient(_array(psi, name="psi"), mat @ g_coeff)
        if "E" in wrt:
            dstate_dE = -1j * t[:, None] * phase * coeff[None, :]
            result["E"] = np.real(np.sum(np.conj(g_s) * dstate_dE, axis=0))
        if "times" in wrt:
            dstate_dt = -1j * phase * e[None, :] * coeff[None, :]
            result["times"] = np.real(np.sum(np.conj(g_s) * dstate_dt, axis=1))
        return result

    return value, pullback


def lin_comb_Q_T(coeff: object, Q_T: object, out: object = None) -> Any:
    """Call :func:`quspin.tools.lanczos.lin_comb_Q_T` (primal only)."""
    return _native("quspin.tools.lanczos.lin_comb_Q_T")(coeff, Q_T, out=out)


def project_op(Obs: object, proj: object, dtype: object = np.complex128) -> Any:
    """Call QuSpin's observable projection routine (primal only)."""
    return _native("quspin.tools.misc.project_op")(Obs, proj, dtype=dtype)


def _projection_inputs(
    Obs: object, proj: object
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Validate dense projection inputs and identify down/up orientation."""
    observable = _matrix(Obs, name="Obs")
    projector = _matrix(proj, name="proj")
    if observable.shape[0] != observable.shape[1]:
        raise TypeError("project_op AD requires a square observable")
    if projector.shape[0] == observable.shape[0]:
        return observable, projector, True
    if projector.shape[1] == observable.shape[0]:
        return observable, projector, False
    raise ValueError("project_op observable/projector dimensions are incompatible")


def _projection_jvp(
    tangents: Mapping[str, object], Obs: object, proj: object, dtype: object
) -> tuple[Any, object]:
    value = project_op(Obs, proj, dtype=dtype)
    _unsupported(project_op, tangents, ("Obs", "proj"))
    observable, projector, down = _projection_inputs(Obs, proj)
    d_obs = _active(tangents, "Obs")
    d_proj = _active(tangents, "proj")
    if d_obs is ad.ZERO and d_proj is ad.ZERO:
        return value, ad.ZERO
    derivative_obs = (
        np.zeros_like(observable)
        if d_obs is ad.ZERO
        else _same_shape(d_obs, observable, name="dObs")
    )
    derivative_proj = (
        np.zeros_like(projector)
        if d_proj is ad.ZERO
        else _same_shape(d_proj, projector, name="dproj")
    )
    if down:
        derivative = (
            derivative_proj.conj().T @ observable @ projector
            + projector.conj().T @ derivative_obs @ projector
            + projector.conj().T @ observable @ derivative_proj
        )
    else:
        derivative = (
            derivative_proj @ observable @ projector.conj().T
            + projector @ derivative_obs @ projector.conj().T
            + projector @ observable @ derivative_proj.conj().T
        )
    return value, {"Proj_Obs": derivative}


@ad.rules.jvp_for(project_op)
def _project_jvp(
    tangents: Mapping[str, object],
    Obs: object,
    proj: object,
    dtype: object = np.complex128,
) -> tuple[Any, object]:
    return _projection_jvp(tangents, Obs, proj, dtype)


@ad.rules.vjp_for(project_op)
def _project_vjp(
    wrt: tuple[str, ...],
    Obs: object,
    proj: object,
    dtype: object = np.complex128,
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    _unsupported(project_op, wrt, ("Obs", "proj"))
    value = project_op(Obs, proj, dtype=dtype)
    observable, projector, down = _projection_inputs(Obs, proj)

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        if not isinstance(cotangent, Mapping) or set(cotangent) != {"Proj_Obs"}:
            raise TypeError("project_op cotangent must map 'Proj_Obs' to a matrix")
        g = _matrix(cotangent["Proj_Obs"], name="cotangent['Proj_Obs']")
        result: dict[str, object] = {}
        if down:
            if "Obs" in wrt:
                result["Obs"] = _input_gradient(
                    observable, projector @ g @ projector.conj().T
                )
            if "proj" in wrt:
                result["proj"] = _input_gradient(
                    projector,
                    observable @ projector @ g.conj().T
                    + observable.conj().T @ projector @ g,
                )
        else:
            if "Obs" in wrt:
                result["Obs"] = _input_gradient(
                    observable, projector.conj().T @ g @ projector
                )
            if "proj" in wrt:
                result["proj"] = _input_gradient(
                    projector,
                    g @ projector @ observable.conj().T
                    + g.conj().T @ projector @ observable,
                )
        return result

    return value, pullback


@ad.rules.jvp_for(lin_comb_Q_T)
def _lincomb_jvp(
    tangents: Mapping[str, object], coeff: object, Q_T: object, out: object = None
) -> tuple[Any, object]:
    if out is not None:
        raise ad.NonDifferentiablePoint("lin_comb_Q_T AD requires out=None")
    value = lin_comb_Q_T(coeff, Q_T, out=out)
    _unsupported(lin_comb_Q_T, tangents, ("coeff", "Q_T"))
    dc = _active(tangents, "coeff")
    dq = _active(tangents, "Q_T")
    if dc is ad.ZERO and dq is ad.ZERO:
        return value, ad.ZERO
    c = _array(coeff, name="coeff")
    q = _array(Q_T, name="Q_T")
    if c.ndim != 1 or q.ndim != 2 or q.shape[0] != c.size:
        raise TypeError("lin_comb_Q_T AD requires coeff shape (m,) and Q_T shape (m,n)")
    tangent = np.zeros(q.shape[1], dtype=np.result_type(c, q))
    if dc is not ad.ZERO:
        tangent = tangent + _same_shape(dc, c, name="dcoeff") @ q
    if dq is not ad.ZERO:
        tangent = tangent + c @ _same_shape(dq, q, name="dQ_T")
    return value, tangent


@ad.rules.vjp_for(lin_comb_Q_T)
def _lincomb_vjp(
    wrt: tuple[str, ...], coeff: object, Q_T: object, out: object = None
) -> tuple[Any, Callable[[object], dict[str, object]]]:
    if out is not None:
        raise ad.NonDifferentiablePoint("lin_comb_Q_T AD requires out=None")
    _unsupported(lin_comb_Q_T, wrt, ("coeff", "Q_T"))
    value = lin_comb_Q_T(coeff, Q_T, out=out)
    c = _array(coeff, name="coeff")
    q = _array(Q_T, name="Q_T")
    if c.ndim != 1 or q.ndim != 2 or q.shape[0] != c.size:
        raise TypeError("lin_comb_Q_T AD requires coeff shape (m,) and Q_T shape (m,n)")

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        g = _array(cotangent, name="cotangent")
        if g.shape != (q.shape[1],):
            raise ValueError("lin_comb_Q_T cotangent must match output shape")
        result: dict[str, object] = {}
        if "coeff" in wrt:
            result["coeff"] = _input_gradient(c, q.conj() @ g)
        if "Q_T" in wrt:
            result["Q_T"] = _input_gradient(q, np.outer(c.conj(), g))
        return result

    return value, pullback


def register_upstream_rules() -> tuple[str, ...]:
    """Register rules for the actual upstream function identities when present.

    The sidecar wrappers are always registered.  This optional bridge lets a
    caller pass ``quspin.tools.misc.KL_div`` (rather than ``quspin_ad.KL_div``)
    to :func:`chainrules.jvp`/``vjp``.  Registration is best-effort and never
    supplies a non-QuSpin fallback primal.
    """
    registered: list[str] = []
    pairs = (
        ("quspin.tools.misc.KL_div", KL_div),
        ("quspin.basis.coherent_state", coherent_state),
        ("quspin.operators.commutator", commutator),
        ("quspin.operators.anti_commutator", anti_commutator),
        ("quspin.tools.evolution.ED_state_vs_time", ED_state_vs_time),
        ("quspin.tools.lanczos.lin_comb_Q_T", lin_comb_Q_T),
        ("quspin.tools.misc.project_op", project_op),
    )
    # RuleRegistry is identity based and intentionally rejects duplicate
    # registration, so only perform this bridge once per process.
    for path, wrapper in pairs:
        try:
            native = _native(path)
            if native is wrapper:
                continue
            # Obtain the private rule functions by callable identity.  This is
            # preferable to maintaining a second, divergent implementation.
            dispatch = {
                KL_div: (_kl_jvp, _kl_vjp),
                coherent_state: (_coherent_jvp, _coherent_vjp),
                commutator: (_comm_jvp, _comm_vjp),
                anti_commutator: (_anti_jvp, _anti_vjp),
                ED_state_vs_time: (_ed_jvp, _ed_vjp),
                lin_comb_Q_T: (_lincomb_jvp, _lincomb_vjp),
                project_op: (_project_jvp, _project_vjp),
            }[wrapper]
            # The public registry has no contains operation; duplicate bridge
            # calls are harmlessly ignored based on RuleNotFound probing.
            try:
                ad.rules.get_jvp(native)
            except ad.RuleNotFound:
                ad.rules.jvp_for(native)(dispatch[0])
            try:
                ad.rules.get_vjp(native)
            except ad.RuleNotFound:
                ad.rules.vjp_for(native)(dispatch[1])
            registered.append(path)
        except (ImportError, ModuleNotFoundError):
            continue
    return tuple(registered)


# In normal installations QuSpin is present and explicit import registration
# is useful.  Keep failures silent so users may inspect/install the sidecar
# before installing QuSpin itself; invoking a wrapper still reports the real
# missing dependency.
register_upstream_rules()


# ---------------------------------------------------------------------------
# Fixed-grid dynamic controls

def differentiable_drive(function: Callable[..., Any], derivatives: Mapping[str, Callable[..., Any]],
                         parameter_names: Iterable[str] | None = None) -> Callable[..., Any]:
    """Attach an explicit derivative contract to a QuSpin drive callback.

    ``derivatives`` maps parameter names to callables evaluated as
    ``derivative(t, *callback_args)``.  QuSpin still receives the original
    callback and arguments; the metadata is used only by the sidecar.
    """
    if not callable(function) or not isinstance(derivatives, Mapping) or not derivatives:
        raise TypeError("a callable and a non-empty derivative mapping are required")
    names = tuple(parameter_names or derivatives)
    if set(names) != set(derivatives):
        raise ValueError("parameter_names and derivatives must contain the same names")
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)
    wrapped.__name__ = getattr(function, "__name__", "drive")
    wrapped._quspin_ad_derivatives = dict(derivatives)
    wrapped._quspin_ad_parameter_names = names
    return wrapped


def _drive_metadata(H: Any) -> tuple[dict[str, list[tuple[Any, Any]]], tuple[str, ...]]:
    """Collect callback derivative contracts and their operator matrices."""
    found: dict[str, list[tuple[Any, Any]]] = {}
    dynamic = getattr(H, "dynamic", None)
    if dynamic is None:
        raise TypeError("dynamic trajectory requires a QuSpin Hamiltonian")
    for callback, matrix in dynamic.items():
        base = getattr(callback, "_f", callback)
        derivatives = getattr(base, "_quspin_ad_derivatives", None)
        if derivatives is None:
            raise ad.NonDifferentiablePoint(
                "every dynamic callback needs an explicit derivative contract; "
                "use differentiable_drive()"
            )
        for name, derivative in derivatives.items():
            if not callable(derivative):
                raise TypeError(f"derivative contract for {name!r} is not callable")
            found.setdefault(name, []).append((callback, (matrix, derivative)))
    if not found:
        raise ad.NonDifferentiablePoint("no differentiable dynamic callback was found")
    return found, tuple(found)


def _dynamic_matrix(H: Any, time: float, tangents: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    static = getattr(H, "_static", None)
    if static is None:
        static = np.zeros((H.Ns, H.Ns), dtype=np.complex128)
    matrix = np.asarray(static.toarray() if hasattr(static, "toarray") else static, dtype=np.complex128).copy()
    metadata, names = _drive_metadata(H)
    derivatives = {name: np.zeros_like(matrix) for name in names}
    for callback, dyn_matrix in getattr(H, "dynamic").items():
        op, derivative_map = dyn_matrix, {}
        base = getattr(callback, "_f", callback)
        contract = getattr(base, "_quspin_ad_derivatives", {})
        coefficient = callback(time)
        cm = np.asarray(op.toarray() if hasattr(op, "toarray") else op, dtype=np.complex128)
        matrix += coefficient * cm
        args = getattr(callback, "_args", ())
        for name, derivative in contract.items():
            derivatives[name] += derivative(time, *args) * cm
    return matrix, derivatives


def _dynamic_sensitivity(H: Any, v0: Any, t0: float, times: Any, directions: Mapping[str, Any],
                        *, eom: str = "SE", checkpoint_interval: int | None = None,
                        solver_name: str = "DOP853", **solver_args: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if eom != "SE":
        raise ad.NonDifferentiablePoint("dynamic trajectory AD currently supports eom='SE' only")
    if checkpoint_interval is not None and (not isinstance(checkpoint_interval, int) or checkpoint_interval < 1):
        raise ValueError("checkpoint_interval must be a positive integer")
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(np.diff(t) < 0):
        raise ValueError("times must be a non-empty, non-decreasing fixed grid")
    state0 = np.asarray(v0, dtype=np.complex128)
    if state0.ndim != 1 or state0.size != H.Ns:
        raise ValueError("dynamic trajectory AD requires a one-dimensional initial state")
    names = tuple(directions)
    value = np.asarray(H.evolve(v0, t0, t, eom=eom, iterate=False, solver_name=solver_name, **solver_args))
    sens = {name: np.zeros_like(value, dtype=np.complex128) for name in names}
    # Integrate the state and all tangent states together.  This is the exact
    # variational equation of the fixed-grid ODE; no perturbed primal solves
    # or finite differences are used.  solve_ivp is only an integrator.
    try:
        from scipy.integrate import solve_ivp
        n = state0.size
        def unpack(y):
            return y[:n], {name: y[(i + 1) * n:(i + 2) * n] for i, name in enumerate(names)}
        y0 = np.concatenate([state0] + [
            np.asarray(d, dtype=np.complex128) if name == "__initial_state__" and d is not ad.ZERO
            else np.zeros(n, dtype=np.complex128)
            for name, d in directions.items()
        ])
        def rhs(time, y):
            state, tangent = unpack(y)
            mat, dmat = _dynamic_matrix(H, float(time), directions)
            dy = [-1j * mat.dot(state)]
            for name in names:
                d = tangent[name]
                drive = (np.zeros_like(state) if name == "__initial_state__" else
                         dmat[name].dot(state) * (0.0 if directions[name] is ad.ZERO else directions[name]))
                dy.append(-1j * (mat.dot(d) + drive))
            return np.concatenate(dy)
        opts = {"rtol": solver_args.pop("rtol", 1e-10), "atol": solver_args.pop("atol", 1e-12), "method": solver_name}
        sol = solve_ivp(rhs, (float(t0), float(t[-1])), y0, t_eval=t, **opts)
        if not sol.success:
            raise RuntimeError(sol.message)
        for i, name in enumerate(names):
            sens[name] = sol.y[(i + 1) * n:(i + 2) * n, :]
    except ImportError:  # pragma: no cover - scipy is a QuSpin dependency
        raise RuntimeError("dynamic trajectory AD requires scipy.integrate.solve_ivp")
    return value, sens


def dynamic_trajectory(H: Any, v0: Any, t0: float, times: Any, *, eom: str = "SE",
                       tangents: Mapping[str, Any] | None = None, checkpoint_interval: int | None = None,
                       **solver_args: Any) -> tuple[np.ndarray, dict[str, np.ndarray]] | np.ndarray:
    """Differentiate a QuSpin dynamic Hamiltonian on a supplied fixed grid."""
    value = np.asarray(H.evolve(v0, t0, times, eom=eom, iterate=False, **solver_args))
    if tangents is None:
        return value
    _, tangent = _dynamic_sensitivity(H, v0, t0, times, tangents, eom=eom,
                                      checkpoint_interval=checkpoint_interval, **solver_args)
    return value, tangent


def _dynamic_args_from_call(function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
    H = function.__self__
    if kwargs:
        # Keep the upstream keyword surface but reject iterator/adaptive paths.
        if kwargs.get("iterate", False):
            raise ad.NonDifferentiablePoint("dynamic trajectory AD requires iterate=False")
    v0 = args[0] if args else kwargs.pop("v0")
    t0 = args[1] if len(args) > 1 else kwargs.pop("t0")
    times = args[2] if len(args) > 2 else kwargs.pop("times")
    return H, v0, t0, times, kwargs


def dynamic_evolve_jvp(function: Any, tangents: Mapping[str, Any], *args: Any, **kwargs: Any):
    H, v0, t0, times, call_kwargs = _dynamic_args_from_call(function, args, kwargs)
    supported = ("psi0", "v0", "psi", "times", "t0") + tuple(_drive_metadata(H)[1])
    _unsupported(function, tangents, supported)
    directions = {name: value for name, value in tangents.items() if name not in ("v0", "psi0", "psi", "times", "t0")}
    dstate = tangents.get("v0", tangents.get("psi0", tangents.get("psi", ad.ZERO)))
    if dstate is not ad.ZERO:
        directions["__initial_state__"] = dstate
    value = np.asarray(function(*args, **call_kwargs))
    integration_kwargs = dict(call_kwargs); integration_kwargs.pop("eom", None); integration_kwargs.pop("iterate", None)
    _, ds = _dynamic_sensitivity(H, v0, t0, times, directions, eom=call_kwargs.get("eom", "SE"), **integration_kwargs)
    tangent = np.zeros_like(value, dtype=np.complex128)
    if "__initial_state__" in ds:
        tangent += ds.pop("__initial_state__")
    if "times" in tangents:
        raise ad.NonDifferentiablePoint("time-grid tangents are not supported; use a fixed grid")
    tangent += sum(ds.values(), np.zeros_like(tangent))
    return value, tangent


def dynamic_evolve_vjp(function: Any, wrt: tuple[str, ...], *args: Any, **kwargs: Any):
    H, v0, t0, times, call_kwargs = _dynamic_args_from_call(function, args, kwargs)
    supported = ("v0", "psi0", "psi") + tuple(_drive_metadata(H)[1])
    _unsupported(function, wrt, supported)
    value = np.asarray(function(*args, **call_kwargs))
    def pullback(cotangent: Any) -> dict[str, Any]:
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        g = np.asarray(cotangent)
        if g.shape != value.shape:
            raise ValueError("dynamic trajectory cotangent must match output shape")
        result = {}
        for name in wrt:
            if name in ("v0", "psi0", "psi"):
                arr = np.zeros_like(v0, dtype=np.complex128)
                for i in np.ndindex(np.asarray(v0).shape):
                    d = np.zeros_like(v0, dtype=np.complex128); d[i] = 1.0
                    integration_kwargs = dict(call_kwargs); integration_kwargs.pop("eom", None); integration_kwargs.pop("iterate", None)
                    _, ds = _dynamic_sensitivity(H, v0, t0, times, {"__x__": d}, eom=call_kwargs.get("eom", "SE"), **integration_kwargs)
                    arr[i] = np.real(np.vdot(g, ds["__x__"]))
                result[name] = _input_gradient(np.asarray(v0), arr)
            else:
                integration_kwargs = dict(call_kwargs); integration_kwargs.pop("eom", None); integration_kwargs.pop("iterate", None)
                _, ds = _dynamic_sensitivity(H, v0, t0, times, {name: 1.0}, eom=call_kwargs.get("eom", "SE"), **integration_kwargs)
                result[name] = np.real(np.vdot(g, ds[name]))
        return result
    return value, pullback


# ---------------------------------------------------------------------------
# Floquet eigensystem with branch and gap checks

def floquet_eigensystem(UF: Any, T: float, *, gap_tol: float = 1e-10,
                        branch: str = "principal") -> dict[str, np.ndarray]:
    U = _matrix(UF, name="UF").astype(np.complex128)
    if not np.isscalar(T) or not np.isfinite(T) or T == 0:
        raise ValueError("T must be a finite non-zero scalar")
    theta, vectors = np.linalg.eig(U)
    EF = np.real(1j / T * np.log(theta))
    order = np.argsort(EF)
    theta, vectors, EF = theta[order], vectors[:, order], EF[order]
    gaps = np.abs(theta[:, None] - theta[None, :])
    np.fill_diagonal(gaps, np.inf)
    if np.any(gaps < gap_tol):
        raise ad.NonDifferentiablePoint("Floquet eigensystem has a degenerate or unresolved spectral gap")
    # Normalize columns and use a deterministic phase convention.  Projectors
    # remain invariant if a caller changes the input eigenvector phases.
    vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
    for i in range(vectors.shape[1]):
        j = int(np.argmax(np.abs(vectors[:, i])))
        vectors[:, i] *= np.exp(-1j * np.angle(vectors[j, i]))
    return {"EF": EF, "VF": vectors, "thetaF": theta}


def _floquet_linear(UF: Any, T: float, dUF: Any, dT: Any, gap_tol: float):
    out = floquet_eigensystem(UF, T, gap_tol=gap_tol)
    U = _matrix(UF, name="UF"); theta = out["thetaF"]; V = out["VF"]
    dU = np.zeros_like(U) if dUF is ad.ZERO else _same_shape(dUF, U, name="dUF")
    dt = 0.0 if dT is ad.ZERO else float(np.asarray(dT))
    dtheta = np.diag(V.conj().T @ dU @ V)
    dEF = np.real(1j * dtheta / (T * theta) - 1j * np.log(theta) * dt / (T * T))
    dV = np.zeros_like(V)
    for i in range(V.shape[1]):
        for j in range(V.shape[1]):
            if i != j:
                dV[:, i] += V[:, j] * (V[:, j].conj() @ dU @ V[:, i]) / (theta[i] - theta[j])
    return out, {"EF": dEF, "VF": dV, "thetaF": dtheta}


@ad.rules.jvp_for(floquet_eigensystem)
def _floquet_jvp(tangents: Mapping[str, Any], UF: Any, T: float, *, gap_tol: float = 1e-10, branch: str = "principal"):
    _unsupported(floquet_eigensystem, tangents, ("UF", "T"))
    out, tangent = _floquet_linear(UF, T, _active(tangents, "UF"), _active(tangents, "T"), gap_tol)
    return out, tangent


@ad.rules.vjp_for(floquet_eigensystem)
def _floquet_vjp(wrt: tuple[str, ...], UF: Any, T: float, *, gap_tol: float = 1e-10, branch: str = "principal"):
    _unsupported(floquet_eigensystem, wrt, ("UF", "T")); out = floquet_eigensystem(UF, T, gap_tol=gap_tol, branch=branch)
    def pullback(cotangent: Any):
        if cotangent is ad.ZERO: return dict.fromkeys(wrt, ad.ZERO)
        if not isinstance(cotangent, Mapping): raise TypeError("Floquet cotangent must map EF, VF, and thetaF outputs")
        result = {}
        for name in wrt:
            if name == "T":
                _, d = _floquet_linear(UF, T, ad.ZERO, 1.0, gap_tol)
                result[name] = sum(np.real(np.vdot(cotangent.get(k, 0), d[k])) for k in d)
            else:
                U = _matrix(UF, name="UF"); grad = np.zeros_like(U, dtype=np.complex128)
                for i in np.ndindex(U.shape):
                    dU = np.zeros_like(U); dU[i] = 1.0
                    _, d = _floquet_linear(U, T, dU, ad.ZERO, gap_tol)
                    grad[i] = sum(np.real(np.vdot(cotangent.get(k, 0), d[k])) for k in d)
                result[name] = grad
        return result
    return out, pullback


# ---------------------------------------------------------------------------
# Exact second directional derivatives for the existing smooth primitives

def _canonical(function: Any) -> Any:
    for candidate in (KL_div, coherent_state, commutator, anti_commutator,
                      ED_state_vs_time, lin_comb_Q_T, project_op, floquet_eigensystem):
        if function is candidate or (getattr(function, "__name__", None) == getattr(candidate, "__name__", None)
                                     and getattr(function, "__module__", "").startswith("quspin")):
            return candidate
    raise ad.RuleNotFound(function, "second-order")


def second_jvp(function: Any, /, *args: Any, tangents: Mapping[str, Any], **kwargs: Any):
    fn = _canonical(function)
    # A direction may be supplied once (D²f[d,d]) or as a pair (D²f[d1,d2]).
    d1, d2 = {}, {}
    for name, direction in tangents.items():
        if isinstance(direction, tuple) and len(direction) == 2:
            d1[name], d2[name] = direction
        else:
            d1[name] = d2[name] = direction
    value = fn(*args, **kwargs)
    z = lambda name: d1.get(name, ad.ZERO)
    w = lambda name: d2.get(name, ad.ZERO)
    if fn is KL_div:
        x, y = _array(args[0], name="p1"), _array(args[1], name="p2"); dx,dy,dxx,dyy=d1.get("p1",ad.ZERO),d1.get("p2",ad.ZERO),d2.get("p1",ad.ZERO),d2.get("p2",ad.ZERO)
        dx=np.zeros_like(x) if dx is ad.ZERO else _same_shape(dx,x,name="dp1"); dy=np.zeros_like(y) if dy is ad.ZERO else _same_shape(dy,y,name="dp2"); dxx=np.zeros_like(x) if dxx is ad.ZERO else _same_shape(dxx,x,name="ddp1"); dyy=np.zeros_like(y) if dyy is ad.ZERO else _same_shape(dyy,y,name="ddp2")
        return np.sum(dx*dxx/x - dx*dyy/y - dy*dxx/y + x*dy*dyy/(y*y))
    if fn is coherent_state:
        a,n=args[0],args[1]; aa=np.asarray(a); da=np.asarray(0 if z("a") is ad.ZERO else z("a")); db=np.asarray(0 if w("a") is ad.ZERO else w("a")); val=np.asarray(value); k=np.arange(val.size,dtype=np.result_type(val,float)); l1=-np.real(np.conj(aa)*da)+k*da/aa; l2=-np.real(np.conj(aa)*db)+k*db/aa; cross=-np.real(np.conj(db)*da)-k*da*db/(aa*aa); return val*(l1*l2+cross)
    if fn in (commutator, anti_commutator):
        a,b=args; da=np.zeros_like(a) if z("H1") is ad.ZERO else z("H1"); db=np.zeros_like(b) if z("H2") is ad.ZERO else z("H2"); dA=np.zeros_like(a) if w("H1") is ad.ZERO else w("H1"); dB=np.zeros_like(b) if w("H2") is ad.ZERO else w("H2"); plus=fn is anti_commutator; op=lambda x,y:x@y+(y@x if plus else -y@x); return op(da,dB)+op(dA,db)
    if fn is lin_comb_Q_T:
        c,q=args[:2]; dc=np.zeros_like(c) if z("coeff") is ad.ZERO else z("coeff"); dq=np.zeros_like(q) if z("Q_T") is ad.ZERO else z("Q_T"); ec=np.zeros_like(c) if w("coeff") is ad.ZERO else w("coeff"); eq=np.zeros_like(q) if w("Q_T") is ad.ZERO else w("Q_T"); return dc@eq+ec@dq
    if fn is project_op:
        O,P=args[:2]; dO=np.zeros_like(O) if z("Obs") is ad.ZERO else z("Obs"); dP=np.zeros_like(P) if z("proj") is ad.ZERO else z("proj"); dO2=np.zeros_like(O) if w("Obs") is ad.ZERO else w("Obs"); dP2=np.zeros_like(P) if w("proj") is ad.ZERO else w("proj"); down=np.asarray(P).shape[0]==np.asarray(O).shape[0];
        if down: return {"Proj_Obs":dP2.conj().T@dO@P+dP.conj().T@dO2@P+dP.conj().T@O@dP2+dP2.conj().T@O@dP+P.conj().T@dO2@dP+P.conj().T@dO@dP2}
        return {"Proj_Obs":dP2@dO@P.conj().T+dP@dO2@P.conj().T+dP@O@dP2.conj().T+dP2@O@dP.conj().T+P@dO2@dP.conj().T+P@dO@dP2.conj().T}
    if fn is ED_state_vs_time:
        psi,E,V,t=args[:4]; dpsi,dE,dt=z("psi"),z("E"),z("times"); epsi,eE,et=w("psi"),w("E"),w("times"); p=np.asarray(psi); ee=np.asarray(E); tt=np.asarray(t); M=np.asarray(V); dpsi=np.zeros_like(p) if dpsi is ad.ZERO else dpsi; dE=np.zeros_like(ee) if dE is ad.ZERO else dE; dt=np.zeros_like(tt) if dt is ad.ZERO else dt; epsi=np.zeros_like(p) if epsi is ad.ZERO else epsi; eE=np.zeros_like(ee) if eE is ad.ZERO else eE; et=np.zeros_like(tt) if et is ad.ZERO else et; phase=np.exp(-1j*tt[:,None]*ee[None,:]); c=M.conj().T@p; dc=M.conj().T@dpsi; ec=M.conj().T@epsi; A1=dt[:,None]*ee[None,:]+tt[:,None]*dE[None,:]; A2=et[:,None]*ee[None,:]+tt[:,None]*eE[None,:]; A12=dt[:,None]*eE[None,:]+et[:,None]*dE[None,:]; dph1=-1j*phase*A1; dph2=-1j*phase*A2; dph12=phase*(-A1*A2-1j*A12); return M@(dph12*c[None,:]+dph1*ec[None,:]+dph2*dc[None,:]).T
    raise ad.RuleNotFound(function, "second-order")


def hessian_vector_product(function: Any, /, *args: Any, wrt: str | Iterable[str], vector: Mapping[str, Any], cotangent: Any = 1.0, **kwargs: Any):
    fn = _canonical(function); names = (wrt,) if isinstance(wrt, str) else tuple(wrt); value = fn(*args, **kwargs); result = {}
    for name in names:
        primal = {"p1": args[0], "p2": args[1]} if fn is KL_div else {}
        if fn is coherent_state: primal={"a":args[0]}
        elif fn in (commutator, anti_commutator): primal={"H1":args[0],"H2":args[1]}
        elif fn is lin_comb_Q_T: primal={"coeff":args[0],"Q_T":args[1]}
        elif fn is project_op: primal={"Obs":args[0],"proj":args[1]}
        elif fn is ED_state_vs_time: primal={"psi":args[0],"E":args[1],"times":args[3]}
        x=np.asarray(primal[name]); out=np.zeros_like(x,dtype=np.result_type(x,float))
        for i in np.ndindex(x.shape):
            d={k:ad.ZERO for k in primal}; d[name]=np.zeros_like(x); d[name][i] = 1
            mixed={k:vector.get(k,ad.ZERO) for k in primal}; sec=second_jvp(fn,*args,tangents={k:(d[k],mixed[k]) for k in primal},**kwargs)
            if isinstance(sec, Mapping): sec=sec.get("Proj_Obs")
            c_arr = np.asarray(cotangent)
            s_arr = np.asarray(sec)
            pairing = np.real(np.sum(np.conj(c_arr) * s_arr)) if c_arr.ndim == 0 else np.real(np.vdot(c_arr, s_arr))
            out[i] = pairing
        result[name]=out
    return result
