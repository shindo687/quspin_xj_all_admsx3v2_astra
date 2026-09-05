from __future__ import annotations

import numpy as np
import pytest

import quspin_ad
import chainrules as ad
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def _drive(t, amplitude):
    return amplitude * np.sin(t)


def test_dynamic_fixed_grid_jvp_vjp_and_metadata_boundary():
    drive = quspin_ad.differentiable_drive(_drive, {"amplitude": lambda t, amplitude: np.sin(t)})
    basis = spin_basis_1d(L=1)
    def make(amplitude):
        return hamiltonian([], [["z", [[1.0, 0]], drive, (amplitude,)]], basis=basis, dtype=np.complex128)
    H = make(0.7)
    psi = np.array([1.0 + 0j, 0j])
    times = np.linspace(0.0, 1.0, 9)
    value, tangent = ad.jvp(H.evolve, psi, 0.0, times, eom="SE", tangents={"amplitude": 1.0})
    eps = 1e-5
    oracle = (make(0.7 + eps).evolve(psi, 0.0, times) - make(0.7 - eps).evolve(psi, 0.0, times)) / (2 * eps)
    assert np.allclose(tangent, oracle, rtol=2e-5, atol=2e-7)
    cotangent = np.ones_like(value) * (1.0 + 0.3j)
    _, pullback = ad.vjp(H.evolve, psi, 0.0, times, eom="SE", wrt="amplitude")
    assert np.allclose(pullback(cotangent)["amplitude"], np.real(np.vdot(cotangent, tangent)))

    raw = _drive
    H_bad = hamiltonian([], [["z", [[1.0, 0]], raw, (0.7,)]], basis=basis, dtype=np.complex128)
    with pytest.raises(ad.NonDifferentiablePoint, match="derivative contract"):
        ad.jvp(H_bad.evolve, psi, 0.0, times, tangents={"amplitude": 1.0})


def test_floquet_eigensystem_branch_gap_and_jvp():
    theta = np.array([0.2, 1.1])
    U = np.diag(np.exp(-1j * theta))
    out, tangent = ad.jvp(quspin_ad.floquet_eigensystem, U, 2.0, tangents={"UF": np.zeros_like(U)})
    assert np.allclose(out["EF"], theta / 2.0)
    assert np.allclose(tangent["EF"], 0.0)
    with pytest.raises(ad.NonDifferentiablePoint, match="degenerate"):
        quspin_ad.floquet_eigensystem(np.eye(2), 2.0)


def test_exact_second_jvp_and_hvp():
    p = np.array([0.2, 0.3, 0.5]); q = np.array([0.3, 0.3, 0.4])
    direction = np.array([0.1, -0.04, -0.06])
    second = ad.nested_jvp(quspin_ad.KL_div, p, q, tangents={"p1": direction})
    eps = 2e-4
    oracle = (quspin_ad.KL_div(p + eps * direction, q) - 2 * quspin_ad.KL_div(p, q) + quspin_ad.KL_div(p - eps * direction, q)) / eps**2
    assert np.allclose(second, oracle, rtol=3e-5, atol=3e-6)
    hvp = ad.hvp(quspin_ad.KL_div, p, q, wrt=("p1", "p2"), vector={"p1": direction, "p2": np.zeros(3)})
    assert np.all(np.isfinite(hvp["p1"])) and np.all(np.isfinite(hvp["p2"]))
