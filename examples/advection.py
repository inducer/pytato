#!/usr/bin/env python
import numpy as np
import pytest
import functools
from dg_tools import DGDiscr1D, integrate, DGOps1D, DGOps1DRef

memoized = functools.lru_cache(maxsize=None)


class AdvectionOperator(object):
    """A class representing a DG advection operator."""

    def __init__(self, discr, c, flux_type, dg_ops):
        """
        Inputs:
            discr: an instance of DGDiscr1D
            c: advection speed parameter
            flux_type: "upwind" or "central"
            dg_ops: An instance of AbstractDGOps1D
        """
        self.discr = discr
        self.c = c
        assert flux_type in ("upwind", "central")
        self.flux_type = flux_type
        self.dg = dg_ops

    def weak_flux(self, vec):
        """Apply the flux, weak form.

        Inputs:
            dg: a DGOps1D instance
            vec: vector of nodal values at the faces

        Signature: (m, 2) -> (m, 2)
        """
        if self.flux_type == "central":
            flux = (vec + self.dg.face_swap(vec)) / 2

        elif self.flux_type == "upwind":
            swp = self.dg.face_swap(vec)
            if self.c >= 0:
                flux = self.dg.array_ops.stack((vec[:, 0], swp[:, 1]), axis=1)
            else:
                flux = self.dg.array_ops.stack((swp[:, 0], vec[:, 1]), axis=1)

        flux = flux * self.c * self.dg.normals

        return flux

    def strong_flux(self, vec):
        """Apply the flux, strong form.

        Inputs:
            dg: a DGOps1D instance
            vec: vector of nodal values at the faces

        Signature: (m, 2) -> (m, 2)
        """
        return self.c * self.dg.normals * vec - self.weak_flux(vec)

    def apply(self, vec):
        """Main operator implementation.

        Signature: (m, n) -> (m, n)
        """
        dg = self.dg
        return -dg.inv_mass(
                dg.face_mass(self.strong_flux(dg.interp(vec)))
                - self.c * dg.stiffness(vec))

    def __call__(self, vec):
        """Apply the DG advection operator to the vector of degrees of freedom.

        Signature: (m*n,) -> (m*n,)
        """
        vec = vec.reshape((self.discr.nelements, self.discr.nnodes))
        return self.apply(vec).reshape((-1,))


def rk4(rhs, initial, t_initial, t_final, dt):
    """RK4 integrator.

    Inputs:
        - rhs: a callable that takes arguments (t, y)
        - initial: initial value
        - t_initial: initial time
        - t_final: final time
        - dt: step size

    Returns:
        The solution computed at the final time.
    """
    t = t_initial
    sol = initial

    while t < t_final:
        dt = min(dt, t_final - t)
        s0 = rhs(t, sol)
        s1 = rhs(t + dt/2, sol + dt/2 * s0)
        s2 = rhs(t + dt/2, sol + dt/2 * s1)
        s3 = rhs(t + dt, sol + dt * s2)
        sol = sol + dt / 6 * (s0 + 2 * s1 + 2 * s2 + s3)
        t += dt

    return sol


def test_rk4():
    assert np.isclose(rk4(lambda t, y: -y, 1, 0, 5, 0.01), np.exp(-5))


@pytest.mark.parametrize("order", (3, 4, 5))
@pytest.mark.parametrize("flux_type", ("central", "upwind"))
def test_ref_advection_convergence(order, flux_type):
    errors = []
    hs = []

    for nelements in (8, 12, 16, 20):
        discr = DGDiscr1D(0, 2*np.pi, nelements=nelements, nnodes=order)
        u_initial = np.sin(discr.nodes())
        op = AdvectionOperator(
                discr, c=1, flux_type=flux_type, dg_ops=DGOps1DRef(discr))
        u = rk4(lambda t, y: op(y), u_initial, t_initial=0, t_final=np.pi,
                dt=0.01)
        u_ref = -u_initial
        hs.append(discr.h)
        errors.append(integrate(discr, (u - u_ref)**2)**0.5)

    eoc, _ = np.polyfit(np.log(hs), np.log(errors), 1)
    assert eoc >= order - 0.1, eoc


@pytest.mark.parametrize("order", (3, 4, 5))
@pytest.mark.parametrize("flux_type", ("central", "upwind"))
def test_advection_convergence(order, flux_type):
    errors = []
    hs = []

    import pytato as pt
    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    for nelements in (8, 12, 16, 20):
        discr = DGDiscr1D(0, 2*np.pi, nelements=nelements, nnodes=order)
        dg_ops = DGOps1D(discr)
        u_initial = np.sin(discr.nodes())

        u = pt.make_placeholder(name="u", shape=(dg_ops.nelements, dg_ops.nnodes),
                                dtype=np.float64)
        op = AdvectionOperator(discr, c=1, flux_type=flux_type,
                               dg_ops=dg_ops)
        result = op.apply(u)

        prog = pt.generate_loopy(result, cl_device=queue.device)

        u = rk4(lambda t, y: prog(
                queue,
                u=y.reshape(nelements, order))[1][0].reshape(-1),
                u_initial, t_initial=0, t_final=np.pi, dt=0.01)
        u_ref = -u_initial
        hs.append(discr.h)
        errors.append(integrate(discr, (u - u_ref)**2)**0.5)

    eoc, _ = np.polyfit(np.log(hs), np.log(errors), 1)
    assert eoc >= order - 0.1, eoc


def main():
    import pytato as pt
    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    nelements = 20
    nnodes = 3

    discr = DGDiscr1D(0, 2*np.pi, nelements=nelements, nnodes=nnodes)
    dg_ops = DGOps1D(discr)

    op = AdvectionOperator(discr, c=1, flux_type="central",
                           dg_ops=dg_ops)
    u = pt.make_placeholder(name="u", shape=(dg_ops.nelements, dg_ops.nnodes),
                            dtype=np.float64)
    result = op.apply(u)

    prog = pt.generate_loopy(result, cl_device=queue.device)
    print(prog.program)

    u = np.sin(discr.nodes())
    print(prog(queue, u=u.reshape(nelements, nnodes))[1][0])


if __name__ == "__main__":
    pytest.main([__file__])
    main()
