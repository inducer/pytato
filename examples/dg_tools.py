import numpy as np
import numpy.polynomial.legendre as leg
import numpy.linalg as la
import contextlib

__doc__ = """
Notation convention for operator shapes
=======================================

* m - number of elements in the discretization
* n - number of volume degrees of freedom per element
"""

from functools import cached_property

import pytato as pt


def ortholegvander(x, deg):
    """See numpy.polynomial.legendre.legvander(). Uses an orthonormal basis."""
    result = leg.legvander(x, deg)
    factors = np.array([np.sqrt((2*n+1)/2) for n in range(0, 1 + deg)])
    return result * factors


def ortholegder(c):
    """See numpy.polynomial.legendre.legder(). Uses an orthonormal basis."""
    fw_factors = np.array([np.sqrt((2*n+1)/2) for n in range(len(c))])
    derivs = leg.legder(c * fw_factors)
    return derivs / fw_factors[:len(derivs)]


def ortholegval(x, c):
    """See numpy.polynomial.legendre.legval(). Uses an orthonormal basis."""
    factors = np.array([np.sqrt((2*n+1)/2) for n in range(len(c))])
    return leg.legval(x, c * factors)


class DGDiscr1D(object):
    """A one-dimensional Discontinuous Galerkin discretization."""

    def __init__(self, left, right, nelements, nnodes):
        """
        Inputs:
            left - left endpoint
            right - right endpoint
            nelements - number of discretization panels
            nnodes - number of degrees of freedom per panel
        """
        self.left = left
        self.right = right
        self.nelements = nelements
        self.nnodes = nnodes

    @cached_property
    def ref_nodes(self):
        """Return reference nodes for a single element.

        Signature: ->(n,)
        """
        nodes, _ = leg.leggauss(self.nnodes)
        return nodes

    @cached_property
    def ref_weights(self):
        """Return reference quadrature weights for a single element.

        Signature: ->(n,)
        """
        _, weights = leg.leggauss(self.nnodes)
        return weights

    def zeros(self):
        """Return a zero solution.

        Signature: ->(n*m,)
        """
        return np.zeros(self.nnodes * self.nelements)

    @property
    def h(self):
        """Return the element size.

        Signature: ->()
        """
        return self.elements[0, 1] - self.elements[0, 0]

    def nodes(self):
        """Return the vector of node coordinates.

        Signature: ->(n*m,)
        """
        centers = (self.elements[:, 0] + self.elements[:, 1]) / 2
        radii = (self.elements[:, 1] - self.elements[:, 0]) / 2
        return ((self.ref_nodes[:, np.newaxis] * radii) + centers).T.ravel()

    @cached_property
    def vdm(self):
        """Return the elementwise Vandermonde (modal-to-nodal) matrix.

        Signature: ->(n, n)
        """
        return ortholegvander(self.ref_nodes, self.nnodes - 1)

    @cached_property
    def _ref_mass(self):
        """Return the (volume) mass matrix for the reference element.

        Signature: ->(n, n)
        """
        return la.inv(self.vdm @ self.vdm.T)

    @cached_property
    def mass(self):
        """Return the elementwise volume mass matrix.

        Signature: ->(n, n)
        """
        h = (self.right - self.left) / self.nelements
        return (h/2) * self._ref_mass

    @cached_property
    def inv_mass(self):
        """Return the inverse of the elementwise volume mass matrix.

        Signature: ->(n, n)
        """
        return la.inv(self.mass)

    @cached_property
    def face_mass(self):
        """Return the face mass matrix.

        The face mass matrix combines the effects of applying the face mass
        operator on each face and interpolating the output to the volume nodes.

        Signature: ->(n, 2)
        """
        return self.interp.T.copy()

    @cached_property
    def diff(self):
        """Return the elementwise differentiation matrix.

        Signature: ->(n, n)
        """
        VrT = []  # noqa: N806
        for row in np.eye(self.nnodes):
            deriv = ortholegder(row)
            VrT.append(ortholegval(self.ref_nodes, deriv))
        Vr = np.vstack(VrT).T  # noqa: N806
        return Vr @ la.inv(self.vdm)

    @cached_property
    def stiffness(self):
        """Return the stiffness matrix.

        Signature: ->(n, n)
        """
        return (self._ref_mass @ self.diff)

    @cached_property
    def interp(self):
        """Return the volume-to-face interpolation matrix.

        Signature: ->(2, n)
        """
        return ortholegvander([-1, 1], self.nnodes - 1) @ la.inv(self.vdm)

    @cached_property
    def elements(self):
        """Return the list of elements, each given by their left/right boundaries.

        Signature: ->(m, 2)
        """
        h = (self.right - self.left) / self.nelements
        return np.array(list(zip(
            np.linspace(self.left, self.right, self.nelements, endpoint=False),
            np.linspace(h + self.left, self.right, self.nelements))))

    @property
    def dg_ops(self):
        """Return a context manager yielding a DGOps1D instance.
        """
        return contextlib.contextmanager(lambda: (yield DGOps1DRef(self)))

    @property
    def normals(self):
        """Return the face unit normals.

        Signature: ->(m, 2)
        """
        result = np.zeros((self.nelements, 2))
        result[:, 0] = -1
        result[:, 1] = 1
        return result


def interpolate(discr, vec, nodes):
    """Return an interpolated solution at *nodes*.

    Input:
        discr - a DGDiscr1D instance
        vec - vector of nodal values at degrees of freedom
        nodes - vector of nodes to interpolate to

    Signature:  (m*n,) -> (len(nodes),)
    """
    elements = discr.elements
    nelements = discr.nelements
    nnodes = discr.nnodes
    inv_vdm = la.inv(discr.vdm)

    sorter = np.argsort(nodes)
    sorted_nodes = nodes[sorter]
    result = []

    indices = np.searchsorted(sorted_nodes, elements)
    for i, (start, end) in enumerate(indices):
        if i == 0:
            start = 0
        elif i == nelements - 1:
            end = len(nodes)

        center = (elements[i][0] + elements[i][1]) / 2
        radius = (elements[i][1] - elements[i][0]) / 2
        element_nodes = sorted_nodes[start:end]
        mapped_nodes = (element_nodes - center) / radius

        modal_vals = inv_vdm @ vec[i * nnodes:(i + 1) * nnodes]
        nodal_vals = ortholegvander(mapped_nodes, nnodes - 1) @ modal_vals
        result.append(nodal_vals)

    result = np.hstack(result)
    unsorter = np.arange(len(nodes))[sorter]
    return result[unsorter]


def integrate(discr, soln):
    """Return the integral of the solution.

    Signature: (n*m,) -> ()
    """
    soln = soln.reshape((discr.nelements, discr.nnodes))
    h = discr.elements[0][1] - discr.elements[0][0]
    weights = discr.ref_weights * h / 2
    return np.sum(soln * weights)


def elementwise(mat, vec):
    """Apply a matrix to rows of the input representing per-element
    degrees of freedom.

    Inputs:
        mat: Shape (a, b)
        vec: Shape (c, b)

    Signature: (a, b), (c, b) -> (c, a)
    """
    return np.einsum("ij,kj->ki", mat, vec)


class AbstractDGOps1D(object):
    def __init__(self, discr):
        self.discr = discr

    @property
    def array_ops(self):
        raise NotImplementedError

    @property
    def normals(self):
        """Return the vector of normals at the faces.

        Signature: ->(m, 2)
        """
        raise NotImplementedError

    def interp(self, vec):
        """Apply elementwise volume-to-face interpolation.

        Signature: (m, n) -> (m, 2)
        """
        raise NotImplementedError

    def inv_mass(self, vec):
        """Apply the elementwise inverse mass matrix.

        Signature: (m, n) -> (m, n)
        """
        raise NotImplementedError

    def stiffness(self, vec):
        """Apply the elementwise stiffness matrix.

        Signature: (m, n) -> (m, n)
        """
        raise NotImplementedError

    def face_mass(self, vec):
        """Apply the elementwise face mass matrix.

        Signature: (m, 2) -> (m, n)
        """
        raise NotImplementedError

    def face_swap(self, vec):
        """Swap values at opposite faces.

        Signature: (m, 2) -> (m, 2)
        """
        raise NotImplementedError


class DGOps1DRef(AbstractDGOps1D):
    """A reference NumPy implementation of the AbstractDGOps1D interface."""

    @AbstractDGOps1D.array_ops.getter
    def array_ops(self):
        return np

    @AbstractDGOps1D.normals.getter
    def normals(self):
        return self.discr.normals

    def interp(self, vec):
        return elementwise(self.discr.interp, vec)

    def inv_mass(self, vec):
        return elementwise(self.discr.inv_mass, vec)

    def stiffness(self, vec):
        return elementwise(self.discr.stiffness, vec)

    def face_mass(self, vec):
        return elementwise(self.discr.face_mass, vec)

    def face_swap(self, vec):
        result = np.zeros_like(vec)
        result[:, 0] = np.roll(vec[:, 1], +1)
        result[:, 1] = np.roll(vec[:, 0], -1)
        return result


class DGOps1D(AbstractDGOps1D):

    @AbstractDGOps1D.array_ops.getter
    def array_ops(self):
        return pt

    def __init__(self, discr):
        self.discr = discr

    nelements = pt.make_size_param("nelements")
    nnodes = pt.make_size_param("nnodes")

    # {{{ DG data

    @cached_property
    def normals(self):
        return pt.make_data_wrapper(self.discr.normals, shape=(self.nelements, 2))

    @cached_property
    def interp_mat(self):
        return pt.make_data_wrapper(self.discr.interp, shape=(2, self.nnodes))

    @cached_property
    def inv_mass_mat(self):
        return pt.make_data_wrapper(self.discr.inv_mass, shape=(self.nnodes,
                                                                self.nnodes))

    @cached_property
    def stiffness_mat(self):
        return pt.make_data_wrapper(self.discr.stiffness, shape=(self.nnodes,
                                                                 self.nnodes))

    @cached_property
    def face_mass_mat(self):
        return pt.make_data_wrapper(self.discr.face_mass, shape=(self.nnodes, 2))

    # }}}

    def interp(self, vec):
        return pt.matmul(self.interp_mat, vec.T).T

    def inv_mass(self, vec):
        return pt.matmul(self.inv_mass_mat, vec.T).T

    def stiffness(self, vec):
        return pt.matmul(self.stiffness_mat, vec.T).T

    def face_mass(self, vec):
        return pt.matmul(self.face_mass_mat, vec.T).T

    def face_swap(self, vec):
        return pt.stack(
                (
                    pt.roll(vec[:, 1], +1),
                    pt.roll(vec[:, 0], -1)),
                axis=1)
