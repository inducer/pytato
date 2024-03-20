__copyright__ = "Copyright (C) 2024 Kaushik Kulkarni"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import pytato as pt
import numpy as np


def test_push_indirections_on_dg_flux_terms():
    nel = 1000
    ndof = 35  # we love 3D-P4 cells
    n_intface = 2800
    n_facedof = 20
    n_face = 4

    u1 = pt.make_placeholder("u1", (nel, ndof))
    u2 = pt.make_placeholder("u2", (nel, ndof))
    u = u1 + u2
    from_el_indices = pt.make_placeholder("from_el_indices",
                                          n_intface,
                                          np.int32)
    dof_pick_list_indices = pt.make_placeholder("dof_pick_list_indices",
                                                n_intface,
                                                np.int32)
    dof_pick_lists = pt.make_placeholder("dof_pick_lists",
                                         (n_face, n_facedof),
                                         np.int32)
    result = u[from_el_indices.reshape(-1, 1),
               dof_pick_lists[dof_pick_list_indices]]
    transformed = pt.push_axis_indirections_towards_materialized_nodes(result)
    assert transformed == (u1[from_el_indices.reshape(-1, 1),
                              dof_pick_lists[dof_pick_list_indices]]
                           + u2[from_el_indices.reshape(-1, 1),
                                dof_pick_lists[dof_pick_list_indices]])
