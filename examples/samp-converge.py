__copyright__ = "Copyright (C) 2020 University of Illinois Board of Trustees"

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

import os
import math
import random
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pytest

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import DOFArray, thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge import sym as grudge_sym
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import DISCR_TAG_BASE, DTAG_BOUNDARY
from mirgecom.sampling import query_eval, u_eval
from mirgecom.integrators import rk4_step
from mirgecom.diffusion import (
    diffusion_operator,
    DirichletDiffusionBoundary,
    NeumannDiffusionBoundary)
from mirgecom.mpi import mpi_entry_point
from pytools.obj_array import make_obj_array
import pyopencl.tools as cl_tools


@mpi_entry_point
def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 3

    def cube_trig(x, y, z, t):
        return np.exp(-t)*np.sin(x)*np.cos(y)*np.sin(z)

    q_arr = 10*np.random.rand(1000,3)

    Nx = [4, 5, 6, 8, 11, 20]
    err = np.array([])
    dt = 0.0005
    t_final = 0.1

    u_check = np.array([])
    for point in q_arr:
        u_check = np.append(u_check,cube_trig(point[0], point[1], point[2], t_final))

    for nx in Nx:
        if mesh_dist.is_mananger_rank():
            from meshmode.mesh.generation import generate_regular_rect_mesh
            mesh = generate_regular_rect_mesh(
                a=(0,0,0),
                b=(10,10,10),
                nelements_per_axis=(nx,nx,nx),
                boundary_tag_to_face={
                    "bdy_x": ["+x", "-x"],
                    "bdy_y": ["+y", "-y"],
                    "bdy_z": ["+z", "-z"]
                    }
                )
            n_elem = mesh.nelements
            print("%d elements" % n_elem)

            part_per_element = get_partition_by_pymetis(mesh, num_parts)

            local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

            del mesh

        else:
            local_mesh = mesh_dist.receive_mesh_part()

        order = 3

        discr = EagerDGDiscretization(actx, local_mesh, order=order,
                        mpi_communicator=comm)

        nodes = thaw(actx, discr.nodes())

        vis = make_visualizer(discr, order+3 if dim == 2 else order)

        def cube_trig_soln(nodes, t):
            return np.exp(-t) * actx.np.sin(nodes[0]) * actx.np.cos(nodes[1]) * actx.np.sin(nodes[2])

        boundaries = {
            DTAG_BOUNDARY("bdy_x"): DirichletDiffusionBoundary(0.),
            DTAG_BOUNDARY("bdy_y"): NeumannDiffusionBoundary(0.),
            DTAG_BOUNDARY("bdy_z"): DirichletDiffusionBoundary(0.)
        }

        def rhs(t, u):
            return diffusion_operator(discr, quad_tag=DISCR_TAG_BASE, alpha=1.0, 
                boundaries=boundaries, u=u)

        u = cube_trig_soln(nodes, 0.0)
        t = 0
        istep = 0
        while True:
            if istep % 50 == 0:
                print(istep, t, discr.norm(u))
                # vis.write_vtk_file("29-fld-act2-bdy-%03d-%04d.vtu" % (rank, istep),
                #         [
                #             ("u", u)
                #             ])

            if t >= t_final:
                break

            u = rk4_step(u, t, dt, rhs)
            t += dt
            istep += 1


        tol = 1e-5
        u_modal = np.array([])
        for point in q_arr:
            u_point = u_eval(u, point, actx, discr, dim, tol) # <--- 1x1 PyOpenCL array
            u_modal = np.append(u_modal, u_point)

        diff = abs(u_modal - u_check)
        print(diff)
        print(type(diff))

        err_inf = np.amax(diff)
        err_2 = la.norm(diff, ord=2) # <--- This doesn't work
        err = np.append(err, err_2)

    print(err)
if __name__ == "__main__":
    main()

# vim: foldmethod=marker
