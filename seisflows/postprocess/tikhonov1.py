
import sys
import numpy as np

from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.array import grid2mesh, mesh2grid, stack
from seisflows.tools.tools import exists
from seisflows.config import ParameterError, custom_import
from seisflows.tools.math import nabla

import time

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']


class tikhonov1(custom_import('postprocess', 'regularize')):
    """ Adds regularization options to base class

        Available options include 0-, 1-, and 2- order Tikhonov and total
        variation regularization. While the underlying theory is classical,
        application to unstructured numerical grids via the
        "seisflows.tools.math.nabla" operator is somewhat complicated. 

        So far, can only be used for 2D inversion, because the required spatial
        derivative operator "nabla" is not yet available for 3D grids.
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(tikhonov1, self).check()

        if 'CREEPING' not in PAR:
            setattr(PAR, 'CREEPING', False)

        if not PAR.LAMBDA:
            raise ValueError


    def nabla(self, mesh, m, g):
        if PAR.CREEPING:
            G, grid = mesh2grid(g, mesh)
            #DG = nabla(G, order=1)
            DG = nabla(G, h=[])
            dg = grid2mesh(DG, grid, mesh)
            return -dg/np.mean(m)

        else:
            print('============= Tikhonov1 ================')
            t1 = time.time()
            M, grid = mesh2grid(m, mesh)
            #DM = nabla(M, order=1)
            DM = nabla(M, h=[])
            dm = grid2mesh(DM, grid, mesh)
            t2 = time.time()
            print('=========  Tikhonov1 time : ',t2-t2 ,'===========')
            return dm/np.mean(m)

