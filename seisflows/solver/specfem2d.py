
import sys, os
from os.path import basename, join
from glob import glob
import time

import numpy as np

from seisflows.plugins.solver.specfem2d import smooth_legacy
from seisflows.tools.seismic import getpar, setpar

from seisflows.tools import msg
from seisflows.tools import unix
from seisflows.tools.seismic import call_solver
from seisflows.tools.tools import exists
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
preprocess = sys.modules['seisflows_preprocess']


class specfem2d(custom_import('solver', 'base')):
    """ Python interface for SPECFEM2D

      See base class for method descriptions
    """
    if PAR.MATERIALS == 'LegacyAcoustic':
        parameters = []
        parameters += ['vs']


    def check(self):
        """ Checks parameters and paths
        """
        super(specfem2d, self).check()

        # check time stepping parameters
        if 'NT' not in PAR:
            raise Exception

        if 'DT' not in PAR:
            raise Exception

        if 'F0' not in PAR:
            raise Exception

        # check data format
        if 'FORMAT' not in PAR:
            raise Exception()

        if PAR.FORMAT != 'su':
            raise Exception()


    def check_solver_parameter_files(self):
        """ Checks solver parameters
        """
        nt = getpar('nt', cast=int)
        dt = getpar('deltat', cast=float)
        f0 = getpar('f0', file='DATA/SOURCE', cast=float)

        if nt != PAR.NT:
            if self.taskid == 0: print "WARNING: nt != PAR.NT"
            setpar('nt', PAR.NT)

        if dt != PAR.DT:
            if self.taskid == 0: print "WARNING: dt != PAR.DT"
            setpar('deltat', PAR.DT)

        if f0 != PAR.F0:
            if self.taskid == 0: print "WARNING: f0 != PAR.F0"
            setpar('f0', PAR.F0, filename='DATA/SOURCE')

        if self.mesh_properties.nproc != PAR.NPROC:
            if self.taskid == 0:
                print 'Warning: mesh_properties.nproc != PAR.NPROC'

        if 'MULTIPLES' in PAR:
            if PAR.MULTIPLES:
                setpar('absorbtop', '.false.')
            else:
                setpar('absorbtop', '.true.')


    def generate_data(self, **model_kwargs):
        """ Generates data
        """
        self.generate_mesh(**model_kwargs)

        unix.cd(self.cwd)
        setpar('SIMULATION_TYPE', '1')
        setpar('SAVE_FORWARD', '.false.')

        call_solver(system.mpiexec(), 'bin/xmeshfem2D')
        call_solver(system.mpiexec(), 'bin/xspecfem2D')
        

        if PAR.FORMAT in ['SU', 'su']:
            src = glob('OUTPUT_FILES/*.su')
            dst = 'traces/obs'
            unix.mv(src, dst)

        if PAR.SAVETRACES:
            self.export_traces(PATH.OUTPUT+'/'+'traces/obs')


    def initialize_adjoint_traces(self):
        super(specfem2d, self).initialize_adjoint_traces()

        # work around SPECFEM2D's use of different name conventions for
        # regular traces and 'adjoint' traces
        if PAR.FORMAT in ['SU', 'su']:
            files = glob('traces/adj/*.su')
            unix.rename('.su', '.su.adj', files)

        # work around SPECFEM2D's requirement that all components exist,
        # even ones not in use
        if PAR.FORMAT in ['SU', 'su']:
            unix.cd(self.cwd +'/'+ 'traces/adj')
            for channel in ['x', 'y', 'z', 'p']:
                # li_chao
                src = 'U%s_file_single_d.su.adj' % PAR.CHANNELS[0]
                dst = 'U%s_file_single.su.adj' % channel
                if not exists(dst):
                    unix.cp(src, dst)


    def generate_mesh(self, model_path=None, model_name=None, model_type='gll'):
        """ Performs meshing and database generation
        """
        assert(model_name)
        assert(model_type)

        self.initialize_solver_directories()
        unix.cd(self.cwd)

        assert(exists(model_path))
        self.check_mesh_properties(model_path)

        src = glob(join(model_path, '*'))
        dst = join(self.cwd, 'DATA')
        unix.cp(src, dst)

        if self.taskid == 0:
            self.export_model(PATH.OUTPUT +'/'+ model_name)


    ### low-level solver interface

    def forward(self, path='traces/syn'):
        """ Calls SPECFEM2D forward solver
        """
        setpar('SIMULATION_TYPE', '1')
        setpar('SAVE_FORWARD', '.true.')
        
        time1 = time.time()

        call_solver(system.mpiexec(), 'bin/xmeshfem2D')
        call_solver(system.mpiexec(), 'bin/xspecfem2D')

        time2 = time.time()
        #if (time2-time1) > 60
        #    print('Warning: forward time = %8,0f sec' % (time2-time1))

        if PAR.FORMAT in ['SU', 'su']:
            filenames = glob('OUTPUT_FILES/*.su')
            unix.mv(filenames, path)

    def forward_linesearch(self, path='traces/syn'):
        """ Calls SPECFEM2D forward solver
        """
        setpar('SIMULATION_TYPE', '1')
        setpar('SAVE_FORWARD', '.false.')
        
        time1 = time.time()

        call_solver(system.mpiexec(), 'bin/xmeshfem2D')
        call_solver(system.mpiexec(), 'bin/xspecfem2D')

        time2 = time.time()
        #print('forward time = ' ,time2-time1)

        if PAR.FORMAT in ['SU', 'su']:
            filenames = glob('OUTPUT_FILES/*.su')
            unix.mv(filenames, path)


    def adjoint(self):
        """ Calls SPECFEM2D adjoint solver
        """
        setpar('SIMULATION_TYPE', '3')
        setpar('SAVE_FORWARD', '.false.')
        unix.rm('SEM')
        unix.ln('traces/adj', 'SEM')

        # hack to deal with different SPECFEM2D name conventions for
        # regular traces and 'adjoint' traces
        if PAR.FORMAT in ['SU', 'su']:
            files = glob('traces/adj/*.su')
            unix.rename('.su', '.su.adj', files)

        time1 = time.time()

        call_solver(system.mpiexec(), 'bin/xmeshfem2D')
        call_solver(system.mpiexec(), 'bin/xspecfem2D')

        time2 = time.time()
        #print('adjoint time = ',time2-time1)

    #==== update for GPU support, added by YYR (Dec 22,2021) ===
    #replace the internal version of smoothing in Python
    def smooth(self, input_path='',output_path='',parameters=[],span=0.):
        """ Smoooths kernels by convolving them with a Gaussian. Wrapper over
            xsmooth_sem utility.
        """
        print('Starting smoothing')

        t1 = time.time()
        if not exist(input_path):
            raise Exception
        if not exist(output_path):
            unix.mkdir(output_path)

        unix.cd(self.cwd)
        os.system('cp %s %s' %(PATH.SPECFEM_DATA+'proc000000_jacobian.bin',input_path))
        os.system('cp %s %s' %(PATH.SPECFEM_DATA+'proc000000_NSPEC_ibool.bin',input_path))
        os.system('cp %s %s' %(PATH.SPECFEM_DATA+'proc000000_z.bin',input_path))
        os.system('cp %s %s' %(PATH.SPECFEM_DATA+'proc000000_x.bin',input_path))

        # apply smoothing operator
        for name in parameters or self.parameters:
            print 'smoothing', name
            # using GPU to speed up smoothing
            call_solver(
                system.mpiexec(),
                PATH.SPECFEM_BIN +'/'+ 'xsmooth_sem '
                + str(span) + ' '
                + str(span) + ' '
                + name + '_kernel' + ' '
                + input_path + '/ '
                + output_path + '/ '
                + GPU_mode,
                output='/dev/null')
            filename1 = output_path + '/' + 'proc000000_' + name + '_kernel_smooth.bin'
            filename2 = output_path + '/' + 'proc000000_' + name + '_kernel.bin'
            os.system("mv %s %s" % (filename1,filename2))

        t2 = time.time()
        print('smooth done. t = ' ,t2-t1)

    ### file transfer utilities

    def import_model(self, path):
        src = glob(path +'/'+ 'model/*')
        dst = join(self.cwd, 'DATA/')
        unix.cp(src, dst)

    def export_model(self, path):
        unix.mkdir(path)
        src = glob(join(self.cwd, 'DATA/*.bin'))
        dst = path
        unix.cp(src, dst)


    @property
    def data_filenames(self):
        if PAR.CHANNELS:
            if PAR.FORMAT in ['SU', 'su']:
               filenames = []
               for channel in PAR.CHANNELS:
                   #filenames += ['U%s_file_single.su' % channel]
                   filenames += ['U%s_file_single_d.su' % channel]
               return filenames

        else:
            unix.cd(self.cwd)
            unix.cd('traces/obs')

            if PAR.FORMAT in ['SU', 'su']:
                #return glob('U?_file_single.su')
                return glob('U?_file_single_d.su')

    @property
    def adj_filenames(self):
        if PAR.CHANNELS:
            if PAR.FORMAT in ['SU', 'su']:
               filenames = []
               for channel in PAR.CHANNELS:
                   filenames += ['U%s_file_single.su.adj' % channel]
               return filenames

        else:
            unix.cd(self.cwd)
            unix.cd('traces/obs')

            if PAR.FORMAT in ['SU', 'su']:
                return glob('U?_file_single.su.adj')

    @property
    def model_databases(self):
        return join(self.cwd, 'DATA')

    @property
    def kernel_databases(self):
        return join(self.cwd, 'OUTPUT_FILES')

    @property
    def source_prefix(self):
        return 'SOURCE'

    # workaround for older versions of SPECFEM2D,
    # which lacked a smoothing utility
    if not exists(PATH.SPECFEM_BIN+'/'+'xsmooth_sem'):
        smooth = staticmethod(smooth_legacy)

