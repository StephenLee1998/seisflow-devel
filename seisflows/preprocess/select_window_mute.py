
import sys
import numpy as np
import obspy

from seisflows.tools import msg, unix
from seisflows.tools.tools import exists, getset

from seisflows.plugins import adjoint, misfit, readers, writers
from seisflows.tools import signal
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class select_window_mute(custom_import('preprocess','base')):
    """ select window strategy data preprocessing class
	  Step1: split the observed data and synthetic data to different windows
	  Step2: compare cross-correlation value, amplitude ratio, and traveltime difference to decide
	         if we will use data in this window
	  Step3: Sum the adjoint source.

      Provides data processing functions for multi-bandpass filter strategy
    """

    def check(self):
        """ Checks parameters and paths
        """
        # used for inversion
        if 'WINDOW_NUMBER' not in PAR:
            raise ParameterError(PAR, 'WINDOW_NUMBER')
        if 'WINDOW_LENGTH' not in PAR:
            raise ParameterError(PAR, 'WINDOW_LENGTH')
        if 'TSHIFT_ACCEPTANCE_LEVEL' not in PAR:
            raise ParameterError(PAR, 'TSHIFT_ACCEPTANCE_LEVEL')
        if 'DLNA_ACCEPTANCE_LEVEL' not in PAR:
            raise ParameterError(PAR, 'DLNA_ACCEPTANCE_LEVEL')
        if 'CC_ACCEPTANCE_LEVEL' not in PAR:
            raise ParameterError(PAR, 'CC_ACCEPTANCE_LEVEL')

        # data filtering option
        if 'FILTER' not in PAR:
            setattr(PAR, 'FILTER', None)

############################################## original #######################################
#    def prepare_eval_grad(self, path='.'):
#        """
#         Prepares solver for gradient evaluation by writing residuals and
#         adjoint traces
#
#         :input path: directory containing observed and synthetic seismic data
#        """
#        solver = sys.modules['seisflows_solver']
#
#        icount_window_number=0
#        icount_total_window_number=0
#        #print('==========================================')
#        for iwindow in range(PAR.WINDOW_NUMBER):
#            unix.mkdir(path+'/'+'traces/adj'+'%d'%iwindow)
#            #print('iwindow', iwindow)
#            for filename in solver.data_filenames:
#                obs = self.reader(path+'/'+'traces/obs', filename)
#                syn = self.reader(path+'/'+'traces/syn', filename)
#
#                # process observations
#                obs = self.apply_filter(obs)
#                obs = self.apply_mute_select_window(obs,iwindow,PAR.WINDOW_LENGTH)
#
#                # process synthetics
#                syn = self.apply_filter(syn)
#                syn = self.apply_mute_select_window(syn,iwindow,PAR.WINDOW_LENGTH)
#
#                ############# compare obs and syn to reject window ##############
#                nr, _ = self.get_network_size(syn)
#                for ir in range(nr):
#                    max_cc_value,cc_shift,dlnA=self._calc_criteria(obs[ir].data[:],syn[ir].data[:])
#                    #if ir==0:
#                    #    print('max_cc_value, cc_acceptance_level', max_cc_value, PAR.CC_ACCEPTANCE_LEVEL)
#                    #    print('cc_shift, tshift_acceptance_level', cc_shift, PAR.TSHIFT_ACCEPTANCE_LEVEL)
#                    #    print('dlnA, dlna_acceptance_level', dlnA, PAR.DLNA_ACCEPTANCE_LEVEL)
#                    #    print('==========================================')
#                    #if (cc_shift>=1.0/(PAR.F0*PAR.DT*2) or dlnA>=2 or max_cc_value<=0.80):
#                    # windows - reject
#                    if (abs(cc_shift)>=PAR.TSHIFT_ACCEPTANCE_LEVEL or dlnA>=PAR.DLNA_ACCEPTANCE_LEVEL or max_cc_value<=PAR.CC_ACCEPTANCE_LEVEL):
#                        obs[ir].data[:]=0.0
#                        syn[ir].data[:]=0.0
#                    # calculate how many windows in the inversion
#                    if (abs(dlnA)>0.0):
#                        icount_total_window_number=icount_total_window_number+1
#                    # calculate how many windows are used in the inversion
#                    if (abs(cc_shift)<PAR.TSHIFT_ACCEPTANCE_LEVEL and dlnA<PAR.DLNA_ACCEPTANCE_LEVEL and max_cc_value>PAR.CC_ACCEPTANCE_LEVEL):
#                        icount_window_number=icount_window_number+1
#                ############# compare obs and syn to reject window ##############
#                self.write_adjoint_traces(path+'/'+'traces/adj'+'%d'%iwindow, syn, obs, filename)
#        print('window used in the inversion, total window number', icount_window_number, icount_total_window_number)
#        #print('==========================================')
#
#        if PAR.MISFIT:
#            obs = self.reader(path+'/'+'traces/obs', filename)
#            syn = self.reader(path+'/'+'traces/syn', filename)
#            # process observations to calculate misfit
#            obs = self.apply_filter(obs)
#            obs = self.apply_mute(obs)
#            obs = self.apply_normalize(obs)
#            # process synthetics to calculate misfit
#            syn = self.apply_filter(syn)
#            syn = self.apply_mute(syn)
#            syn = self.apply_normalize(syn)
#
#            self.write_residuals(path, syn, obs)
#
#        adj_sum = syn
#        nr, _ = self.get_network_size(syn)
#        for ir in range(nr):
#            adj_sum[ir].data[:] = 0.0
#
#        for iwindow in range(PAR.WINDOW_NUMBER):
#            #print 'SUMMATION, iwindow='
#            #print(iwindow)
#            for filename in solver.data_filenames:
#            #for filename in self.adj_filenames:
#                adj = self.reader(path+'/'+'traces/adj'+'%d'%iwindow, filename)
#                for ir in range(nr):
#                    adj_sum[ir].data = adj_sum[ir].data + adj[ir].data
#
#        # mute adj_sum if necessary
#        adj_sum = self.apply_mute(adj_sum)
#        adj_sum = self.apply_normalize(adj_sum)
#
#        # output adj_sum
#        for filename in solver.data_filenames:
#            self.writer(adj_sum, path+'/'+'traces/adj', filename)
#        #print 'Output adj_sum_end'
############################################## original #######################################

############################################# jiang change to speed up #######################################
    def prepare_eval_grad(self, path='.'):
        """
         Prepares solver for gradient evaluation by writing residuals and
         adjoint traces

         :input path: directory containing observed and synthetic seismic data
        """
        solver = sys.modules['seisflows_solver']

        icount_window_number=0
        icount_total_window_number=0
        #print('==================filter start========================')
        unix.mkdir(path+'/'+'traces/obs_filter')
        unix.mkdir(path+'/'+'traces/syn_filter')

        #===
        # !!!Warning : Observation data and synthetic dara (SU) has different
        # headers , need to fix in future.
        #   YYR Dec 26, 2021
        #===
        # process observations and synthetics
        for filename in solver.data_filenames:
            # porcess observations
            obs = self.reader(path+'/'+'traces/obs', filename)
            obs = self.apply_filter(obs)
            obs = self.apply_mute(obs)
            obs = self.apply_mute_nearoffset(obs)
            obs = self.apply_normalize(obs)
            self.writer(obs, path+'/'+'traces/obs_filter', filename)

            # process synthetics
            syn = self.reader(path+'/'+'traces/syn', filename)
            syn = self.apply_filter(syn)
            syn = self.apply_mute(syn)
            syn = self.apply_mute_nearoffset(syn)
            syn = self.apply_normalize(syn)
            self.writer(syn, path+'/'+'traces/syn_filter', filename)

        #print('==================windowing========================')

        total_misfit = 0.0

    
        for filename in solver.data_filenames:
            obs_original = self.reader(path+'/'+'traces/obs_filter', filename)
            syn_original = self.reader(path+'/'+'traces/syn_filter', filename)

            nt, dt, _ = self.get_time_scheme(syn)
            nr, _ = self.get_network_size(syn)

            trace_residual = np.zeros(nr)
            for iwindow in range(PAR.WINDOW_NUMBER):
                unix.mkdir(path+'/'+'traces/adj'+'%d'%iwindow)
                
                # operate on a copy for each window
                obs = obs_original.copy()
                syn = syn_original.copy()

                # process observations
                [obs,const1,const2] = self.apply_mute_select_window(obs,iwindow,PAR.WINDOW_LENGTH)
                #obs = self.apply_mute_nearoffset(obs)
                
                # process synthetics
                [syn,const1,const2] = self.apply_mute_select_window(syn,iwindow,PAR.WINDOW_LENGTH)
                #syn = self.apply_mute_nearoffset(syn)

                # YYR cut window for c.c.
                const1 = const1.astype('int')
                const2 = (const2-1).astype('int')
                #print(const1,const2)

                # residuals (misfit) for unmuted part of tracce
                #trace_residual += self.window_residuals(syn, obs)

                ############# compare obs and syn to reject window ##############
                nr, _ = self.get_network_size(syn)        
                for ir in range(nr):
                    max_cc_value,cc_shift,dlnA=self._calc_criteria(obs[ir].data[const1:const2],syn[ir].data[const1:const2])
        
                    # debug:
                    #if ir==0:
                    #    print('max_cc_value, cc_acceptance_level', max_cc_value, PAR.CC_ACCEPTANCE_LEVEL)
                    #    print('cc_shift, tshift_acceptance_level', cc_shift, PAR.TSHIFT_ACCEPTANCE_LEVEL)
                    #    print('dlnA, dlna_acceptance_level', dlnA, PAR.DLNA_ACCEPTANCE_LEVEL)
                    #    print('==========================================')
                    #if (cc_shift>=1.0/(PAR.F0*PAR.DT*2) or dlnA>=2 or max_cc_value<=0.80):

                    # windows - reject
                    if (abs(cc_shift)>=PAR.TSHIFT_ACCEPTANCE_LEVEL or dlnA>=PAR.DLNA_ACCEPTANCE_LEVEL or max_cc_value<=PAR.CC_ACCEPTANCE_LEVEL):
                        obs[ir].data[:]=0.0
                        syn[ir].data[:]=0.0
        
                    # calculate how many windows in the inversion
                    if (abs(dlnA)>0.0):
                        icount_total_window_number += 1
        
                    # calculate how many windows are used in the inversion
                    if (abs(cc_shift)<PAR.TSHIFT_ACCEPTANCE_LEVEL and dlnA<PAR.DLNA_ACCEPTANCE_LEVEL and max_cc_value>PAR.CC_ACCEPTANCE_LEVEL):
                        icount_window_number=icount_window_number+1

                trace_residual += self.window_residuals(syn, obs)        
                ############# compare obs and syn to reject window ##############
                self.write_adjoint_traces(path+'/'+'traces/adj'+'%d'%iwindow, syn, obs, filename)
        
            # adds up misfits of all traces
            total_misfit += np.sum(trace_residual)
        # normalize the misfit by window conts to make the inversion stable
        # more windows are selected after each interation , have to keep the misfit normalized so
        # the total misfit won't stop the inversion by the increase of windows
        #if icount_window_number > 0:
        #    total_misfit /= icount_window_number

        print(' %6d/%-6d window used in the inversion, total misfit: %10.4f ' % (icount_window_number, icount_total_window_number,total_misfit))

        if PAR.MISFIT:
            filename = path + '/' + 'residuals'
            # trace_residual / = icount_window_number
            np.savetxt(filename, trace_residual)
            #obs = self.reader(path+'/'+'traces/obs', filename)
            #syn = self.reader(path+'/'+'traces/syn', filename)
            ## process observations to calculate misfit
            #obs = self.apply_mute(obs)
            #obs = self.apply_mute_nearoffset(obs)
            #obs = self.apply_normalize(obs)
            ## process synthetics to calculate misfit
            #syn = self.apply_mute(syn)
            #syn = self.apply_mute_nearoffset(syn)
            #syn = self.apply_normalize(syn)
#
            #self.write_residuals(path, syn, obs)

        #===
        # Adjoint sources (windows) of each shot are normalized by the total windows 
        # in this shot . This means each window are equally treated within the shot,
        # so traces with more window or windows with larger waveform differences will
        # contribute more to the misfit . This weighting is balance the contribution
        # of shots no matter how many windows are selected.
        # IT IS VERY DANGEROUS when one shot have only a few windows selected !!!!!
        #                -- YYR Dec 25, 2021
        #===
        adj_sum = syn
        nr, _ = self.get_network_size(syn)
        for ir in range(nr):
            adj_sum[ir].data[:] = 0.0

        for iwindow in range(PAR.WINDOW_NUMBER):
            for filename in solver.data_filenames:
            #for filename in self.adj_filenames:
                adj = self.reader(path+'/'+'traces/adj'+'%d'%iwindow, filename)
                for ir in range(nr):
                    adj_sum[ir].data = adj_sum[ir].data + adj[ir].data

        # mute adj_sum if necessary
        adj_sum = self.apply_mute(adj_sum)
        adj_sum = self.apply_mute_nearoffset(adj_sum)
        
        # normalized the adjoint source will destroy the weighting applied !!!
        #adj_sum = self.apply_normalize(adj_sum)

        # output adj_sum
        #for filename in solver.data_filenames:
        for filename in solver.adj_filenames:
            #== YYR added for specfem2d-2021 b/c difference naming of adj and dat
            self.writer(adj_sum, path+'/'+'traces/adj', filename)



    @property
    def adj_filenames(self):
        if PAR.CHANNELS:
            if PAR.FORMAT in ['SU', 'su']:
               filenames = []
               for channel in PAR.CHANNELS:
                   filenames += ['U%s_file_single_d.su.adj' % channel]
               return filenames

        else:
            unix.cd(self.cwd)
            unix.cd('traces/obs')

            if PAR.FORMAT in ['SU', 'su']:
                return glob('U?_file_single_d.su.adj')

    def apply_mute_select_window(self, traces, iwindow, window_length):
        [traces,const1,const2] = self.mute_select_window(traces,
            iwindow, window_length,
            self.get_time_scheme(traces))

        return traces,const1,const2

    def mute_select_window(self, traces, iwindow, window_length, time_scheme):
        """ Applies tapered mask to record section, muting seismic waves outside the window
    
            Signals arriving before
    		    iwindow * window_length 	
            and after
    		    (iwindow+1) * window_length 	
    
            are muted, where SLOPE is has units of velocity**-1, 
            CONST has units of time, and
            || s - r || is distance between source and receiver.
        """
    
        nr = len(traces)
        nt, dt, _ = time_scheme
    
        # offset and slope are always zero
        offset = 0.0
        slope  = 0.0

        const1 = iwindow * window_length * dt 
        const2 = (iwindow+1) * window_length * dt
        
        con1 = np.floor(iwindow * window_length)
        con2 = np.ceil((iwindow+1) * window_length)
    
        for ir in range(nr):
    
            # apply tapered mask
            traces[ir].data *= self.mask_window(slope, const1, offset, time_scheme)
            traces[ir].data *= (1. - self.mask_window(slope, const2, offset, time_scheme))
    
        return traces,con1,con2

    ### functions acting on individual traces
    def mask_window(self, slope, const, offset, time_scheme, length=200):
        """ Constructs tapered mask that can be applied to trace to
          mute early or late arrivals.
        """
    
        nt, dt, _ = time_scheme
    
        mask = np.ones(nt)
    
        # construct taper
        win = np.sin(np.linspace(0, np.pi, 2*length))
        win = win[0:length]
    
        # caculate offsets
        itmin = int(np.ceil((slope*abs(offset)+const)/dt)) - length/2
        itmax = itmin + length
    
        if 1 < itmin < itmax < nt:
            mask[0:itmin] = 0.
            mask[itmin:itmax] = win*mask[itmin:itmax]
        elif itmin < 1 <= itmax:
            mask[0:itmax] = win[length-itmax:length]*mask[0:itmax]
        elif itmin < nt < itmax:
            mask[0:itmin] = 0.
            mask[itmin:nt] = win[0:nt-itmin]*mask[itmin:nt]
        elif itmin > nt:
            mask[:] = 0.
    
        return mask

    def _xcorr_win(self, d, s):
        cc = np.correlate(d, s, mode="full")
        time_shift = cc.argmax() - len(d) + 1
        # Normalized cross correlation.
        if np.sqrt((s ** 2).sum() * (d ** 2).sum()) > 0.0:
            max_cc_value = cc.max() / np.sqrt((s ** 2).sum() * (d ** 2).sum())
        else:
            max_cc_value = 1.0
        return max_cc_value, time_shift

    def _dlnA_win(self, d, s):
        if np.sum(s ** 2) > 0.0 and np.sum(d ** 2) > 0.0:
            return 0.5 * np.log(np.sum(d ** 2) / np.sum(s ** 2))
        else:
            return 0.0

    def _calc_criteria(self, d, s):
        max_cc_value, time_shift = self._xcorr_win(d, s)
        dlnA = self._dlnA_win(d, s)
        return max_cc_value, time_shift, dlnA

    def apply_mute_nearoffset(self, traces):
        if not PAR.MUTE_NEAROFFSET:
            return traces

        if 'MuteShortOffsets' in PAR.MUTE_NEAROFFSET:
            #print(''MuteShortOffsets)
            #print(PAR.MUTE_SHORT_OFFSETS_DIST)
            traces = signal.mute_short_offsets(traces,
                PAR.MUTE_SHORT_OFFSETS_DIST,
                self.get_source_coords(traces),
                self.get_receiver_coords(traces))

        if 'MuteLongOffsets' in PAR.MUTE_NEAROFFSET:
            #print(''MuteLongOffsets)
            #print(PAR.MUTE_LONG_OFFSETS_DIST)
            traces = signal.mute_long_offsets(traces,
                PAR.MUTE_LONG_OFFSETS_DIST,
                self.get_source_coords(traces),
                self.get_receiver_coords(traces))
                
        return traces



