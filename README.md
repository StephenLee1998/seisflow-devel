Add new functions to SeisFlows

1. ./seisflows/plugins/adjoint.py 
======================================================	
Envelope, avoid divide zero when mute near-offset data.
======================================================	
2. ./seisflows/tools/signal.py
   ./seisflows/preprocess/base.py
======================================================	
	Apply tapered mask to shot gather, muting body waves and later waves to preserve surface wave.
======================================================	
3. ./seisflows/workflow/migration.py
   ./seisflows/solver/base.py
======================================================	
	Fixed some bugs in RTM. The original code failed to get correct adjoint source
======================================================	
4.  ./seisflows/solver/base.py
======================================================	
	Allow to use different STATION file for different SOURCE. The naming rule must be same. The station name must be STATIONS_******.
	Add a new parameter 'PAR.USER_DEFINE_STATION' to control this feature. Remember to set use_existing_STATIONS=.true. when you use this option. 
	You can ignore 'PAR.USER_DEFINE_STATION' if you use a fixed receiver array in synthetic test. This fixed receiver array can be define either by STATIONS file under specfem2d-master/DATA or parameters in Par_file.
======================================================	


