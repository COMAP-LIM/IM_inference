Code to do inference from line intensity maps

TODO:
- Implement functions to calculate covariance matrices
- Tests, and automatic tests (pytest?)
- Make "COMAP Light" experiment setup (same as COMAP only smaller redshift range for faster runtime)
- Include additional model output (e.g. luminosity functions)
- Automatically run on same input data as previous run
- Automatically run on same experiment and mcmc setup as previous run
- Continue from state of previous run
- Lots of ideas for different models to compare

In progress: 
- Make run_mcmc save results and parameters used together with 
log file
      - save experiment params as well
      - save exact command used when running current script in log file (e.g. python run_mcmc.py experiment_params_comap_lignt.py mcmc_params_full_parallell.py)
- Parallellization (fix issue with big halo objects)
- Implement cosmology (angles, redshift, Mpc etc.) (Check and fix this)

Done:
- Sampling initial points from priors
- Create function to automatically generate mock data
- Add instrumental noise
- Log file could summirize results (e.g. theta = 2.3 +- 0.21)
- Let run_mcmc be run with parameter files as command line arguments instead to use them instead of the default ones
- Add instrumental beam
