Code to do inference from line intensity maps

TODO:
- Implement cosmology (angles, redshift, Mpc etc.)
- Implement functions to calculate covariance matrices
- Tests, and automatic tests (pytest?)
- Parallellization
- Make "COMAP Light" experiment setup (same as COMAP only smaller redshift range for faster runtime)
- Let run_mcmc be run with parameter files as command line arguments instead to use them instead of the default ones
- Include additional model output (e.g. luminosity functions)
- Automatically run on same input data as previous run
- Automatically run on same experiment and mcmc setup as previous run

- Lots of ideas for different models to compare

In progress: 
- Make run_mcmc save results and parameters used together with 
log file
      - save experiment params as well
      - save exact command used when running current script in log file (e.g. python run_mcmc.py experiment_params_comap_lignt.py mcmc_params_full_parallell.py)
      - Log file could summirize results (e.g. theta = 2.3 +- 0.21)


Done:
- Sampling initial points from priors
- Create function to automatically generate mock data
- Add instrumental noise
- Add instrumental beam
