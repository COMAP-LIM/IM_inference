Code to do inference from line intensity maps

TODO:
- Test parallelization with openmp for different realizations
- Implement map_object saved with header (like real observed temperature map)
- Tests, and automatic tests (pytest?)
- Automatically run on same input data as previous run
- Automatically run on same experiment and mcmc setup as previous run
- Continue from state of previous run
- Lots of ideas for different models to compare

In progress: 
- Make run_mcmc save results and parameters used together with 
- Parallellization (fix issue with big halo objects)

Done:
- Save imput data in log
- Sampling initial points from priors
- Create function to automatically generate mock data
- Add instrumental noise
- Log file could summirize results (e.g. theta = 2.3 +- 0.21)
- Let run_mcmc be run with parameter files as command line arguments instead to use them instead of the default ones
- Add instrumental beam
- Implement cosmology (angles, redshift, Mpc etc.) (Check and fix this)
      - Decide on consitent way of defining, dx, dy, dz, v_vox and v_full
- Include additional model output (e.g. luminosity functions)
- log file
      - save experiment params as well
      - save exact command used when running current script in log file (e.g. python run_mcmc.py experiment_params_comap_lignt.py mcmc_params_full_parallell.py) (not easy to do)
- Implement functions to calculate covariance matrices
