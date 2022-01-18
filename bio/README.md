Code to run several MCMC online thinning methods (ours and SPMCMC) on data from the paper [Stein Thinning](http://stein-thinning.org/)

To download the goodwin oscillator and calcium signaling model samples and grads use
```
bash scripts/get_data.sh
```

If you want to run a smoke test for the calcium signaling model you will need to run 
```
bash scripts/create_cardiac_smoke_test_data.sh
```
which creates a subset of the MCMC data.

The goodwin oscillator does not require separate smoke test data.

To run the cardiac or goodwin experiments use 
```
bash scripts/run_cardiac.sh
bash scripts/run_goodwin.sh
```

To plot the results use
```
bash scripts/plot_all.sh --show
```
If you don't use the `--show` flag, the figures will be saved in `../plots`.
