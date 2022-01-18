Code to reproduce sensitivity studies based on varying either model order growth paramter alpha or the number of modes.

The scripts are run by
```
bash scripts/run_alpha_sensitivity.sh
bash scripts/run_mode_sensitivity.sh
```
and you can plot the results with 
```
bash scripts/plot_all.sh
```
This will save results to the `../plots` folder, but if you use the `--show` flag this will pop out plots on your screen.
```
bash scripts/plot_all.sh --show
```
