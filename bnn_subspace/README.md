## Setup
Code to reproduce the BNN subspace experiments from the paper. 

Data and model checkpoints are stored in the `/data` directory, which you may need to create on your system. To use our pretrained checkpoints for the CNN-LSTM and FRN-20 models, you should download them from [this google drive link](https://drive.google.com/drive/folders/1me76YH46yGui8pxOfx6qo6Y85ym0P74y?usp=sharing) and locate the folders `cnn_lstm_checkpoints` and `frn_checkpoints` under the `/data` directory. The checkpoints occupy ~90MB. With these checkpoints downloaded, you can run any stage of our pretrain-->train curve-->sampler baseline-->sample pipeline. If not, your will need to start from the beginning and train your own checkpoints.

## Reproduction

This assumes you have installed the prerequisites using the 'setup.sh' script which is in the root directory. Below we detail the pipeline stages.  

To run HPO on the base model training:
```
bash scripts/base_hpo_imdb.sh
bash scripts/base_hpo_cifar10.sh
```
Then you can update the base training script with the best config you find. We have pre-set parameters based on the results of our HPO.
```
bash scripts/base_train_imdb.sh
bash scripts/base_train_cifar10.sh
```

To run HPO on the curve model training:
```
bash scripts/curve_hpo_imdb.sh
bash scripts/curve_hpo_cifar10.sh
```
To run the curve model training:
```
bash scripts/curve_train_imdb.sh
bash scripts/curve_train_cifar10.sh
```
To run the baseline MALA sampler HPO:
```
bash scripts/sampler_hpo_imdb.sh
bash scripts/sampler_hpo_cifar10.sh
```
To run the baseline MALA sampler for 20000 steps:
```
bash scripts/sampler_baseline_imdb.sh
bash scripts/sampler_baseline_cifar10.sh
```

To run the final sampler comparisons:
```
bash scripts/paper_run_imdb_imdb.sh
bash scripts/paper_run_cifar10.sh
```

To plot the results
```
bash scripts/plot.sh --show
```
Running without the `--show` flag will save the results in `../plots`.
