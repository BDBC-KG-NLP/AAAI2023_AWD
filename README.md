# AAAI2023_AWD
code and data for AAAI2023 paper "Adversarial Word Dilution as Text Data Augmentation in Low-Resource Regime"

## Prepare dataset

A sample script to prepare a dataset:

```
python datasets/create_fsl_dataset.py -datadir datasets/stsa -num_train 10 -num_dev 10 -lower
```

## Sample script

run.sh: A sample bash script to run on STSA dataset, remember to change the path to root directory.
