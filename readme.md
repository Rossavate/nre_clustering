## Requirments

The model is implemented using tensorflow, The versions of packages used in my experiment are shown below.

* tensorflow = 1.12.1
* numpy = 1.15.4
* scikit-learn = 0.20.0
* matplotlib = 3.0.3

## preprocess data

NYTdataset can be downloaded from [here](https://pan.baidu.com/s/1uMVEncWKx3UDL_aCOjOgkg), extract code:meir , the origin data should be place in the directory data/origin_data/

model main codes in the directory `src/bag-level/`, run

```
python init_data.py
```

to generate processed data

## Train the model

running script in the directory `bin/`

use command:  `bash run.bash`  to train model

you can edit the .bash file to decide epoch number, pattern number, and whether use gpu and entity type.

our model can test in the training process, and the best model and test result will save in the directory `runs/bag/`

## draw pr figures

firstly go to the srcipts/

run command `python get_sparse_pr.py` to generate data that used to draw. 

then run 

`python bag_plot_tpr.py` to draw the pr figure of baselines and our model.

run `python bag_plot_mpr.py` to draw pr figure of different cluster numbers.

## baseline

CNN+AVE

CNN+ATT

PCNN+AVE

PCNN+ATT

APCNN+soft-label:  EMNLP2017 A Soft-label Method for Noise-tolerant Distantly Supervised Relation Extraction

CNN+RL :  AAAI2018 Reinforcement Learning for Relation Classification from Noisy Data



