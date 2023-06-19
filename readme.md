# GraphGLOW in PyTorch

PyTorch implementation of *GraphGLOW: Universal and Generalizable Structure Learning for Graph Neural Networks* [1].

## Introduction

In this work, we attempt to learn a generalizable graph structure learning model which is trained with multiple source graphs and can be directly adapted for inference (without re-training or fine-tuning) on new unseen target graphs. As displayed in the following figure, we achieve this goal through jointly learning a single dataset-shared structure learner and multiple dataset-specific GNNs.

<img src="/Users/zhaowentao/university/实验室/2021_summer/中稿/2023kdd open source/GraphGLOW_open_source/data flow.jpg" style="zoom:80%;" />

## Environment Set Up
Install the required packages according to ```requirements.txt```.

Note that there is a change of the interface of higher version of *PyYAML* package, so we recommend you to install this package with the exact version number specified in ```requirements.txt```.

Most of the experiments in paper [1] were conducted on an NVIDIA GeForce RTX 2080 Ti with 11GB memory. For experiments involving two larger datasets, PubMed and Cornell5, we utilized an NVIDIA GeForce RTX 3090 with 24 GB memory.

## Usage
First run preprocessing code to preprocess four Facebook100 datasets.
```
cd data
python prep_fb100.py --dataset amherst
python prep_fb100.py --dataset jh
python prep_fb100.py --dataset reed
python prep_fb100.py --dataset cornell
```
Next, switch to the source directory using
```
cd ../src
```

To train a structure learner with CiteSeer and Cora as source graphs and transfer it to Amherst41, run the following code:
```
python main.py --config config/transfer/citeseer_cora_trans_amherst.yml 
```

To run other experiments, just replace the .yml file with other files in config/transfer.

## References
[1] Wentao Zhao, Qitian Wu, Chenxiao Yang, and Junchi Yan. 2023. GraphGLOW: Universal and Generalizable Structure Learning for Graph Neural Networks. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23).

## Citation
If you find our code useful, please consider citing our work
```
@inproceedings{zhao2023glow,
  title={GLOW: Universal and Generalizable Structure Learning for Graph Neural Networks},
  author={Zhao, Wentao and Wu, Qitian and Yang, Chenxiao and Yan, Junchi}
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```