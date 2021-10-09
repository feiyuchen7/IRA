# Interclass-Relativity-Adaptive Metric Learning for Cross-Modal Matching and Beyond
Pytorch implementation for the paper:
[Interclass-Relativity-Adaptive Metric Learning for Cross-Modal Matching and Beyond](https://ieeexplore.ieee.org/document/9178501), IEEE Transactions on Multimedia.

We provide the source codes built on [SCAN](https://github.com/kuanghuei/SCAN). IRA loss can likewise be evaluated on other benchmarks.

## Environment Requirement
The code has been tested running under Python 3.5.2. The required packages are as follows:
* Pytorch 1.4.0
* numpy 1.15.4

## Data
Download data following the instructions [here](https://github.com/kuanghuei/SCAN#download-data).

## Training
* MS-COCO dataset
`python train.py --data_path ./data --data_name coco_precomp --vocab_path ./vocab --logger_name "$LOG_PATH" --model_name "$SAVE_PATH" --max_violation --bi_gru --margin 0.1 --alpha 4 --beta 1`

* Flickr30K dataset
`python train.py --data_path ./data --data_name f30k_precomp --vocab_path ./vocab --logger_name "$LOG_PATH" --model_name "$SAVE_PATH" --max_violation --bi_gru --margin 0.1 --alpha 3 --beta 1`

## Evaluation
* MS-COCO dataset
```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$SAVE_PATH/model_best.pth.tar", data_path="$DATA_PATH", split="testall", fold5=True)
```

* Flickr30K dataset
```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$SAVE_PATH/model_best.pth.tar", data_path="$DATA_PATH", split="test")
```

##Reference

If you found this code useful, please cite the following paper:
```
@article{DBLP:journals/tmm/ChenSZXS21,
  author    = {Feiyu Chen and Jie Shao and Yonghui Zhang and Xing Xu and Heng Tao Shen},
  title     = {Interclass-Relativity-Adaptive Metric Learning for Cross-Modal Matching and Beyond},
  journal   = {{IEEE} Trans. Multim.},
  volume    = {23},
  pages     = {3073--3084},
  year      = {2021},
}
```
