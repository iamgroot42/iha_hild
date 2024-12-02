# auditing_mi

Code for our paper [Do Parameters Reveal More than Loss for Membership Inference?](https://arxiv.org/pdf/2406.11544) at TMLR.

## Instructions

First install python dependencies
```
pip install -r requirements.txt
```

Then, install the package

```
pip install -e .
```

## Setting environment variables

You can either provide the following environment variables, or pass them via your config/CLI:

```
MIB_DATA_SOURCE: Path to data directory
MIB_CACHE_SOURCE: Path to save models, signals, and paths.
```

## Training models

Use `mib/train.py` to train models. Arguments related to training are specified in the file itself.
For example, to train models on Purchase-100 for MLP2 architecture, run:

```bash
python mib/train.py --dataset purchase100 --model_arch mlp2
```

## Running attacks

Use `mib/attack.py` to run attacks. Arguments related to attacks are specified in the file itself.
For example, to generate attack signals for Purchase-100 for MLP2 for IHA, run:

```bash
python mib/attack.py --dataset purchase100 --model_arch mlp2 --attack ProperTheoryRef --num_points -1
```

## Citation

```bibtex
@article{suri2024do,
  title     = {Do Parameters Reveal More than Loss for Membership Inference?},
  author    = {Anshuman Suri and Xiao Zhang and David Evans},
  journal   = {Transactions on Machine Learning Research (TMLR)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.11544}
}
```
