# Fine-tune scGPT for scRNA-seq integration

Please see our example code in [tutorials/scgpt_finetune.ipynb](tutorials/scgpt_finetune.ipynb). By default, the script assumes the scGPT checkpoint folder stored in the `save` directory.

## How to install

```bash


git clone https://github.com/bowang-lab/scGPT.git
cd path_to/scGPT

conda env create -f docs/environment_cu124.yml
conda activate scgpt_cu124

pip install -r docs/requirements_1_cu124.txt
pip install -r docs/requirements_2_cu124.txt
pip install -e .
pip install flash_attn
pip install wandb

python -m ipykernel install \
  --prefix="/scratch/group/p.nairr250244.000/.conda/envs/scgpt_finetune" \
  --name scgpt_finetune \
  --display-name "Python (scgpt_finetune)"
```

## Acknowledgements

We sincerely thank the authors of following open-source projects:

- [flash-attention](https://github.com/HazyResearch/flash-attention)
- [scanpy](https://github.com/scverse/scanpy)
- [scvi-tools](https://github.com/scverse/scvi-tools)
- [scib](https://github.com/theislab/scib)
- [datasets](https://github.com/huggingface/datasets)
- [transformers](https://github.com/huggingface/transformers)

## Citing scGPT

```bibtex
@article{cui2023scGPT,
title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}

python -m ipykernel install --prefix=$CONDA_PREFIX --name scgpt_finetune --display-name "Python (scgpt_finetune)"