# Fine-tune scGPT for scRNA-seq integration

Please see our example code in [tutorials/scgpt_finetune.ipynb](tutorials/scgpt_finetune.ipynb). By default, the script assumes the scGPT checkpoint folder stored in the `examples/save` directory.

## How to install

```bash
module load Anaconda3/2024.02-1 CUDA/11.7.0 cuDNN/8.5.0.96-CUDA-11.7.0

git clone https://github.com/bowang-lab/scGPT.git
cd path_to/scGPT

conda env create -f docs/environment.yml
conda activate scgpt_finetune

pip install -r requirements.txt
pip install scgpt "flash-attn<1.0.5"
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
```
