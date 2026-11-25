# scMetaRoute

scMetaRoute constructs metacells from single-cell RNA-seq counts and retains their relationships as a hierarchical tree and a sparse weighted graph. It combines a negative-binomial VAE, Ward aggregation, and a second training phase guided by tree-smoothed metacell centers.

## Quick start

After [installation](#installation), run from the repository root:

```bash
python examples/run_demo.py
```

This runs preprocessing, training and downstream analysis on the included `data/sle_5k.h5ad`, saving results to `runs/sle_demo/`. The input contains 5,000 cells × 30,172 genes from 20 SLE/control donors; the workflow produces 100 metacells using 2,000 selected genes.

To reproduce the GPU example, explicitly select the device and a separate output directory:

```bash
python examples/run_demo.py --device cuda:0 --output runs/sle_demo_cuda
```

`run_demo.py` defaults to CPU and `runs/sle_demo/`; both flags can be omitted for the CPU example. Choose a new `--output DIRECTORY` when repeating a run.

## Installation

Open a terminal in the repository root and create the environment:

```bash
conda create -n scmetaroute python=3.11 pip -y
conda activate scmetaroute
```

Install **one** PyTorch build. For a CPU-only installation:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

Alternatively, for an NVIDIA GPU compatible with CUDA 12.8 wheels:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

For other hardware, choose a compatible build from the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v280). Then install the dependencies and package:

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

The example runs entirely in Python. R/edgeR is used only by the separate islet analysis.

## Reproduction settings and results

`run_demo.py` supplies the SLE settings internally: seed 1101, 2,000 selected genes, 150 microclusters, 100 metacells, batch size 512 and 160 + 40 epochs. You only need to pass `--device` or `--output` when changing them. The general `scmetaroute fit` command has different defaults: 750 microclusters, 500 metacells and an automatic batch size. When calling it directly, reproduce the example with `--microcells 150 --metacells 100 --batch-size 512`; see the [step-by-step commands](docs/reproduction.md#bundled-sle-example).

| Metric | CPU | RTX 3090 |
| --- | ---: | ---: |
| Purity | 0.874656 | 0.878211 |
| Compactness | 0.587019 | 0.592928 |
| Separation | 0.326966 | 0.322501 |
| Downstream clustering NMI | 0.718957 | 0.735890 |
| Donor-held-out annotation Macro F1 | 0.877896 | 0.887337 |

The complete workflow took about 5.8 minutes on CPU and 28 seconds on RTX 3090 in the tested environment. Full result tables are in [examples/expected_results](examples/expected_results/). The example produces five donor-level disease comparisons; significance-dependent disease ranking is marked inapplicable because no reference genes pass BH < 0.05 in this small subset.

## Outputs

Under `runs/sle_demo/` (or your `--output` directory):

| File | Contents |
| --- | --- |
| `model/METACELL_EXPRESSION.h5ad` | 100 × 2,000 metacell expression, raw count layers, latent centers and sparse relations. |
| `model/CELL_TO_METACELL.tsv` | Cell-to-metacell assignments for all 5,000 input cells. |
| `model/STAGE2_METACELL_WARD_LINKAGE.npy` | Hierarchy over final metacells. |
| `model/STAGE2_TREE_GRAPH.npz` | 100 × 100 weighted metacell relation matrix. |
| `quality/METRICS_RAW.tsv` | Ten metacell quality and expression-concordance metrics. |
| `differential_expression/DE_SUMMARY.tsv` | Donor-level disease-effect comparisons. |
| `downstream/DOWNSTREAM_METRICS.tsv` | Neighborhood, clustering, co-expression and annotation metrics. |

## Other datasets

Only `sle_5k.h5ad` is bundled. Download other datasets separately from the links below. The H5AD links identify the exact atlas versions used locally rather than the latest mutable collection release.

| Dataset | Collection / study source | Versioned H5AD |
| --- | --- | --- |
| ACR_CLAD | [CELLxGENE](https://cellxgene.cziscience.com/collections/e276e3e2-197a-4524-abd1-a753a48dc33a), [GSE274199](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274199) | [Download](https://datasets.cellxgene.cziscience.com/af6e81be-e65c-4821-987e-e0eb6c8acd59.h5ad) |
| Adult_Human_Kidney | [CELLxGENE](https://cellxgene.cziscience.com/collections/9c9d04c4-8899-417f-bb6f-6107dcadf14f), [KPMP](https://doi.org/10.48698/16dd-vj20) | [Download](https://datasets.cellxgene.cziscience.com/255ce81a-eb4e-48ff-9f18-e50455ccb24d.h5ad) |
| Alzheimer_PD | [CELLxGENE](https://cellxgene.cziscience.com/collections/8b35aa1f-6bcf-4a51-abc3-a3f336a44ae6) | [Download](https://datasets.cellxgene.cziscience.com/060f346b-c744-4845-9191-65595dd893b9.h5ad) |
| COVID | [CELLxGENE](https://cellxgene.cziscience.com/collections/8f126edf-5405-4731-8374-b5ce11f53e82), [COMBAT](https://doi.org/10.5281/zenodo.6120249) | [Download](https://datasets.cellxgene.cziscience.com/687c09ff-731a-4e3d-ac07-4c29c33a6338.h5ad) |
| IBD_Crohn | [CELLxGENE](https://cellxgene.cziscience.com/collections/7f7fdf50-aa0e-4a6f-979c-53655cac9d5d), [E-MTAB-16986](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-16986) | [Download](https://datasets.cellxgene.cziscience.com/e3460364-c300-4616-9923-3a0224894998.h5ad) |
| SLE | [CELLxGENE](https://cellxgene.cziscience.com/collections/436154da-bcf1-4130-9c8b-120ff9a888f2), [GSE137029](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137029) | [Download](https://datasets.cellxgene.cziscience.com/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad) |
| human_pancreatic | [CELLxGENE](https://cellxgene.cziscience.com/collections/58e85c2f-d52e-4c19-8393-b854b84d516e), [GSE221156](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221156) | [Download](https://datasets.cellxgene.cziscience.com/3a4e48e2-9e80-4ea8-ac63-87fdf70f1a97.h5ad) |
| mouse_pancreatic | [CELLxGENE](https://cellxgene.cziscience.com/collections/296237e2-393d-4e31-b590-b03f74ac5070), [GSE211799](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211799) | [Download](https://datasets.cellxgene.cziscience.com/2cf0b367-e31b-4ab3-8557-8f2c4ac2a31a.h5ad) |

The islet stimulus reference is available at [GSE237448](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE237448).

See [docs/reproduction.md](docs/reproduction.md) for individual downstream commands, other-dataset usage, paper settings and islet analysis. Data provenance and licensing are in [data/README.md](data/README.md).

## Attribution

This repository is the official implementation of our paper:

> **scMetaRoute: Hierarchical metacell representation of single-cell atlases**
> JiaNan Li, JiaHao Liu, XiaoJiang Yu, Fan Yang, Feng Zeng
> *Preprint, 2026*

If you find this repository helpful, please cite:

```bibtex
@misc{li2026scmetaroute,
  title={scMetaRoute: Hierarchical metacell representation of single-cell atlases},
  author={Li, JiaNan and Liu, JiaHao and Yu, XiaoJiang and Yang, Fan and Zeng, Feng},
  year={2026},
  note={Preprint},
  url={https://github.com/thigiangh471-tech/scMetaRoute}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
