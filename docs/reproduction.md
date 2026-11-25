# Paper reproduction

The `downstream_tasks/` directory contains single-run scorers and the islet statistical replay. The repository bundles one small SLE example under `data/`; the other datasets and historical checkpoints are obtained separately. A reproduction of the full comparison figures also needs the frozen inputs/assignments and the separately archived figure source data and styling. Run the commands below from the repository root.

## Bundled SLE example

The shortest command is `python examples/run_demo.py`. The following commands run the same workflow step by step in a separate directory:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scmetaroute prepare data/sle_5k.h5ad runs/sle_manual/panel
python downstream_tasks/export_metadata.py data/sle_5k.h5ad runs/sle_manual/metadata.tsv
scmetaroute fit runs/sle_manual/panel runs/sle_manual/model \
  --device cpu --microcells 150 --metacells 100 --batch-size 512 --save-checkpoint
python downstream_tasks/metacell_quality.py runs/sle_manual/panel \
  runs/sle_manual/model/CELL_TO_METACELL.tsv runs/sle_manual/metadata.tsv runs/sle_manual/quality
python downstream_tasks/differential_expression.py runs/sle_manual/panel \
  runs/sle_manual/model/CELL_TO_METACELL.tsv runs/sle_manual/metadata.tsv \
  runs/sle_manual/differential_expression --case 'systemic lupus erythematosus'
python downstream_tasks/downstream_metrics.py runs/sle_manual/panel \
  runs/sle_manual/model/CELL_TO_METACELL.tsv runs/sle_manual/metadata.tsv runs/sle_manual/downstream \
  --de-genes runs/sle_manual/differential_expression/DE_GENES.tsv.gz
```

The direct `fit` command needs the three size overrides shown above to match the example: `--microcells 150 --metacells 100 --batch-size 512`. The seed (1101), epochs (160 + 40), support floor (8) and preparation gene count (2,000) already match the CLI defaults. `--save-checkpoint` enables weight export. Replace `--device cpu` with `--device cuda:0` to reproduce the GPU example. The runner supplies all these settings internally; it accepts `--device` and `--output` only.

`--case` is required by the disease-DE script; the control defaults to `normal`. `--de-genes` is optional for downstream scoring and supplies the disease gene table when provided. Labels are read after training. Quality-script `Accuracy` is post-fit majority-label agreement, whereas cross-donor annotation holds donors out during scoring.

The example produces five eligible cell-type disease contrasts. No reference gene passes BH < 0.05 in this subset, so disease ranking is marked not applicable in `TASK_STATUS.tsv`; expression-effect comparisons remain available.

## Rebuild the SLE subset

The example is already included. To recreate it from the exact atlas version, download the source to the concrete path used below:

```bash
curl -L 'https://datasets.cellxgene.cziscience.com/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad' -o data/SLE.h5ad
python examples/create_demo_data.py data/SLE.h5ad runs/rebuilt/sle_5k.h5ad
```

The source atlas is about 12 GB. The same source and NumPy version yield the same selected cells, genes and counts. Across HDF5 library versions, compare matrix contents and ordered IDs rather than file bytes.

## Other-dataset workflow

After downloading the linked COVID atlas to `data/COVID.h5ad`, run:

```bash
scmetaroute prepare data/COVID.h5ad runs/covid/panel --counts raw
scmetaroute fit runs/covid/panel runs/covid/model --device cuda:0 --save-checkpoint
python downstream_tasks/export_metadata.py data/COVID.h5ad runs/covid/metadata.tsv
python downstream_tasks/metacell_quality.py runs/covid/panel \
  runs/covid/model/CELL_TO_METACELL.tsv runs/covid/metadata.tsv runs/covid/quality
python downstream_tasks/differential_expression.py runs/covid/panel \
  runs/covid/model/CELL_TO_METACELL.tsv runs/covid/metadata.tsv runs/covid/de --case 'COVID-19'
python downstream_tasks/downstream_metrics.py runs/covid/panel \
  runs/covid/model/CELL_TO_METACELL.tsv runs/covid/metadata.tsv runs/covid/downstream \
  --de-genes runs/covid/de/DE_GENES.tsv.gz
```

For other datasets, choose their count location and disease labels. `--counts` accepts `X`, `raw` or a layer name; automatic detection checks `layers['counts']`, `raw.X`, then `X`. Custom metadata column names can be specified with `export_metadata.py --cell-type-key NAME --donor-key NAME --disease-key NAME --tissue-key NAME`. Inputs need unique cell/gene IDs and nonnegative integer counts, with positive panel counts per cell.

## Product details

In `METACELL_EXPRESSION.h5ad`, `X` is log-CP10K expression; `layers['raw_sum']` and `layers['raw_mean']` store count sums and per-cell means; `obsm['X_latent']` stores centers; and `obsp['relation']` stores undirected weighted relations. The terminal linkage's leaves use the same indices as the metacell rows. A separate `STAGE2_WARD_LINKAGE.npy` describes microcluster aggregation.

```python
import anndata as ad
import numpy as np
from scipy.sparse import load_npz

mc = ad.read_h5ad('runs/sle_demo/model/METACELL_EXPRESSION.h5ad')
relations = load_npz('runs/sle_demo/model/STAGE2_TREE_GRAPH.npz')
tree = np.load('runs/sle_demo/model/STAGE2_METACELL_WARD_LINKAGE.npy')
print(mc.shape, relations.shape, tree.shape)
# (100, 2000) (100, 100) (99, 4)
```

`RUN_REPORT.json` records effective settings, `TRAINING_HISTORY.tsv` records losses, and `PRODUCT_AUDIT.json` records count and assignment checks. A training checkpoint is saved with `--save-checkpoint`. Preprocessing writes disk-backed arrays requiring roughly 8 bytes per cell–selected-gene pair, plus the source and products.

## Frozen training inputs and outputs

Place these files in one directory: `X_panel_raw.npy` (cells × ordered panel genes, nonnegative counts), `X_encoder.npy` (same dimensions, historical standardized encoder input), `panel_library_size.npy` (the exact exposure used by the recorded run), `CELL_IDS.tsv` (`cell_id`, in row order), and `PANEL_GENES.tsv` (`gene_id`, in column order). `cell_index`, if present, is zero-based and sequential. Some historical exports used `panel_index` values with gaps following gene filtering; file row order is the matrix column order. For a legacy panel without `CELL_IDS.tsv`, export the cell IDs from the exact original source in its original order only after verifying the cohort, or use the matching archived membership TSV's first two columns.

Do not regenerate `X_encoder.npy` with generic preprocessing when attempting exact paper replay. Historical panels can differ in feature selection and library reference even if cell counts are identical. The training command consumes frozen arrays rather than guessing their original transformations.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scmetaroute fit frozen-panel out/run --seed 1101 --batch-size 4096 --save-checkpoint
```

This is the verified fresh-fit configuration for all 56,676 ACR_CLAD cells on CUDA, with CPU clustering limited to one thread. It is not a benchmark timing configuration. For exact reported scores, use the archived assignments.

The benchmark configurations in Figures 2–3 use seed 1101, a 32-dimensional latent space, 750 initial microclusters, a support floor of 8, 500 target metacells, and 8 cophenetic neighbors. Use the historical `RUN_REPORT.json` and per-run overrides for the exact batch size, epoch budget and device. The release retains the original large-cohort behavior (mini-batch microclustering above 200,000 rows and a reduced epoch budget at 500,000 rows or more); all rows still participate in fitting and final assignment. The frozen input for the islet application is distinct from the human-pancreas benchmark input; the islet reference run used batch size 8192. `MC124` refers to ID 124 in the frozen islet assignment, not to ID 124 in a new fit.

## Figure 2: metacell quality and cell-type DE concordance

```bash
python downstream_tasks/metacell_quality.py frozen-panel out/run/CELL_TO_METACELL.tsv metadata.tsv out/quality
```

`metadata.tsv` needs `cell_id` and `cell_type`. The scorer writes the original post-fit metric definitions, within-type membership retention, and the signed prefix concordance curve. Its `Accuracy` and `Balanced Accuracy` are post-fit label mapping checks, distinct from the donor-held-out label transfer below.

## Figure 3: donor disease effects, downstream tasks, scaling, and ablation

```bash
python downstream_tasks/differential_expression.py frozen-panel out/run/CELL_TO_METACELL.tsv metadata.tsv out/de --case 'COVID-19'
python downstream_tasks/downstream_metrics.py frozen-panel out/run/CELL_TO_METACELL.tsv metadata.tsv out/tasks \
  --de-genes out/de/DE_GENES.tsv.gz
```

Add `donor_id`, `disease` and `tissue` to metadata as required. Figure 3 used `normal` as the control, at least three donors per arm and at least 20 cells per donor/type. For the other disease cohorts pass the archived conditions (`systemic lupus erythematosus`, `Crohn disease`, `Alzheimer disease`, `Parkinson disease`) via `--case`. DE outputs separate signed-gene recovery from expression-effect correlation, RMSE and sign concordance. The downstream scorer reports neighborhood, clustering, cross-donor annotation and co-expression metrics, plus signed disease-DE ranking when a gene-level table is provided.

For Figure 3f, use the same panel, seed, batch size and epoch budget in each run. The four presented configurations are:

| Configuration | `fit` flags |
| --- | --- |
| Full | `--anchor graph --final-merger ward` |
| Without Ward | `--anchor graph --final-merger kmeans` |
| Without anchoring | `--anchor none --final-merger ward` |
| Without anchoring and Ward | `--anchor none --final-merger kmeans` |

The original matched ablation used 160+40 epochs, batch size 4096 and deterministic PyTorch settings with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and four CPU threads. Use `--deterministic` for this protocol. Score each run separately with `metacell_quality.py`; the paper summary is an equal-weight mean across the eight datasets. The runtime in `RUN_REPORT.json` measures the release training and metacell export as one run; it is not a reconstruction of the historical runtime figure or its estimates.

## Figure 4: fixed islet region

The region is the intersection of MC124 with author-annotated heat-shock-associated β cells, referenced against the remainder of that same state. Preserve the frozen membership, donor metadata and feature/signature definitions across analyses.

```bash
Rscript downstream_tasks/islet_differential_expression.R MC124_state4_PAIRED_COUNTS.tsv.gz \
  MC124_state4_PAIRED_SAMPLES.tsv out/islet_edger.tsv
python -m pip install ".[paper]"
python downstream_tasks/islet_donor_analysis.py islet-figure-source-data out/islet
```

Install Bioconductor edgeR for the first command (verified with R 4.5.2 and edgeR 4.8.2). Sample rows must follow the pseudobulk count columns; each donor has one `rest` and one `local` sample, with at least 20 cells per region. The second command replays age/sex-adjusted abundance models and donor-paired signature contrasts from `DISEASE_REGION_DONORS.tsv`, `PARENT_STATE_DONORS.tsv`, `STRESS_DONOR_PAIRS.tsv`, and `EXACT_SIZE_DONOR_PAIRS.tsv`. Candidate selection, cross-candidate multiplicity corrections, and stimulus signature construction are upstream frozen inputs to this replay; it does not rerun those steps or recalculate a different FDR family from MC124 alone.

## Data sources

Download links and exact atlas-version URLs are listed in the [main README](../README.md#other-datasets). Public accessions refer to the source atlases. The exact paper replay additionally requires the archived cell order, frozen expression inputs and evaluation metadata.
