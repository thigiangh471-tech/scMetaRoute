# Bundled SLE example

`sle_5k.h5ad` contains 5,000 real PBMCs × 30,172 genes, sampled from the SLE atlas by Perez et al., *Single-cell RNA-seq reveals cell type–specific molecular and genetic associations to lupus*, Science (2022), [DOI 10.1126/science.abf1970](https://doi.org/10.1126/science.abf1970).

Source collection: [CZ CELLxGENE](https://cellxgene.cziscience.com/collections/436154da-bcf1-4130-9c8b-120ff9a888f2). Exact source H5AD version: [4118e166-34f5-4c1f-9eed-c64b90a3dace](https://datasets.cellxgene.cziscience.com/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad). Raw study accession: [GSE137029](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137029).

## Run the included example

After installation, run from the repository root:

```bash
python examples/run_demo.py
```

The script reads `data/sle_5k.h5ad` and saves the complete workflow to `runs/sle_demo/`. For preprocessing alone, use the included file directly:

```bash
scmetaroute prepare data/sle_5k.h5ad runs/sle_panel
```

The input's `X` contains raw counts; the default preparation selects 2,000 variable genes. Each command uses its own output directory.

## Contents and selection

- `X`: sparse nonnegative integer RNA counts, copied from the source's `raw/X`.
- `obs`: original cell IDs, `cell_type`, `donor_id`, `disease`, `tissue`.
- `var`: all source raw-gene IDs, plus available gene names and references.
- `uns['citation']`: source citation and version.

Using NumPy `default_rng(1101)`, sample ten donors from each of the `normal` and `systemic lupus erythematosus` groups among donors with at least 500 source cells. Within each selected group, sort the selected donor IDs and sample 250 cells per donor. Finally restore original source row order. Selection uses donor and condition labels to obtain a small two-condition tutorial; cell-type labels and model scores are not used. Training uses counts only. No genes are removed in the bundled file; the tutorial subsequently selects 2,000 variable genes.

The subset illustrates the workflow. Its scores describe this subset and are not the full-atlas manuscript scores. `sle_5k_provenance.json` records the selection and hashes of the file and ordered IDs.

To rebuild the subset from the source atlas, use [create_demo_data.py](../examples/create_demo_data.py) and the [download/rebuild commands](../docs/reproduction.md#rebuild-the-sle-subset). The bundled file is ready to use.

## Data attribution

The original contributors and CZ CELLxGENE distribute the source data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as described in the [CELLxGENE data submission policy](https://cellxgene.cziscience.com/docs/032__Contribute%20and%20Publish%20Data). This subset retains that attribution and license. The modifications are row subsampling, retention of selected public metadata fields, placement of original raw counts in `X`, and conversion of counts to integer storage. Cite the original study and source collection when reusing it.
