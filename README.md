> 🚧 **Under Construction:**  
> This is an initial release of the functionality. Further documentation and cleanup are still in progress. Currently only model training and inference is supported. Everything currently documented in this README should be runnable. If you encounter challenges, reach out to dm958[at]cam[dot]ac[dot]uk.

# NexerraR1
![Demo](assets/header.gif)
Open-source code for NexerraR1 as described in the following pre-print.

```bib
@article{ahern2025atom,
  title={A chemical language model for reticular materials design},
  author={Menon, Dhruv and Singh, Vivek and Chen, Xu and Alizadeh Kiapi, Mohammad Reza and Zyuzin, Ivan and MacLeod, Hamish W and Rampal, Nakul and Shepard, William and Yaghi, Omar and Fairen-Jimenez, David},
  journal={arRxiv preprint},
  year={2026}
}
```

This repository currently focuses on model training and inference for:
- direct linker design
- scaffold-constrained linker design
- flow-guided seeded generation

## Set-up
<a id="set-up"></a>

This repo is intended to be used through a curated Conda environment.

1. **Clone the repo**

2. **Create the supported Conda environment**

   From the repo root, run:

   ```bash
   conda env create -f environment.yml
   conda activate nexerra
   ```

   The provided `environment.yml` is a curated environment for the main Nexerra workflow.

   Note:
   This should be more portable than a full machine-specific environment export, but some optional workflows still rely on system-sensitive tooling.

3. **Add the repo to your `PYTHONPATH`**

   If you installed the repo into `/my/path/to/nexerra-R1`, run:

   ```bash
   export PYTHONPATH="/my/path/to/nexerra-R1"
   ```

4. **Keep the required repo artifacts in place**

   The current inference code expects these assets to exist:

   ```text
   artifacts/ckpt/vae/no_prop_vae_epoch_120.pt
   artifacts/ckpt/flow/otcfm_step_180000.pt
   artifacts/latent_banks/latent_bank.pt
   data/processed/tokenized_dataset.pkl
   data/processed/train_smiles.txt
   data/processed/processed_stats.json
   designed/linker/inference_config.txt
   ```


## Inference
<a id="inference"></a>

The main production-relevant inference entrypoint is:

```text
nexerra/inference/FlowDesign.py
```

Run it from `nexerra/inference/`:

```bash
python FlowDesign.py --alpha 0.9 --num-samples 1000 --batch-size 128 --reward gas --threshold 0.5 --filters
```

This path uses CUDA automatically when a working GPU PyTorch install is available:

```python
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

The output of flow inference is driven by:

```text
designed/linker/run/input.txt
designed/linker/run/output.txt
designed/linker/run/output_all.txt
designed/linker/inference_config.txt
```

## Supported Modes
<a id="supported-modes"></a>

### Direct Design

The direct design path uses the VAE-based inference code in:

```text
nexerra/inference/Design.py
```

This mode expects a single seed linker with `[Lr]` connector atoms in:

```text
designed/linker/run/input.txt
```

### Scaffold-Constrained Design

The scaffold-constrained path uses:

```text
nexerra/inference/ScafDesign.py
```

This mode expects:
- first line: scaffold/core SMILES
- second line: arm SMILES

### Flow-Guided Seeded Design

The flow-guided seeded design path uses:

```text
nexerra/inference/FlowDesign.py
```

This combines a pretrained VAE with a latent OT-CFM model and uses the flow checkpoint metadata to steer decoding.

## Example Inputs
<a id="example-inputs"></a>

Example linker inputs are documented in:

```text
designed/linker/README.md
```

That file includes examples for:
- direct design
- scaffold-constrained design

## Repository Layout
<a id="repository-layout"></a>

The most relevant top-level directories are:

- `nexerra/`
  Core code for models, inference, utilities, and build scripts.

- `artifacts/`
  Model checkpoints, latent banks, and related runtime assets.

- `data/processed/`
  Tokenizer data and processed statistics used by inference and evaluation paths.

- `designed/linker/`
  User-facing linker inference inputs, outputs, configuration, and examples.

- `designed/reference/`
  Reference molecules, linkers, scaffold variants, and related generated assets.

- `designed/cif/`
  CIF outputs, examples, and build-side artifacts. This is not required for core inference-only production use.

## Dependencies
<a id="dependencies"></a>

This repository currently mixes:
- core inference dependencies
- optional training and evaluation dependencies
- MOF construction dependencies
- vendored utility dependencies
- a separate bio/reward subproject under `nexerra/inference/bio/`

For most users, the practical dependency entrypoint is:

```bash
conda env create -f environment.yml
conda activate nexerra
```

The breakdown below exists so it is clear what the repo actually depends on and which parts are optional.

### Core inference dependencies

These are the practical dependencies for the main linker inference paths in:
- `nexerra/inference/Design.py`
- `nexerra/inference/ScafDesign.py`
- `nexerra/inference/FlowDesign.py`

Required Python packages:
- `torch`
- `numpy`
- `pandas`
- `rdkit`
- `selfies`
- `tqdm`
- `pyfiglet`
- `botorch`
- `torchdiffeq`
- `torchcfm`

Also used by core inference and reward logic:
- `matplotlib` for some utility and analysis paths
- `pickle`
- `json`
- `pathlib`
- `argparse`
- `logging`

### MOF construction and CIF tooling

Used by:
- `nexerra/build/Construct.py`
- `nexerra/build/ScaffConstruct.py`
- `nexerra/build/addH.py`

Required or expected:
- `rdkit`
- `openbabel` / `pybel`
- external `tobacco3.0` checkout for MOF assembly

Notes:
- `Construct.py` and `ScaffConstruct.py` do not bundle ToBaCCo; you need to provide it separately.
- `addH.py` uses Open Babel Python bindings.

### Training and preprocessing dependencies

Used by the VAE and flow training / preprocessing utilities:
- `nexerra/utils/preprocess.py`
- `nexerra/model/`
- `nexerra/cfm/`

Required Python packages:
- `torch`
- `numpy`
- `pandas`
- `rdkit`
- `selfies`
- `tqdm`
- `pyfiglet`
- `torchdiffeq`
- `torchcfm`

### Analysis and diagnostics dependencies

Used by:
- `nexerra/utils/analysis.py`
- `nexerra/utils/diagnostics.py`

Required Python packages:
- `matplotlib`
- `umap-learn`
- `rdkit`
- `numpy`
- `pandas`
- `tqdm`

### Vendored SCScore utility dependencies

The repository vendors SCScore-related code under:
- `nexerra/utils/scscore/`

Inference-side usage mainly depends on:
- `rdkit`
- `numpy`
- `six`

Legacy training utilities inside the vendored SCScore code additionally use:
- `tensorflow`
- `h5py`
- `gzip`
- `pymongo`

These are not required for normal Nexerra inference.

### Bio / reward subproject dependencies

There is a separate subproject at:
- `nexerra/inference/bio/`

Its `pyproject.toml` declares:
- `python >=3.10,<3.13`
- `pandas`
- `numpy`
- `pyarrow`
- `rdkit-pypi`
- `jupyter`
- `ipykernel`
- `matplotlib`
- `python-dateutil`
- `tqdm`
- `umap-learn`
- `pytest` as a dev dependency

This subproject is not required for the main production linker inference path.

### Standard library modules used throughout the repo

Commonly used built-in modules include:
- `os`
- `sys`
- `math`
- `json`
- `pickle`
- `argparse`
- `logging`
- `pathlib`
- `subprocess`
- `random`
- `collections`
- `itertools`
- `warnings`
- `csv`
- `time`
- `tempfile`
- `shutil`

### Runtime artifacts and non-package dependencies

The code also depends on several on-disk assets rather than installable packages:
- `artifacts/ckpt/vae/no_prop_vae_epoch_120.pt`
- `artifacts/ckpt/flow/otcfm_step_180000.pt`
- `artifacts/latent_banks/latent_bank.pt`
- `data/processed/tokenized_dataset.pkl`
- `data/processed/train_smiles.txt`
- `data/processed/processed_stats.json`
- `designed/linker/inference_config.txt`

For some build workflows, additional external files are expected:
- ToBaCCo templates and node/edge directories
- CIF templates for the selected net

## Production Notes
<a id="production-notes"></a>

If you are moving to a production inference-only deployment:

- keep `nexerra/`
- keep `artifacts/`
- keep the required files in `data/processed/`
- keep `designed/linker/`
- keep only the specific reference assets you still need under `designed/reference/`
- exclude `evaluation/`
- exclude generated CIF outputs if you are not shipping MOF construction artifacts

## Additional Notes
<a id="additional-notes"></a>

- The evaluation scripts are not required for production inference.
- GPU speeds up sampling significantly, but chemistry-heavy filtering and RDKit-based analysis remain CPU-bound.
- Some MOF construction utilities depend on external tooling not bundled in this repo.
