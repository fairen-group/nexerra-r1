> 🚧 **Under Construction:**  
> This is an initial release of the functionality. Further documentation and cleanup are still in progress. Currently only model training and inference is supported. Everything currently documented in this README should be runnable. If you encounter challenges, reach out to dm958[at]cam[dot]ac[dot]uk.

> This repository will be under continuous development; a stable version will be released upon final publication.

# Nexerra-R1
![Demo](assets/header.gif)

Open-source code for NexerraR1 as described in the following pre-print.

```bib
@article{menon2026a,
  title={A chemical language model for reticular materials design},
  author={Menon, Dhruv and Singh, Vivek and Chen, Xu and Alizadeh Kiapi, Mohammad Reza and Zyuzin, Ivan and MacLeod, Hamish W and Rampal, Nakul and Shepard, William and Yaghi, Omar and Fairen-Jimenez, David},
  journal={arRxiv preprint},
  year={2026}
}
```

This repository currently focuses on model training and inference for:
- direct linker design using the 'Direct Design' mode
- scaffold-constrained linker design using the 'Scaffold-constrained Design' mode
- flow-guided linker design using the 'Flow Design' mode

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

3. **Add the repo to your `PYTHONPATH`**

   If you installed the repo into `/my/path/to/nexerra-R1`, run:

   ```bash
   export PYTHONPATH="/my/path/to/nexerra-R1"
   ```

4. **Keep the required repo artifacts in place**
5. 
   The current inference code expects these assets to exist:

   ```text
   artifacts/ckpt/vae/no_prop_vae_epoch_120.pt
   artifacts/ckpt/flow/otcfm_step_180000.pt
   artifacts/latent_banks/latent_bank.pt
   artifacts/latent_banks/latent_bank_len.pt
   data/processed/tokenized_dataset.pkl
   data/processed/train_smiles.txt
   designed/linker/inference_config.txt
   ```

6. **Download external runtime assets**

   Large runtime assets are intentionally not stored in Git. After cloning, fetch them from the Zenodo deposit into the expected locations (if this is not up already, it will be soon; any relevant instructions may be updated):

   ```bash
   python setup_assets.py --base-url "https://zenodo.org/records/<record-id>/files"
   ```

   The bootstrapper downloads the default runtime bundle into:
   - `artifacts/ckpt/vae/no_prop_vae_epoch_120.pt`
   - `artifacts/ckpt/flow/otcfm_step_180000.pt`
   - `artifacts/latent_banks/latent_bank.pt`
   - `artifacts/latent_banks/latent_bank_len.pt`
   - `data/processed/tokenized_dataset.pkl`

   Notes:
   - `setup.py` is available as a thin wrapper around `setup_assets.py`, so `python setup.py --base-url ...` works too.

> [!TIP]
> **Note for macOS Users:** If you are using macOS with Apple Silicon, please be aware of potential numerical instability with the MPS backend. We recommend using the CPU device for MatterSim on Mac to avoid these issues.


## Model Training
<a id="model-training"></a>

The training workflow has two stages:
- train the VAE linker model
- build a latent bank and train the OT-CFM flow model on top of the pretrained VAE

### 1. Prepare tokenized training data

From `nexerra/utils/`, preprocess the raw training CSV into the tokenized dataset bundle used by both the VAE and flow stages:

```bash
python preprocess.py --mode gen --src ../../data/raw/training_dataset.csv --dst ../../data/processed
```

This produces artifacts such as:
- `data/processed/tokenized_dataset.pkl`
- `data/processed/tok2id.json`
- `data/processed/id2tok.json`

### 2. Train the VAE model

The VAE training entrypoint is:

```text
nexerra/model/Trainer.py
```

Run it from `nexerra/model/` with the tokenized dataset and the raw training CSV:

```bash
python Trainer.py \
  --data ../../data/processed/tokenized_dataset.pkl \
  --rs ../../data/raw/training_dataset.csv \
  --batch 128
```

The production-oriented defaults in the training script use a `latent_dim` of `128`. Checkpoints are written per epoch as:

```text
artifacts/ckpt/vae/vae_epoch_<n>.pt
```

To resume training from an existing checkpoint:

```bash
python Trainer.py \
  --data ../../data/processed/tokenized_dataset.pkl \
  --rs ../../data/raw/training_dataset.csv \
  --batch 128 \
  --resume ../../artifacts/ckpt/vae/vae_epoch_<n>.pt \
  --epoch <n>
```

### 3. Build the latent bank

The flow model is trained on latent encodings produced by the pretrained VAE. Build the latent bank from the training SMILES list using:

```text
nexerra/cfm/build_bank.py
```

Run it from `nexerra/cfm/`:

```bash
python build_bank.py \
  --mode build \
  --data ../../data/processed/train_smiles.txt \
  --batch_size 128 \
  --savepath ../../artifacts/latent_banks/latent_bank.pt
```

This script loads the VAE checkpoint from `artifacts/ckpt/vae/no_prop_vae_epoch_120.pt` by default and writes the latent bank to the path you provide.

### 4. Train the OT-CFM flow model

The flow training entrypoint is:

```text
nexerra/cfm/otcfm_trainer.py
```

Run it from `nexerra/cfm/` using the latent bank built in the previous step:

```bash
python otcfm_trainer.py \
  --latent_pt ../../artifacts/latent_banks/latent_bank.pt \
  --percentile_start 70 \
  --percentile_end 100 \
  --steps 50000 \
  --batch 2048 \
  --out_path ../../artifacts/ckpt/flow/otcfm_step_50000.pt
```

Important defaults from the training script:
- the flow trainer reloads the pretrained VAE from `artifacts/ckpt/vae/no_prop_vae_epoch_120.pt`
- the tokenizer bundle is loaded from `data/processed/tokenized_dataset.pkl`
- `--matcher otcfm` is the active matcher choice
- `--mode max` or `--mode min` controls whether the conditioned property is maximized or minimized

The percentile arguments define the property slice used for conditional flow matching. Adjust them to match the reward/property regime you want the model to learn.

### 5. Evaluate a trained flow checkpoint

The same flow training script also supports evaluation sweeps over guidance scales:

```bash
python otcfm_trainer.py \
  --eval_ckpt ../../artifacts/ckpt/flow/otcfm_step_50000.pt \
  --eval_scales 0.0,1.0,1.5,2.0,3.0 \
  --eval_samples 10000
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

## Additional Notes
<a id="additional-notes"></a>
- GPU speeds up sampling significantly, but chemistry-heavy filtering and RDKit-based analysis remain CPU-bound.
- Some MOF construction utilities depend on external tooling not bundled in this repo.
