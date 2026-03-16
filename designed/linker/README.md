# Linker Design Inference

This document covers the current linker design inference inputs, basic usage, and example outputs.

> [🚧] Under construction: this is an initial release of the linker design functionality. Further documentation and cleanup are still in progress. For now, only inference is supported. The examples documented in this README should be runnable.

## How to Run
The current production-relevant inference entrypoint is `nexerra/inference/FlowDesign.py` and `nexerra/inference/Design.py`

Run it from `nexerra/inference/` (for example):

```bash
python FlowDesign.py --alpha 0.9 --num-samples 1000 --batch-size 128 --reward gas --threshold 0.5 --filters
```

Inputs and outputs:
- Input seed file: `designed/linker/run/input.txt`
- Filtered output: `designed/linker/run/output.txt`
- Full output: `designed/linker/run/output_all.txt`
- Runtime settings: `designed/linker/inference_config.txt`

The scaffold-constrained mode is available through `nexerra/inference/ScafDesign.py` if needed.

## Direct Design
The model expects a single SMILES string with the coordinating moieties denoted as `[Lr]`.

Example input:

```txt
[Lr]c1cc([Lr])cc([Lr])c1
```

The corresponding output is a `.txt` file containing SMILES strings that pass the configured criteria.

Example output:

```txt
c1cc([Lr])cc2ncc([Lr])cc12
NC(=C(Cl)[Lr])c1ccc([Lr])cc1N
ClC([Lr])SC1=CC2=CC=C(SC[Lr])C2=C1
Nc1ccc2c(N)c([Lr])ccc2c1C(Cl)[Lr]
Nc1c([Lr])ccc2c(N)c([Lr])ccc12
NCc1cc([Lr])ccc1C=CC([Lr])NCl
Nc1cc([Lr])ccc1-c1ccc([Lr])c(N)c1N
NCc1cc([Lr])ccc1C#C[Lr]
ClCSc1cc([Lr])ccc1C=CSC[Lr]
Nc1c([Lr])cc2c(N)cc([Lr])cc2c1N
NC(=CC(F)[Lr])c1ccc([Lr])cc1N
S=Cc1cc([Lr])ccc1C#C[Lr]
Nc1cc([Lr])ccc1-c1ccc([Lr])cc1S
Cc1cc([Lr])ccc1C=CCC([Lr])C#N
NC1=c2cc([Lr])cc(S)c2=C([Lr])C1(N)N
Nc1cc([Lr])ccc1C(Cl)=C(Cl)[Lr]
Nc1cc2c(N)c([Lr])ccc2c(N)c1[Lr]
Nc1cc([Lr])ccc1C=CC(N)(N)[Lr]
Nc1c([Lr])ccc2c(NCl)c([Lr])ccc12
NCc1cc([Lr])ccc1C=CC(N)[Lr]
C#Cc1cc([Lr])ccc1SC[Lr]
Nc1cc([Lr])cc2c(Cl)c(Cl)c([Lr])cc12
Nc1cc([Lr])ccc1-c1c(N)cc([Lr])cc1N
Nc1cc([Lr])ccc1C(Cl)=CC(N)[Lr]
Nc1cc([Lr])ccc1C#C[Lr]
Nc1cc([Lr])cc(S)c1C=CC(N)[Lr]
```

## Scaffold-Constrained Design

The model expects two SMILES strings with coordinating moieties denoted as `[Lr]`.

Input format:
- First line: core or scaffold
- Second line: 2-connected arm

Example input:

```txt
[Lr]c1cc([Lr])cc([Lr])c1
[Lr]c1ccc([Lr])cc1
```

The corresponding output is a `.csv` file with three columns: `scaffold`, `arm`, and `linker`.

Example output:

```csv
scaffold,arm,linker
[Lr]c1cc([Lr])cc([Lr])c1,[Lr]c1ccc([Lr])cc1,[Lr]c1ccc(-c2cc(-c3ccc([Lr])cc3)cc(-c3ccc([Lr])cc3)c2)cc1
```
