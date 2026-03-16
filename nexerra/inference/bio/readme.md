This project shall create a reward function for a model that generates metal organic frameworks (MOFs) based on a seed. This reward function shall select MOFs most suitable for drug delivery applications by analysing their metal nodes and organic linkers and extracting their properties using SMILES calls to RDKit.
The reward function prioritises:
1) Known non-toxic metals
2) Linker length: longer => high loading capacity and pore volume
3) Solubility: highly soluble => better delivery and lower side efffects
4) Surface area => higher loading
5) Benign linkers with low charges and low interaction with payload.