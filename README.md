# nequip-nac
Predicting non-adiabatic couplings (NACs) vectors with NequIP.

## Install
Create environment and install `torch`, e.g.
```
micromamba create -n nequip-nac python=3.10
micromamba activate nequip-nac
```

Clone `nequip-nac` and install
```
git clone https://github.com/mir-group/nequip-nac
cd nequip-nac
pip install -e .
```
To monitor training with `wandb`, install
```
pip install wandb
```
## Usage
Test in the directory of `nequip-nac/configs`:
```
nequip-train -cn tutorial
```
A folder named `test-nac1` will be created and a `wandb` project named `NequIP-NAC` will be created.

## Data Format
We recommend using `extxyz` format to store the data which includes atomic positions, species, energies from two states, forces from two states, and NACs.
You can checkout the `trial_data` folder to see an example of the data format. The `extxyz` files can be generated using `ase`.
