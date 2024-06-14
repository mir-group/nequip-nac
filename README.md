# nequip-nac
Nonadiabatic coupling vectors with NequIP

## Install 
Create environment and install `torch`, e.g.
```
conda create -n nequip-nac python=3.11
pip install torch
```
Clone `nequip-private` and switch to `cart_tensor` branch before installing
```
git clone https://github.com/mir-group/nequip-private.git
cd nequip-private
git checkout cart_tensor
pip install -e .
```
Clone `nequip-nac` and install
```
git clone https://github.com/cw-tan/nequip-nac.git
cd nequip-nac
pip install -e .
```
To monitor training with `wandb`, install
```
pip install wandb
```
Test
```
nequip-train configs/minimal.yaml
```
