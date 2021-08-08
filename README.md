# mario

## About

* This is only tested on Apple Silicon and Intel Mac
* Please change `clean_fceux` and install fceux to run on other platforms

## Mario GA

### Prereqs

```bash
# install fceux
brew install fceux

# this installs old gym 0.10
pip install numpy ppaquette-gym-super-mario

# upgrade gym
pip install -U gym
```
### How to run

* Genetic algorithm version

```bash
# To train from scratch
python mario_ga.py --stage ppaquette/SuperMarioBros-1-1-Tiles-v0

# To train from saved data (e.g. from gen 4)
python mario_ga.py --stage ppaquette/SuperMarioBros-1-2-Tiles-v0 --gen 4

# To replay (max speed)
python mario_ga.py --stage ppaquette/SuperMarioBros-1-2-Tiles-v0 --gen 4 --replay

# To replay (normal speed)
vim  ~/miniforge3/envs/ml/lib/python3.9/site-packages/ppaquette_gym_super_mario/lua/super-mario-bros.lua

# edit from "maximum" to "normal"
116 else
117     -- algo
118     -- emu.speedmode("maximum");
119     emu.speedmode("normal");

# run
python mario_ga.py --stage ppaquette/SuperMarioBros-1-2-Tiles-v0 --gen 4 --replay
```

## Mario Deep-Q

* Deep Q learning version
* TBD

