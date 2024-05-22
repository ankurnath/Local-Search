#!/bin/bash

python train.py --distribution ER_20 --number_of_vertices 20 --model ECO_DQN --device 0
python train.py --distribution ER_40 --number_of_vertices 40 --model ECO_DQN --device 0
python train.py --distribution ER_60 --number_of_vertices 60 --model ECO_DQN --device 0
python train.py --distribution ER_100 --number_of_vertices 100 --model ECO_DQN --device 0
python train.py --distribution ER_200 --number_of_vertices 200 --model ECO_DQN --device 0

python train.py --distribution BA_20 --number_of_vertices 20 --model ECO_DQN --device 0
python train.py --distribution BA_40 --number_of_vertices 40 --model ECO_DQN --device 0
python train.py --distribution BA_60 --number_of_vertices 60 --model ECO_DQN --device 0
python train.py --distribution BA_100 --number_of_vertices 100 --model ECO_DQN --device 0
python train.py --distribution BA_200 --number_of_vertices 200 --model ECO_DQN --device 0
