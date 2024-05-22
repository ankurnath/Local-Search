#!/bin/bash

python evaluate.py --distribution ER_20 --model ECO_DQN 
python evaluate.py --distribution ER_40 --model ECO_DQN 
python evaluate.py --distribution ER_60 --model ECO_DQN 
python evaluate.py --distribution ER_100  --model ECO_DQN 
python evaluate.py --distribution ER_200  --model ECO_DQN 

python evaluate.py --distribution BA_20 --model ECO_DQN 
python evaluate.py --distribution BA_40 --model ECO_DQN 
python evaluate.py --distribution BA_60  --model ECO_DQN 
python evaluate.py --distribution BA_100  --model ECO_DQN 
python evaluate.py --distribution BA_200  --model ECO_DQN


python evaluate.py --distribution ER_20 --model S2V 
python evaluate.py --distribution ER_40 --model S2V 
python evaluate.py --distribution ER_60 --model S2V 
python evaluate.py --distribution ER_100  --model S2V 
python evaluate.py --distribution ER_200  --model S2V 

python evaluate.py --distribution BA_20 --model S2V 
python evaluate.py --distribution BA_40 --model S2V 
python evaluate.py --distribution BA_60  --model S2V 
python evaluate.py --distribution BA_100  --model S2V 
python evaluate.py --distribution BA_200  --model S2V


python evaluate.py --distribution ER_20 --model LinearRegression 
python evaluate.py --distribution ER_40 --model LinearRegression 
python evaluate.py --distribution ER_60 --model LinearRegression 
python evaluate.py --distribution ER_100  --model LinearRegression 
python evaluate.py --distribution ER_200  --model LinearRegression 

python evaluate.py --distribution BA_20 --model LinearRegression 
python evaluate.py --distribution BA_40 --model LinearRegression 
python evaluate.py --distribution BA_60  --model LinearRegression 
python evaluate.py --distribution BA_100  --model LinearRegression 
python evaluate.py --distribution BA_200  --model LinearRegression


python evaluate.py --distribution ER_20 --model LSDQN 
python evaluate.py --distribution ER_40 --model LSDQN 
python evaluate.py --distribution ER_60 --model LSDQN 
python evaluate.py --distribution ER_100  --model LSDQN 
python evaluate.py --distribution ER_200  --model LSDQN 

python evaluate.py --distribution BA_20 --model LSDQN 
python evaluate.py --distribution BA_40 --model LSDQN 
python evaluate.py --distribution BA_60  --model LSDQN 
python evaluate.py --distribution BA_100  --model LSDQN 
python evaluate.py --distribution BA_200  --model LSDQN
