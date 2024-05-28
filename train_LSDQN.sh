#!/bin/bash

#!/bin/bash

python train.py --distribution ER_20 --number_of_vertices 20 --model LSDQN --device 1
python train.py --distribution ER_40 --number_of_vertices 40 --model LSDQN --device 1
python train.py --distribution ER_60 --number_of_vertices 60 --model LSDQN --device 1
python train.py --distribution ER_100 --number_of_vertices 100 --model LSDQN --device 1
python train.py --distribution ER_200 --number_of_vertices 200 --model LSDQN --device 1

python train.py --distribution BA_20 --number_of_vertices 20 --model LSDQN --device 1
python train.py --distribution BA_40 --number_of_vertices 40 --model LSDQN --device 1
python train.py --distribution BA_60 --number_of_vertices 60 --model LSDQN --device 1
python train.py --distribution BA_100 --number_of_vertices 100 --model LSDQN --device 1
python train.py --distribution BA_200 --number_of_vertices 200 --model LSDQN --device 1


python train.py --distribution HomleKim_200vertices_weighted  --model S2V
python train.py --distribution HomleKim_200vertices_unweighted  --model S2V
python train.py --distribution WattsStrogatz_200vertices_unweighted  --model S2V
python train.py --distribution WattsStrogatz_200vertices_weighted  --model S2V

python train.py --distribution HomleKim_200vertices_weighted  --model LSDQN 
python train.py --distribution HomleKim_200vertices_unweighted  --model LSDQN 
python train.py --distribution WattsStrogatz_200vertices_unweighted  --model LSDQN 
python train.py --distribution WattsStrogatz_200vertices_weighted  --model LSDQN

python train.py --distribution HomleKim_200vertices_weighted  --model LinearRegression
python train.py --distribution HomleKim_200vertices_unweighted  --model LinearRegression
python train.py --distribution WattsStrogatz_200vertices_unweighted  --model LinearRegression
python train.py --distribution WattsStrogatz_200vertices_weighted  --model LinearRegression

