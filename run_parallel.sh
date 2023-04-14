# This shell is designed to run Optuna optimization procedures in
# parallel in the background.
#
# Each optimization is executed in the format:
#       Python lstm_optimize.py "Study_Name" Number_of_trials.
# 
# When optimization is finished for a study, it will be saved in `~/opt_cache/`
# under "Study_Name". 
#
# Notes:
#   Include `wait` in between commands to do them in batches.
#   Run this in terminal via 'bash run_parallel.sh'.

#!/bin/bash

cd ~/Desktop/stock_analysis/
pwd
date

python lstm_optimize.py "OptStudy_v4" 20 & #1 
python lstm_optimize.py "OptStudy_v5" 20 & #"2
python lstm_optimize.py "OptStudy_v3" 20 & #3
python lstm_optimize.py "OptStudy_v4" 20 & #4
wait
python lstm_optimize.py "OptStudy_v5" 20 & #5
python lstm_optimize.py "OptStudy_v6" 20 & #6
python lstm_optimize.py "OptStudy_v7" 20 & #7
python lstm_optimize.py "OptStudy_v8" 20 & #8
wait
