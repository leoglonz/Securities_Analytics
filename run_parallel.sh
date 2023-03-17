# This shell is designed to run Optuna optimization procedures in
# parallel in the background.
#
# Notes:
#   Include `wait` in between commands to do them in batches.
#   Run this in terminal via 'bash run_parallel.sh'.

#!/bin/bash

cd ~/stock_analysis/
pwd
date

python lstm_optimization & #1 
python particletracking.py h329 33 & #2
python particletracking.py h229 27 & #3
python particletracking.py h148 13 & #4
wait
python particletracking.py h229 20 & #5
python particletracking.py h242 24 & #6
python particletracking.py h229 22 & #7
wait
python particletracking.py h242 80 & #8
python particletracking.py h148 37 & #9
python particletracking.py h148 28 & #10 
python particletracking.py h148 68 & #11
wait
python particletracking.py h148 278 & #12
python particletracking.py h229 23 & #13
python particletracking.py h148 45 & #14
wait
python particletracking.py h148 283 & #15
python particletracking.py h229 55 & #16
python particletracking.py h329 137 & #17
wait
python particletracking.py h148 80 & #18
python particletracking.py h148 329 & #19
wait
