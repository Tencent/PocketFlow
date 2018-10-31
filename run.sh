#!/bin/bash

# convert AutoML-generated hyper-parameter file to PocketFlow-compatible format
python automl/cvt_hparam_file.py automl/automl_hparam.conf >> extra_args_from_automl

# run seven job and get job id
jobid=$(seven create --conf automl/seven.yaml --code `pwd` 2> seven.log | tee -a seven.log | \
    python -c "import sys, json; print json.load(sys.stdin)['JobId']" 2>> seven.log)

# save job id to file
echo "seven_id="${jobid} > pid_file

# wait until job finish
seven wait --jobId ${jobid} >> seven.log 2>&1

# get job log
seven log --jobId ${jobid} > results 2>> seven.log

# parse results to desired format
python automl/parse_results.py results >> result_file
