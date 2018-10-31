#!/bin/bash

check_dirs=""
check_dirs=${check_dirs}" ./automl"
check_dirs=${check_dirs}" ./datasets"
check_dirs=${check_dirs}" ./learners"
check_dirs=${check_dirs}" ./nets"
check_dirs=${check_dirs}" ./rl_agents"
check_dirs=${check_dirs}" ./tools"
check_dirs=${check_dirs}" ./utils"
echo "Folders to be checked:"
echo "  "${check_dirs}
pylint --jobs=4 --reports=y ${check_dirs} --ignore=utils/external | tee .pylint_results

echo
echo "***************************************************"
echo "***********  head -n50 .pylint_results  ***********"
echo "***************************************************"
echo
head -n50 .pylint_results
