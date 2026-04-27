#!/bin/bash
# run_retrieval.sh

# Activate the virtual environment
source /home/mathisb/HighRes/bin/activate

# Make sure it knows where to look for starships
export PYTHONPATH=/home/mathisb/Github/starships:$PYTHONPATH

# Run the python code (now any number of arguments can be passed)
echo "yaml file: $1"
echo "Other arguments: ${@:2}"
echo "Running the python code..."
# run_starships_retrieval yaml_file=$1 ${@:2}

cd /home/mathisb/Github/starships/starships
python retrieval.py yaml_file=$1 ${@:2}