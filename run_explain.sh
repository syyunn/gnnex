#!/bin/bash

module load anaconda/2023a

export LD_LIBRARY_PATH="/usr/local/anaconda/anaconda3-2023a/lib:$LD_LIBRARY_PATH"

python ~/gnnex/hetero/explain_edge.py
