#!/bin/bash

ppn=128
nnodes=1
tries=0

EXP="cube"
NAME="imbalance-domain"
w="--wait"

# for ((i=0; i< ${#sizes[@]}; i++ )); do
# done

h="18"
RBCFILE=RBC-h0${h}.pos
CONFIGFILE=config-${ppn}.xml
jobname=${NAME}-i_1-t_${ppn}-H_${h}

echo "Launching: ${jobname}"
sbatch -a 0-$tries --exclusive $w --ntasks-per-node $ppn --exclusive --job-name $jobname --output output/$jobname-%j.out -N 1 experiment.job $EXP $CONFIGFILE $RBCFILE

