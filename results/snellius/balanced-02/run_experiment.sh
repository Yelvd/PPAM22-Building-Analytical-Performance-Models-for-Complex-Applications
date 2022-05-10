#!/bin/bash

ppn=128
nnodes=1
tries=3


# Name should not contain a '-'
NAME="balanced"
w="--wait"

# empyt for loop to copy
# for ((i=0; i< ${#sizes[@]}; i++ )); do
# done


# imbalanced 18H
EXP="cube"
h="02"
RBCFILE=RBC-h0${h}.pos
CONFIGFILE=config-${ppn}.xml

# Jobname should be folloing the rule: expname-p1_v1-p2_v2-jobid
jobname=${NAME}-t_${ppn}-H_${h}

echo "Launching: ${jobname}"
sbatch -a 0-$tries --exclusive $w --ntasks-per-node $ppn --exclusive --job-name $jobname --output output/$jobname-%j.out -N 1 experiment.job $EXP $CONFIGFILE $RBCFILE
