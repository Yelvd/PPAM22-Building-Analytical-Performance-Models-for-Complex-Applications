#!/bin/bash

#hemos=("50" "200")
hemos=("200")
ppn=128
nnodes=1
tries=2

EXP="cube"
# Name should not contain a '-'
NAME="imbalance_hemo"
w="--wait"

# empyt for loop to copy

for ((i=0; i< ${#hemos[@]}; i++ )); do
	h=${hemos[i]}
	RBCFILE=RBC-18-09-${h}.pos
	CONFIGFILE=config-${ppn}.xml

	# Jobname should be folloing the rule: expname-p1_v1-p2_v2-jobid
	jobname=${NAME}-t_${ppn}-H_s${h}

	echo "Launching: ${jobname}"
	sbatch -a 0-$tries --exclusive $w --ntasks-per-node $ppn --exclusive --job-name $jobname --output output/$jobname-%j.out -N 1 experiment.job $EXP $CONFIGFILE $RBCFILE
done

for ((i=0; i< ${#hemos[@]}; i++ )); do
	h=${hemos[i]}
	RBCFILE=RBC-18-00-${h}.pos
	CONFIGFILE=config-${ppn}.xml

	# Jobname should be folloing the rule: expname-p1_v1-p2_v2-jobid
	jobname=${NAME}-t_${ppn}-H_z${h}

	echo "Launching: ${jobname}"
	sbatch -a 0-$tries --exclusive $w --ntasks-per-node $ppn --exclusive --job-name $jobname --output output/$jobname-%j.out -N 1 experiment.job $EXP $CONFIGFILE $RBCFILE
done
