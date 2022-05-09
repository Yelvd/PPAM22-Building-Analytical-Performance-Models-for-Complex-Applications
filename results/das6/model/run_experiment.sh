#!/bin/bash

tasks=( 24 )
sizes=("s1" "s2" "s3" "s4" "s5" "s6" "s7" "s8" "s9")
hemo=("000" "009" "010" "012" "014" "016" "018")
trys=1

for h in "${hemo[@]}"; do
	for t in "${tasks[@]}"; do
		for s in "${sizes[@]}"; do
			cp config-single-node-$s.xml current-config.xml
			cp RBC-h${h}.pos RBC.pos
			jobname=single-node-${t}-${s}-h${h}
	
			CONFIG_NAME=config-single-node-$s.xml sbatch -a 0-$trys --exclusive --ntasks-per-node $t --exclusive --wait --job-name $jobname --output output/$jobname-%j.out -N 1 experiment.job
		done
	done
done
