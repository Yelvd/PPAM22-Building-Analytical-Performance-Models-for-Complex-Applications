#!/bin/bash

tasks=( 128 )
sizes=("s1" "s2" "s3" "s4" "s5" "s6" "s7" "s8" "s9" "s10" "s11" "s12" "s13")
hemo=("000" "009" "010" "012" "014" "016" "018")
tries=1

for s in "${sizes[@]}"; do
	for t in "${tasks[@]}"; do
		for h in "${hemo[@]}"; do
			cp config-single-node-$s.xml current-config.xml
			cp RBC-h${h}.pos RBC.pos
			jobname=single-node-${t}-${s}-h${h}

			wait=""
			if [ $h = ${hemo[-1]} ]; then wait="--wait"; fi

			sbatch -a 0-$tries --exclusive --ntasks-per-node $t --exclusive ${wait} --job-name $jobname --output output/$jobname-%j.out -N 1 experiment.job
		done
	done
done
