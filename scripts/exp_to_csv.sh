#!/bin/bash

dir="${1}/"

cd $dir
dir_names="$(ls -d */)"

for d in $dir_names; do
    echo ${d}
    echo ${d[@]:7:-1}
    square -s ${d}
    cube_cut -o pruned_tree -r "void hemo::HemoCell::iterate()" -o ${d}pruned_tree.cubex ./${d}summary.cubex
    cube_dump -c level=0 -m comp,mpi,execution -s csv2 -o ${d[@]:7:-1}.csv -z incl ./${d}pruned_tree.cubex
    cube_dump -c level=1 -m comp,mpi,execution -s csv2 -z incl ./${d}pruned_tree.cubex | tail -n +2 >> ${d[@]:7:-1}.csv 
done
