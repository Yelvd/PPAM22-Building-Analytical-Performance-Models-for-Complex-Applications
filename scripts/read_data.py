import pandas as pd
import glob
import re
import os
import numpy as np

"""
This script is used to load the SCORP and CSV experiment data into a pandas dataframe
The files must adhere to a few file name rules
Experiment-name: name_of_exp-param1_value1-param2_value2-jobid111

csv files are stored as experiment-name.csv
scorep files as SCOREP-experiment-name/pruned_tree.cubex

All files in a single dir must have the same paramas in the same order
"""


def get_exp_info(filename):
    """
        Extract info from Hemocell.out files
        return as a dict of relevant info
    """
    
    RBCs = -1
    with open(filename.replace(".csv", ".out"),"r") as file:
        for line in file:
            if re.search("\(main\)   nCells \(global\)", line):
                RBCs = int(line.split()[-1])
                break
            if re.search("Number of atomic-blocks:", line):
                atomicblocks = int(line.split()[-1])
            if re.search("tasks for job:", line):
                tasks = int(line.split()[-1])
            if re.search("Smallest atomic-block:", line):
                 min_size = line.strip().split(" ")[-1].split('-')
                 min_size = (int(min_size[0]), int(min_size[2]), int(min_size[4]))
            if re.search("Largest atomic-block:", line):
                 max_size = line.strip().split(" ")[-1].split('-')
                 max_size = (int(max_size[0]), int(max_size[2]), int(max_size[4]))
            if re.search("\(SanityCheck\)Hemocell Profiler Statistics:", line):
                break

    if RBCs == -1:
        print("{} not correct".format(filename))
        return None
    
    return {"RBCs": RBCs, "atomicblocks": atomicblocks, "tasks": tasks, "smallest_subdomain": min_size, "largest_subdomain": max_size}

    

def gen_df(datadir, components=["total", "spreadParticleForce", "collideAndStream", "interpolateFluidVelocity", "syncEnvelopes", "advanceParticles", "applyConstitutiveModel", "deleteNonLocalParticles", "setExternalVector"]):
    """
        gen_df creats two dataframes, that contain all experiment info.
        
        return:
        data-frame with all experiment results
        data-frame with the parameters per jobid
    """
    # get all csv files
    datafiles = glob.glob(datadir + "/*.csv")
 
    # get the cnode ids to map to the components
    df = pd.read_csv(datafiles[0])
    df.columns =[col.strip() for col in df.columns]
    cnodes = np.unique(df['Cnode ID'])
    cnodes.sort()

    # Get the list of parameter names
    filename = os.path.splitext(os.path.split(datafiles[0])[-1])[0]
    split = filename.split('-')
    jobid=split[-1]
    split = split[1:-1]
    params = [x.split('_')[0] for x in split]

    # Map cnode ids to component names
    cnode_translate = {}
    for i, cn in enumerate(cnodes):
        cnode_translate[cn] = components[i]

    data = []
    resultsdf = None

    # for all experiments, read the csv as dataframe and combine all those
    # Add the experiment info to other dataframe
    for file in datafiles:
        fileinfo = get_exp_info(file)
        if fileinfo is None:
            continue

        # Get filename without path or extention
        filename = os.path.splitext(os.path.split(file)[-1])[0]
        split = filename.split('-')
        jobid=split[-1]
        split = split[1:-1]

        df = pd.read_csv(file)
        df.columns =[col.strip() for col in df.columns]
        df.insert(0, 'jobid', jobid)
        if len(df["Cnode ID"]) == 0:
            print("{}: empty csv skipping".format(file))
            continue

        if resultsdf is None:
            resultsdf = df
        else:
            resultsdf = pd.concat([resultsdf, df], ignore_index=True)

        data.append([jobid] + list(fileinfo.values()) + [x.split('_')[-1] for x in split])

    
    resultsdf['Cnode ID'] = [cnode_translate[x] for x in resultsdf['Cnode ID']]
    resultsdf.columns = ["jobid", "component", "threadid", "total", "comm", "comp"]

    return resultsdf, pd.DataFrame(data, columns=["jobid"] + list(fileinfo.keys()) + params)


#res, job = gen_df("/home/jelle/SurfDrive/Hemocell/model/imbalance-hemo/snallius-old/output/")

#print(res)
#print("-----")
#print(job)
