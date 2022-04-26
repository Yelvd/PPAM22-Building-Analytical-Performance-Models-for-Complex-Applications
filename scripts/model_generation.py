# %matplotlib widget
import getopt
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob
from scipy import stats
from scipy.optimize import curve_fit
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import read_data
from scipy.stats import itemfreq
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import subprocess
import json

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams.update({'font.size': 15})

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

results_dir = "../figs/"
machine=""
fig_name= "model_prediction_{}"
iterations = 1

def run_model(models, size, RBCs):
    total = 0
    for component in ["collideAndStream", "setExternalVector"]:
        res = models[component]
        total += res.intercept + res.slope *  size[0] * size[1] * size[2]
    
    for component in ["collideAndStream_comm"]:
        res = models[component]
        total += res.intercept + res.slope *  2*((size[0]*size[1])+(size[0]*size[2])+(size[1]*size[2]))
        
    for component in ["syncEnvelopes", "advanceParticles", "applyConstitutiveModel", "deleteNonLocalParticles", "spreadParticleForce", "interpolateFluidVelocity"]:
        res = models[component]
        total += res.intercept + res.slope * RBCs
        
    for component in ["syncEnvelopes_comm"]:
        popt = models[component]
        total += popt[0] + 2*((size[0]*size[1])+(size[0]*size[2])+(size[1]*size[2]))*popt[1] + RBCs*popt[2]
    
    return total


def calibrate_model(fitting_df):
    """ Calibrate the model by linear fitting the data to the matched relevant parameter(s) """

    models = {}

    for component in ["collideAndStream", "setExternalVector"]:
        tmpdf = fitting_df.loc[fitting_df['component'] == component]
        
        res = stats.linregress(tmpdf['N'], tmpdf['comp'])
        models[component] = res 

    for component in ["collideAndStream"]:
        tmpdf = fitting_df.loc[fitting_df['component'] == component]    

        res = stats.linregress(tmpdf['area'], tmpdf['comm'])
        models[component+"_comm"] = res

        
    for component in ["syncEnvelopes", "applyConstitutiveModel", "deleteNonLocalParticles", "spreadParticleForce", "interpolateFluidVelocity", "advanceParticles"]:
        tmpdf = fitting_df.loc[fitting_df['component'] == component]

        res = stats.linregress(tmpdf['RBCs'], tmpdf['comp'])
        models[component] = res

    def function_calc(x, a, b, c):
        return a + b*x[0] + c*x[1]
        
    for component in ["syncEnvelopes"]:
        tmpdf = fitting_df.loc[fitting_df['component'] == component]    
        
        popt, pcov = curve_fit(function_calc, [tmpdf['area'],tmpdf['RBCs']], tmpdf['comm'])
        models[component+"_comm"] = popt
    
    return models

def model_to_json(models):
    """ Convert the model given by calibrate_model to a json style model """
    smodel = {}

    for k in models.keys():
        for component in ["collideAndStream", "setExternalVector"]:
            res = models[component]
            smodel[component] = {'offset': res.intercept, 'N': res.slope}
        
        for component in ["collideAndStream_comm"]:
            res = models[component]
            smodel[component] = {'offset': res.intercept, 'area': res.slope}

        for component in ["syncEnvelopes_comm"]:
            res = models[component]
            smodel[component] = {'offset': res[0], 'area': res[1], 'RBCs': res[2]}
            
        for component in ["syncEnvelopes", "advanceParticles", "applyConstitutiveModel", "deleteNonLocalParticles", "spreadParticleForce", "interpolateFluidVelocity"]:
            res = models[component]
            smodel[component] = {'offset': res.intercept, 'RBCs': res.slope}


    for k in smodel.keys():
        for x in smodel[k].keys():
            smodel[k][x] = smodel[k][x] / iterations

    return smodel


def load_data(resultsdir):
    # Read Data
    results_df, exp_df = read_data.gen_df(resultsdir)

    exp_df['N'] = [x * y * z for (x, y, z) in exp_df['largest_subdomain']]
    exp_df['area'] = [ 2*(x*y + x*z + y*z)  for (x, y, z) in exp_df['largest_subdomain']]
    exp_df['RBCs-total'] = exp_df["RBCs"]
    exp_df['RBCs'] = exp_df['RBCs-total'] / exp_df['atomicblocks']

    return results_df, exp_df

def unique(x, axis=0):
    seen = []
    new = []
    
    for tmp in x:
        if tmp[axis] not in seen:
            seen.append(tmp[axis])
            new.append(tmp)
    return np.sort(np.array(new, dtype=object), axis=0)


def plot_validation(testing_df, model, name):
    """ Plot the results of prediction the testing_df in results_dir + name + .pdf """
    # grap the longest running thread as the total running time of the application
    testing_df = testing_df.loc[testing_df['component'] == 'total']
    testing_df = testing_df.loc[testing_df.groupby('jobid', sort=False)['total'].idxmax()]
    testing_df['sizestr'] = ["({}, {}, {})".format(x, y, z) for (x, y, z) in testing_df['largest_subdomain']]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(2, 1, 1)

    fit_exp_df = testing_df.sort_values("N")
    width = .7
    stride = width / np.unique(np.sort(fit_exp_df['h'])).size

    for i, sizestr in enumerate(pd.unique(fit_exp_df['sizestr'])):
        s = np.array(fit_exp_df.loc[fit_exp_df['sizestr'] == sizestr]['largest_subdomain'])[0]
        
        legend_handels = []
        offset = width/2
        for hi, H in enumerate(np.unique(np.sort(fit_exp_df['h']))):
            t = fit_exp_df.loc[fit_exp_df['sizestr'] == sizestr]
            r = np.array(t.loc[t['h'] == H]['RBCs'])[0]

            tmp = testing_df.loc[testing_df['h'] == H]
            tmp = tmp.loc[tmp['largest_subdomain'] == s]
            
            plt.errorbar(i - offset, np.mean(tmp['total']), yerr=np.std(tmp['total']), ms=30, color=CB_color_cycle[hi], fmt=".", capsize=5, lw=1)
            plt.plot(i - offset, run_model(model, s, r), 'x', color=CB_color_cycle[hi], ms=20)
            
            offset -= stride
            legend_handels.append( Line2D([0], [0], color=CB_color_cycle[hi], lw=0, marker='o', ms=10, label='H{}\%'.format(H)))
            # legend_handels.append( Line2D([0], [0], color=CB_color_cycle[hi], lw=0, marker='x', label='H{}\ Prediction%'.format(H)))

    legend_handels.insert(0, Line2D([0], [0], color='k', lw=0, marker='x', ms=10, label='Model Prediction'.format(H)))        
    legend_handels.insert(0, Line2D([0], [0], color='k', lw=0, marker='o', ms=10, label='Empirical Results'.format(H)))

    # plt.grid(True, which="both", ls="-", color='0.65')

    plt.rcParams.update({'font.size': 20})
    # plt.rcParams.update({'axes.linewidth': 5})
    plt.rcParams.update({'font.weight': 'bold'})
    # plt.rcParams.update({'font.size': 20})

    plt.legend(handles=legend_handels, loc='lower right')
    ax.set_yscale('log')
    plt.ylim(0.01, 15 ** 3) 
    plt.ylabel("Time in Seconds")
    plt.xlabel("Domain Size in µm (x, y, z)")
    # plt.title("Model Verification on DAS-6 (1 node, 24 processes)")
    # plt.xticks(, pd.unique(fit_exp_df['sizestr']), rotation='vertical')
    plt.xticks(range(np.unique(np.sort(fit_exp_df['largest_subdomain'])).size), [ "({:g}, {:g}, {:g})".format(0.5 * x[0], 0.5 * x[1],0.5 * x[2])  for x in pd.unique(fit_exp_df['largest_subdomain'])])
    plt.tight_layout()
    plt.savefig(results_dir + name + ".pdf", bbox_inches='tight')
    # plt.savefig(results_dir + "model-prediction_das6.svg", bbox_inches='tight')
    # plt.savefig(results_dir + "model-prediction_das6.pdf", bbox_inches='tight')

def gen_validation_table(testing_df, model):
    """Print latex table for the results and prediction of the testing set
    size & hematocrit &  result +- std & prediction & prediction error """

    # grap the longest running thread as the total running time of the application
    testing_df = testing_df.loc[testing_df['component'] == 'total']
    testing_df = testing_df.loc[testing_df.groupby('jobid', sort=False)['total'].idxmax()]
    print(testing_df)
    testing_df['sizestr'] = ["({}, {}, {})".format(x, y, z) for (x, y, z) in testing_df['largest_subdomain']]

    fit_exp_df = testing_df.sort_values("N")

    for i, sizestr in enumerate(pd.unique(fit_exp_df['sizestr'])):
        s = np.array(fit_exp_df.loc[fit_exp_df['sizestr'] == sizestr]['largest_subdomain'])[0]
        
        for hi, H in enumerate(np.unique(np.sort(fit_exp_df['h']))):
            t = fit_exp_df.loc[fit_exp_df['sizestr'] == sizestr]
            r = np.array(t.loc[t['h'] == H]['RBCs'])[0]

            tmp = testing_df.loc[testing_df['h'] == H]
            tmp = tmp.loc[tmp['largest_subdomain'] == s]
            
            pred = run_model(model, s, r)
            res = np.mean(tmp['total'])
            std = np.std(tmp['total'])
            err = np.abs(pred - res) * (100 / res)
            st = np.array(tmp["largest_subdomain"])[0]
            stmp = "({}, {}, {})".format(st[0] *.5, st[1] *.5, st[2] *.5)
            tmpstr = "{} & {}\% ".format(stmp,  H, )
            tmpstr += "& $\\num{{{0:.2f}}}".format(res)
            tmpstr += "\pm \\num{{{0:.2f}}}".format(std)
            tmpstr += "$& $\\num{{{0:.2f}}}".format(pred)
            tmpstr += "$ & $\\num{{{0:.2f}}}$\\\\".format(err)
            print(tmpstr)

def print_model_latex(model):
    model = model_to_json(model)

    for k in model.keys():
        tmpstr = '{} & '.format(k)

        tmpstr += "$"
        tmpstr += "\\num{{{0:.2g}}}".format(model[k]['offset'])
        if 'N' in model[k].keys():
            tmpstr += " + V \\times \\num{{{0:.2g}}}".format(model[k]['N'])
        if 'area' in model[k].keys():
            tmpstr += " + SA \\times \\num{{{0:.2g}}}".format(model[k]['area'])
        if 'RBCs' in model[k].keys():
            tmpstr += " + RBCs \\times \\num{{{0:.2g}}}".format(model[k]['RBCs']) 
        tmpstr += "$\\\\"
        print(tmpstr)
    pass

def validate_model(testing_df, model):
    gen_validation_table(testing_df, model)
    plot_validation(testing_df, model, fig_name)

def main(argv):
    global fig_name
    global machine
    global results_dir
    global iterations
    datadir = ""
    try:
        opts, args = getopt.getopt(argv,"m:r:o:ds",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-o"):
            results_dir = arg
        elif opt in ("-d"):
            machine = "das6"
            iterations = 1000
        elif opt in ("-s"):
            machine = "snellius" 
            iterations = 500
        elif opt in ("-r"):
            datadir = arg

    fig_name = fig_name.format(machine) 
    datadir  = "../results/{}/model/results/".format(machine)
    if datadir == "":
        print("no data dir given")
        sys.exit(1)
    if machine == "":
        print("no machine name given")
        sys.exit(1)

    results_df, exp_df = load_data(datadir)

    if machine == "das6":
        fitting_sizes = ["s5", "s1", "s3", "s6", "s7", 's8']
        testing_sizes = ["s2", "s4", "s8", "s9"]
        # testing_sizes = testing_sizes + fitting_sizes
    if machine == "snellius":
        fitting_sizes = ["s1", "s3", "s5", "s8", "s9", "s11", "s12", 's10']
        testing_sizes = ["s2", "s4", "s7", "s14"]

    # testing_sizes = testing_sizes + fitting_sizes
    
    print(np.unique(exp_df['s']))

    fitting_jobs = exp_df.loc[exp_df['s'].isin(fitting_sizes)]['jobid']
    testing_jobs = exp_df.loc[exp_df['s'].isin(testing_sizes)]['jobid']

    fitting_df = results_df.loc[results_df['jobid'].isin(fitting_jobs)]
    testing_df = results_df.loc[results_df['jobid'].isin(testing_jobs)]

    fitting_df = pd.merge(fitting_df, exp_df, on=['jobid'], how="left")
    testing_df = pd.merge(testing_df, exp_df, on=['jobid'], how="left")

    print(fitting_df)

    model = calibrate_model(fitting_df)
    validate_model(testing_df, model)

    print()
    print(json.dumps(model_to_json(model)))
    print()
    print_model_latex(model)


if __name__ == "__main__":
    main(sys.argv[1:])