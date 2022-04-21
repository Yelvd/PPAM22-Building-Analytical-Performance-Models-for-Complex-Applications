import model_generation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams.update({'font.size': 15})

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

results_dir = "../figs/"
fig_name= "scenario_imbalance_hemo_das6"
datadir = "../results/das6/imbalance-hemo/output/"
iterations = 500
large_hemo = 250

model = {"collideAndStream": {"offset": 0.007182424664883783, "N": 2.4337972047266724e-07}, "setExternalVector": {"offset": -0.0001567977687708435, "N": 2.1377646855844923e-08}, "collideAndStream_comm": {"offset": 0.0009894284527065694, "area": 2.0933177473340455e-07}, "syncEnvelopes": {"offset": 0.00011488342514662909, "RBCs": 3.408385367040535e-05}, "advanceParticles": {"offset": 0.0005015929210854835, "RBCs": 8.077362825627717e-05}, "applyConstitutiveModel": {"offset": -3.1492459389115265e-05, "RBCs": 2.790861572448882e-05}, "deleteNonLocalParticles": {"offset": 9.345415747055719e-06, "RBCs": 7.490896642043115e-06}, "spreadParticleForce": {"offset": 0.0010806997738769582, "RBCs": 0.00025126073806962414}, "interpolateFluidVelocity": {"offset": 0.0002598888412672542, "RBCs": 4.087327524040482e-05}, "syncEnvelopes_comm": {"offset": 0.0004055586584863523, "RBCs": 9.453519478319318e-06, "area": 1.4773915000808698e-08}}

particle_components = ['syncEnvelopes', 'syncEnvelopes_comm', 'advanceParticles', "applyConstitutiveModel", "deleteNonLocalParticles"]
coupling_components = ["spreadParticleForce", "interpolateFluidVelocity"]
fluid_components    = ["collideAndStream", "collideAndStream_comm", "setExternalVector"]


def run_model(models, size, RBCs, iterations=1, fluid_keys=fluid_components, particle_keys=particle_components, coupling_keys=coupling_components):
    N = size[0] * size[1] * size[2]
    area = 2*((size[0]*size[1])+(size[0]*size[2])+(size[1]*size[2]))
    results = {'total': 0}
    
    if fluid_keys != []:
        results['fluid'] =  0
    
    if particle_keys != []:
        results['particle'] = 0
        
    if coupling_keys != []:
        results['coupling'] = 0
        
    for component in models.keys():
        
        m = models[component]
        results[component] = m['offset'] * iterations
        results['total'] += m['offset'] * iterations
        if component in fluid_keys:
            results['fluid'] += m['offset'] * iterations

        if component in particle_keys:
            results['particle'] += m['offset'] * iterations

        if component in coupling_keys:
            results['coupling'] += m['offset'] * iterations

        keys = m.keys()
        if 'N' in keys:
            tmp_value = m['N'] * N * iterations
            results[component] += tmp_value
            results['total'] += tmp_value
            
            if component in fluid_keys:
                results['fluid'] += tmp_value

            if component in particle_keys:
                results['particle'] += tmp_value

            if component in coupling_keys:
                results['coupling'] += tmp_value

            
        if 'area' in keys:
            tmp_value = m['area'] * area * iterations
            results[component] += tmp_value
            results['total'] += tmp_value

            if component in fluid_keys:
                results['fluid'] += tmp_value

            if component in particle_keys:
                results['particle'] += tmp_value

            if component in coupling_keys:
                results['coupling'] += tmp_value

        if 'RBCs' in keys:
            tmp_value = m['RBCs'] * RBCs * iterations
            results[component] += tmp_value
            results['total'] += tmp_value

            if component in fluid_keys:
                results['fluid'] += tmp_value

            if component in particle_keys:
                results['particle'] += tmp_value

            if component in coupling_keys:
                results['coupling'] += tmp_value

    return results

def unique(x, axis=0):
    seen = []
    new = []

    for tmp in x:
        if tmp[axis] not in seen:
            seen.append(tmp[axis])
            new.append(tmp)
    return np.sort(np.array(new, dtype=object), axis=0)


def plot_analysis(analysis_df, model, total=True):

    if total:
        analysis_df = analysis_df.loc[analysis_df['component'] == 'total']
        analysis_df = analysis_df.loc[analysis_df.groupby('jobid', sort=False)['total'].idxmax()]
        analysis_df['sizestr'] = ["({}, {}, {})".format(x, y, z) for (x, y, z) in analysis_df['largest_subdomain']]
    else:
        new_data = []
        for i in np.unique(analysis_df['jobid']):
            tmp = analysis_df.loc[analysis_df['jobid'] == i]
            idx = tmp.loc[tmp['component'] == 'advanceParticles']['total'].idxmax()
            tmp = tmp.loc[tmp['threadid'] == tmp.loc[idx]['threadid']]
            tmp = tmp.loc[tmp['component'].isin(particle_components + coupling_components)]
            new_data.append([np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], np.sum(tmp['total']), np.array(tmp['H'])[0]])
            
        analysis_df = pd.DataFrame(new_data, columns=['largest_subdomain', 'RBCs', 'total', 'H'])


    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(2, 1, 1)
    for i, H in enumerate(pd.unique(analysis_df['H'])):
        tmp = analysis_df.loc[analysis_df['H'] == H]
        
        if total:
            model_res_naive = run_model(model, np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], iterations)
            model_res = run_model(model, np.array(tmp['largest_subdomain'])[0], large_hemo, iterations)
            plt.errorbar(i, np.mean(tmp['total']), yerr=np.std(tmp['total']), ms=30, color=CB_color_cycle[0], fmt=".", capsize=5, lw=1)
            plt.errorbar(i, model_res_naive['total'], yerr=0, ms=30, color=CB_color_cycle[1], fmt="x", capsize=5, lw=1)
            plt.errorbar(i, model_res['total'], yerr=0, ms=30, color=CB_color_cycle[2], fmt="x", capsize=5, lw=1)
        else:
            model_res_naive = run_model(model, np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], iterations)
            model_res = run_model(model, np.array(tmp['largest_subdomain'])[0], large_hemo, iterations)
            plt.errorbar(i, np.mean(tmp['total']), yerr=np.std(tmp['total']), ms=30, color=CB_color_cycle[0], fmt=".", capsize=5, lw=1)
            plt.errorbar(i, model_res_naive['particle'] +  model_res_naive['coupling'], yerr=0, ms=30, color=CB_color_cycle[1], fmt="x", capsize=5, lw=1)
            plt.errorbar(i, model_res['particle'] + model_res['coupling'], yerr=0, ms=30, color=CB_color_cycle[2], fmt="x", capsize=5, lw=1)


    legend_handels = []
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[0], lw=0, marker='.', ms=20, label='Results imbalance'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='x', ms=20, label='Prediction balanced'))        
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[2], lw=0, marker='x', ms=20, label='Prediction unbalanced'))

    plt.rcParams.update({'font.size': 20})
    # plt.rcParams.update({'axes.linewidth': 5})
    plt.rcParams.update({'font.weight': 'bold'})
    # plt.rcParams.update({'font.size': 20})

    plt.legend(handles=legend_handels)
    # plt.legend()
    # ax.set_yscale('log')
    plt.ylim(10)
    plt.ylabel("Time in Seconds")
    plt.xlabel("processes")
    plt.title("Hematocrit imbalance snellius (1 node, 128 processes)")
    plt.xticks(range(np.unique(np.sort(analysis_df['H'])).size), [ "H - {}".format(x)  for x in pd.unique(analysis_df['H'])])
    plt.tight_layout()
    plt.savefig(results_dir + fig_name + "-{}".format("total" if total else "partial") + ".pdf", bbox_inches='tight')


if __name__ == "__main__":
    results_df, exp_df = model_generation.load_data(datadir)
    analysis_df = pd.merge(results_df, exp_df, on=['jobid'], how="left")

    print(analysis_df)

    plot_analysis(analysis_df, model, True)
