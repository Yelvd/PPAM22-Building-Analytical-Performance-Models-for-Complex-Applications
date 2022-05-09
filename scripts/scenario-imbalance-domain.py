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
fig_name= "scenario_imbalance_domain"
datadir = "../results/{}/imbalance-domain/output/"
iterations = 500

model_das  = {"collideAndStream": {"offset": 0.007965768126115606, "N": 2.5082866244023706e-07}, "setExternalVector": {"offset": -0.0002188578044320071, "N": 2.1235687904379402e-08}, "collideAndStream_comm": {"offset": 0.0009439301155141662, "area": 2.2171490592454496e-07}, "syncEnvelopes_comm": {"offset": 0.00045566975961193495, "area": 1.2851244103688774e-08, "RBCs": 9.493552872348104e-06}, "syncEnvelopes": {"offset": 9.24827169270812e-05, "RBCs": 3.599652199729481e-05}, "advanceParticles": {"offset": 0.0005929407189597003, "RBCs": 8.153308088650628e-05}, "applyConstitutiveModel": {"offset": -4.432855061471974e-05, "RBCs": 2.826877556655128e-05}, "deleteNonLocalParticles": {"offset": 2.095287766730597e-05, "RBCs": 7.4864373485193265e-06}, "spreadParticleForce": {"offset": 0.0012118130950354064, "RBCs": 0.00025403341035855966}, "interpolateFluidVelocity": {"offset": 0.0003094544304245024, "RBCs": 4.183950985174891e-05}}
model_snel = {"collideAndStream": {"offset": 0.006203986032170888, "N": 3.48500278199325e-07}, "setExternalVector": {"offset": 2.590788867295757e-05, "N": 4.3112912128790356e-08}, "collideAndStream_comm": {"offset": -0.0004748026755225183, "area": 9.05354320891726e-07}, "syncEnvelopes_comm": {"offset": 0.00048311863052151715, "area": 1.2670340585585472e-07, "RBCs": 3.5270019874035664e-05}, "syncEnvelopes": {"offset": -1.376780408857048e-05, "RBCs": 8.277622850171989e-05}, "advanceParticles": {"offset": 0.0004938208025793962, "RBCs": 0.00013978737497846716}, "applyConstitutiveModel": {"offset": -1.3439321724241892e-05, "RBCs": 4.364834747587415e-05}, "deleteNonLocalParticles": {"offset": 5.103523368379448e-05, "RBCs": 1.4884632423801813e-05}, "spreadParticleForce": {"offset": 0.0008050285810133388, "RBCs": 0.00039740608639334083}, "interpolateFluidVelocity": {"offset": 0.000130060575853455, "RBCs": 7.700985333371006e-05}} 

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

def print_latex_table(analysis_df, models, machines=['das6', 'snellius']):
    analysis_df[0] = analysis_df[0].loc[analysis_df[0]['i'] == '1']
    analysis_df[0] = analysis_df[0].loc[analysis_df[0]['component'] == 'total']
    analysis_df[0] = analysis_df[0].loc[analysis_df[0].groupby('jobid', sort=False)['total'].idxmax()]

    analysis_df[1] = analysis_df[1].loc[analysis_df[1]['i'] == '1']
    analysis_df[1] = analysis_df[1].loc[analysis_df[1]['component'] == 'total']
    analysis_df[1] = analysis_df[1].loc[analysis_df[1].groupby('jobid', sort=False)['total'].idxmax()]


    for i, m in enumerate(machines):
        for j, H in enumerate(np.sort(pd.unique(analysis_df[i]['H']))):
            tmpstr = m + "&"
            tmp = analysis_df[i].loc[analysis_df[i]['H'] == H]

            model_res_naive = run_model(models[i], (100, 100, 100), np.array(tmp['RBCs'])[0] * 2, iterations)
            model_res_3 = run_model(models[i], (100, 100, 50),  np.array(tmp['RBCs'])[0], iterations)

            pred = model_res_naive['total']
            res = np.mean(tmp['total'])
            std = np.std(tmp['total'])
            err = np.abs(pred - res) * (100 / res)
            tmpstr += "{}\% ".format(H)
            tmpstr += "& $\\num{{{0:.2f}}}".format(res)
            tmpstr += "\pm \\num{{{0:.2f}}}$".format(std)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(err)


            pred = model_res_3['total'] * 3
            err = np.abs(pred - res) * (100 / res)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred)
            tmpstr += "& $\\num{{{0:.2f}}}$\\\\".format(err)
            print(tmpstr)


def plot_analysis(analysis_df, models, machines=['das6', 'snellius']):

    analysis_df[0] = analysis_df[0].loc[analysis_df[0]['i'] == '1']
    analysis_df[0] = analysis_df[0].loc[analysis_df[0]['component'] == 'total']
    analysis_df[0] = analysis_df[0].loc[analysis_df[0].groupby('jobid', sort=False)['total'].idxmax()]

    print(analysis_df[0])

    analysis_df[1] = analysis_df[1].loc[analysis_df[1]['i'] == '1']
    analysis_df[1] = analysis_df[1].loc[analysis_df[1]['component'] == 'total']
    analysis_df[1] = analysis_df[1].loc[analysis_df[1].groupby('jobid', sort=False)['total'].idxmax()]

    print(analysis_df[0])

    width = 0.15
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(2, 1, 1)
    exp = []

    for i, m in enumerate(machines):

        offset = -width
        legend_handels = []
        for j, H in enumerate(np.sort(pd.unique(analysis_df[i]['H']))):
            tmp = analysis_df[i].loc[analysis_df[i]['H'] == H]

            model_res_naive = run_model(models[i], (100, 100, 100), np.array(tmp['RBCs'])[0] * 2, iterations)
            model_res   = run_model(models[i], (150,100,100), np.array(tmp['RBCs'])[0] * 3, iterations)
            model_res_3 = run_model(models[i], (100, 100, 50),  np.array(tmp['RBCs'])[0], iterations)
            # model_res = run_model(model, np.array(tmp_imbalance['size'])[0], np.array(tmp_balance_18['RBCs'])[0], mult=3)

            plt.errorbar(i * len(machines) + j, np.mean(tmp['total']), yerr=np.std(tmp['total']), ms=30, color=CB_color_cycle[0], fmt=".", capsize=5, lw=1, zorder=1)
            plt.plot(i * len(machines) + j, model_res_naive['total'], ms=20, color=CB_color_cycle[1], marker="x", lw=0)
            plt.plot(i * len(machines) + j, model_res['total'], ms=20, color=CB_color_cycle[2], marker="x", lw=0)
            # plt.errorbar(i, model_res['total'], yerr=0, ms=30, color=CB_color_cycle[0], fmt="^", capsize=5, lw=1)

            offset = width
            # legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[j], lw=0, marker='.', ms=20, label='H{}%'.format(H)))
            exp.append("{}: H{}\%".format(m, H))

    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[0], lw=0, marker='.', ms=10, label='Results imbalance'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='x', ms=10, label='Prediction naive'))
    # legend_handels.insert(0, Line2D([0], [0], color='k', lw=0, marker='^', ms=20, label='Prediction big subdomain'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[2], lw=0, marker='x', ms=10, label='Prediction x3'))


    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.weight': 'bold'})

    plt.legend(handles=legend_handels)
    plt.ylim(0)
    # plt.xlim(-.5, 1.5)
    plt.ylabel("Time in Seconds")
    plt.xlabel("processes")
    plt.title("Hematocrit imbalance snellius (1 node, 128 processes)")
    # plt.xticks(range(np.unique(np.sort(analysis_df['H'])).size), [ "H - {}".format(x)  for x in np.sort(pd.unique(analysis_df['H']))])
    plt.xticks(np.arange(len(exp)), exp, rotation=30)  # Set text labels and properties.
    # plt.xticks(range(machines), machines)
    plt.savefig(results_dir + fig_name + ".pdf", bbox_inches='tight')


if __name__ == "__main__":
    results_df_das, exp_df_das = model_generation.load_data(datadir.format("das6"))
    results_df_snel, exp_df_snel = model_generation.load_data(datadir.format("snellius"))
    analysis_df_das = pd.merge(results_df_das, exp_df_das, on=['jobid'], how="left")
    analysis_df_snel = pd.merge(results_df_snel, exp_df_snel, on=['jobid'], how="left")

    plot_analysis([analysis_df_das, analysis_df_snel], [model_das, model_snel])
    print_latex_table([analysis_df_das, analysis_df_snel], [model_das, model_snel])
