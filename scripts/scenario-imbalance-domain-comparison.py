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
fig_name= "scenario_imbalance_domain_comparison"
datadir = "../results/{}/imbalance-domain/output/"
datadir_bal = "../results/{}/model/output/"
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

def print_latex_table(analysis_df, balanced_df, models, machines=['das6', 'snellius']):
    for i, _ in enumerate(machines):
        analysis_df[i] = analysis_df[i].loc[analysis_df[i]['i'] == '1']
        analysis_df[i] = analysis_df[i].loc[analysis_df[i]['component'] == 'total']
        analysis_df[i] = analysis_df[i].loc[analysis_df[i].groupby('jobid', sort=False)['total'].idxmax()]

        balanced_df[i] = balanced_df[i].loc[balanced_df[i]['component'] == 'total']
        balanced_df[i] = balanced_df[i].loc[balanced_df[i].groupby('jobid', sort=False)['total'].idxmax()]



    for i, m in enumerate(machines):
        for j, H in enumerate(np.sort(pd.unique(analysis_df[i]['H']))):
            tmpstr = m + "&"
            tmp = analysis_df[i].loc[analysis_df[i]['H'] == H]
            tmp_balanced = balanced_df[i].loc[balanced_df[i]['h'] == '0{}'.format(H)]
            
            model_res_naive = run_model(models[i], (100, 100, 100), np.array(tmp_balanced['RBCs'])[0], iterations)
            model_res_3 = run_model(models[i], (100, 100, 50),  np.array(tmp['RBCs'])[0], iterations)

            pred = model_res_3['total'] * 3
            res = np.mean(tmp['total'])
            std = np.std(tmp['total'])
            err = np.abs(pred - res) * (100 / res)

            pred_balanced = model_res_naive['total']
            res_balanced = np.mean(tmp_balanced['total'])
            std_balanced = np.std(tmp_balanced['total'])
            err_balanced = np.abs(pred_balanced - res_balanced) * (100 / res_balanced)

            res_diff = np.abs(res_balanced - res) * (100 / res_balanced)
            pred_diff = np.abs(pred_balanced - pred) * (100 / pred_balanced)
            pred_diff_err = np.abs(res_diff - pred_diff) * (100 / res_diff)

            tmpstr += "{}\% ".format(H)
            tmpstr += "& $\\num{{{0:.2f}}}".format(res_balanced)
            tmpstr += "\pm \\num{{{0:.2f}}}$".format(std_balanced)
            tmpstr += "& $\\num{{{0:.2f}}}".format(res)
            tmpstr += "\pm \\num{{{0:.2f}}}$".format(std)

            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred_balanced)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(err_balanced)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(err)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(res_diff)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred_diff)
            # tmpstr += "& $\\num{{{0:.2f}}}$".format(pred_diff_err)

            tmpstr += '\\\\'

            print(tmpstr)


def plot_analysis(analysis_df, balanced_df, models, machines=['das6', 'snellius']):

    for i, _ in enumerate(machines):
        analysis_df[i] = analysis_df[i].loc[analysis_df[i]['i'] == '1']
        analysis_df[i] = analysis_df[i].loc[analysis_df[i]['component'] == 'total']
        analysis_df[i] = analysis_df[i].loc[analysis_df[i].groupby('jobid', sort=False)['total'].idxmax()]

        balanced_df[i] = balanced_df[i].loc[balanced_df[i]['component'] == 'total']
        balanced_df[i] = balanced_df[i].loc[balanced_df[i].groupby('jobid', sort=False)['total'].idxmax()]

    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.weight': 'bold'})

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    

    exp = []

    for i, m in enumerate(machines):
        for j, H in enumerate(np.sort(pd.unique(analysis_df[i]['H']))):
            tmpstr = m + "&"
            tmp = analysis_df[i].loc[analysis_df[i]['H'] == H]
            tmp_balanced = balanced_df[i].loc[balanced_df[i]['h'] == '0{}'.format(H)]
            
            model_res_naive = run_model(models[i], (100, 100, 100), np.array(tmp_balanced['RBCs'])[0], iterations)
            model_res_3 = run_model(models[i], (100, 100, 50),  np.array(tmp['RBCs'])[0], iterations)

            pred = model_res_3['total'] * 3
            res = np.mean(tmp['total'])
            std = np.std(tmp['total'])
            err = np.abs(pred - res) * (100 / res)

            pred_balanced = model_res_naive['total']
            res_balanced = np.mean(tmp_balanced['total'])
            std_balanced = np.std(tmp_balanced['total'])
            err_balanced = np.abs(pred_balanced - res_balanced) * (100 / res_balanced)

            res_diff = np.abs(res_balanced - res) * (100 / res_balanced)
            pred_diff = np.abs(pred_balanced - pred) * (100 / pred_balanced)

            legend_handels = []

            ax1.errorbar(i * len(machines) + j, np.mean(tmp_balanced['total']), yerr=np.std(tmp_balanced['total']), ms=30, color=CB_color_cycle[0], fmt=".", capsize=5, lw=1, zorder=1)
            ax1.errorbar(i * len(machines) + j, np.mean(tmp['total']), yerr=np.std(tmp['total']), ms=30, color=CB_color_cycle[1], fmt=".", capsize=5, lw=1, zorder=1)
            ax1.plot(i * len(machines) + j, model_res_naive['total'], ms=20, color=CB_color_cycle[0], marker="x", lw=0)
            ax1.plot(i * len(machines) + j, model_res_3['total'] * 3, ms=20, color=CB_color_cycle[1], marker="x", lw=0)

            pred_error = np.abs(model_res_naive['total'] - np.mean(tmp_balanced['total'])) * (100 / np.mean(tmp_balanced['total']))
            ax2.plot(i * len(machines) + j, pred_error, 'X', color=CB_color_cycle[0], ms=20)

            pred_error = np.abs(model_res_3['total'] * 3 - np.mean(tmp['total'])) * (100 / np.mean(tmp['total']))
            ax2.plot(i * len(machines) + j, pred_error, 'X', color=CB_color_cycle[1], ms=20)

            # legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[j], lw=0, marker='.', ms=20, label='H{}%'.format(H)))
            exp.append("{}\n H{}\%".format(m, H))

    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[0], lw=0, marker='.', ms=10, label='Results imbalance'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='.', ms=10, label='Results balanced'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[0], lw=0, marker='x', ms=10, label='Prediction balanced'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='x', ms=10, label='Prediction imbalanced'))
    # legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[0], lw=0, marker='x', ms=10, label='Prediction balanced'))
    # legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='x', ms=10, label='Prediction imbalanced'))

    legend_handels2 = [] 
    legend_handels2.insert(0, Line2D([0], [0], color=CB_color_cycle[0], lw=0, marker='X', ms=10, label='Balanced Prediction Error'))
    legend_handels2.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='X', ms=10, label='Imbalanced Prediction Error'))


    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.tick_params(axis='both', which='minor', labelsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='both', which='minor', labelsize=25)
    ax1.legend(handles=legend_handels, prop={'size': 15}) 
    ax2.legend(handles=legend_handels2, prop={'size': 15}) 
    ax1.set_ylim(0, 700)
    ax2.set_ylim(0, 30)
    # plt.xlim(-.5, 1.5)
    ax1.set_ylabel("Time in Seconds", fontsize=25)
    ax2.set_ylabel("Prediction error [\%]", fontsize=25)
    # plt.title("Hematocrit imbalance snellius (1 node, 128 processes)")
    # plt.xticks(range(np.unique(np.sort(analysis_df['H'])).size), [ "H - {}".format(x)  for x in np.sort(pd.unique(analysis_df['H']))])
    ax1.set_xticks(np.arange(len(exp)), exp, rotation=45, fontsize=25)  # Set text labels and properties.
    ax2.set_xticks(np.arange(len(exp)), exp, rotation=45, fontsize=25)  # Set text labels and properties.
    # plt.xticks(range(machines), machines)
    plt.savefig(results_dir + fig_name + ".pdf", bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    results_df_das, exp_df_das = model_generation.load_data(datadir.format("das6"))
    results_df_snel, exp_df_snel = model_generation.load_data(datadir.format("snellius"))
    analysis_df_das = pd.merge(results_df_das, exp_df_das, on=['jobid'], how="left")
    analysis_df_snel = pd.merge(results_df_snel, exp_df_snel, on=['jobid'], how="left")


    # Need to do the filtering based on jobid in reslts_df
    results_bal_df_das, exp_bal_df_das = model_generation.load_data(datadir_bal.format("das6"))
    exp_bal_df_das = exp_bal_df_das.loc[exp_bal_df_das['largest_subdomain'] == (100,100,100)]


    results_bal_df_snel, exp_bal_df_snel = model_generation.load_data(datadir_bal.format("snellius"))
    exp_bal_df_snel = exp_bal_df_snel.loc[exp_bal_df_snel['largest_subdomain'] == (100,100,100)]
    
    analysis_bal_df_das = pd.merge(results_bal_df_das, exp_bal_df_das, on=['jobid'], how="right")
    analysis_bal_df_snel = pd.merge(results_bal_df_snel, exp_bal_df_snel, on=['jobid'], how="right")

    # print(analysis_df_das)
    # plot_analysis([analysis_df_das, analysis_df_snel],  [model_das, model_snel])
    # print(analysis_bal_df_das)
    print_latex_table([analysis_df_das, analysis_df_snel], [analysis_bal_df_das, analysis_bal_df_snel], [model_das, model_snel])
    plot_analysis([analysis_df_das, analysis_df_snel], [analysis_bal_df_das, analysis_bal_df_snel], [model_das, model_snel])
