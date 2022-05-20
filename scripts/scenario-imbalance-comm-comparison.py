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
fig_name= "scenario_imbalance_comm_comparison"
datadir = "../results/{}/imbalance-comm/output/"
datadir_bal = "../results/{}/model/output/"
iterations = 500

model_das  = {"collideAndStream": {"offset": 0.007965768126115606, "N": 2.5082866244023706e-07}, "setExternalVector": {"offset": -0.0002188578044320071, "N": 2.1235687904379402e-08}, "collideAndStream_comm": {"offset": 0.0009439301155141662, "area": 2.2171490592454496e-07}, "syncEnvelopes_comm": {"offset": 0.00045566975961193495, "area": 1.2851244103688774e-08, "RBCs": 9.493552872348104e-06}, "syncEnvelopes": {"offset": 9.24827169270812e-05, "RBCs": 3.599652199729481e-05}, "advanceParticles": {"offset": 0.0005929407189597003, "RBCs": 8.153308088650628e-05}, "applyConstitutiveModel": {"offset": -4.432855061471974e-05, "RBCs": 2.826877556655128e-05}, "deleteNonLocalParticles": {"offset": 2.095287766730597e-05, "RBCs": 7.4864373485193265e-06}, "spreadParticleForce": {"offset": 0.0012118130950354064, "RBCs": 0.00025403341035855966}, "interpolateFluidVelocity": {"offset": 0.0003094544304245024, "RBCs": 4.183950985174891e-05}}
model_snel = {"collideAndStream": {"offset": 0.006203986032170888, "N": 3.48500278199325e-07}, "setExternalVector": {"offset": 2.590788867295757e-05, "N": 4.3112912128790356e-08}, "collideAndStream_comm": {"offset": -0.0004748026755225183, "area": 9.05354320891726e-07}, "syncEnvelopes_comm": {"offset": 0.00048311863052151715, "area": 1.2670340585585472e-07, "RBCs": 3.5270019874035664e-05}, "syncEnvelopes": {"offset": -1.376780408857048e-05, "RBCs": 8.277622850171989e-05}, "advanceParticles": {"offset": 0.0004938208025793962, "RBCs": 0.00013978737497846716}, "applyConstitutiveModel": {"offset": -1.3439321724241892e-05, "RBCs": 4.364834747587415e-05}, "deleteNonLocalParticles": {"offset": 5.103523368379448e-05, "RBCs": 1.4884632423801813e-05}, "spreadParticleForce": {"offset": 0.0008050285810133388, "RBCs": 0.00039740608639334083}, "interpolateFluidVelocity": {"offset": 0.000130060575853455, "RBCs": 7.700985333371006e-05}}

particle_components = ['syncEnvelopes', 'syncEnvelopes_comm', 'advanceParticles', "applyConstitutiveModel", "deleteNonLocalParticles"]
coupling_components = ["spreadParticleForce", "interpolateFluidVelocity"]
fluid_components    = ["collideAndStream", "collideAndStream_comm", "setExternalVector"]


# Scale scales the communication based on the number of communication neighbours
# 1/2 -> half of all potential neighbours
# 1/4 -> quater of all potential neighbours
# new_comm = old_comm * scale
# comm_scale = (collideAndSteam scale, syncEnvelopes scale)
def run_model(models, size, RBCs, iterations=1, fluid_keys=fluid_components, particle_keys=particle_components, coupling_keys=coupling_components, comm_scale=(1,1)):
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
        scale = 1

        if component == 'collideAndStream_comm':
            scale = comm_scale[0]
        if component == 'syncEnvelopes_comm':
            scale = comm_scale[1]
        
        m = models[component]

        tmp_value = m['offset'] * iterations * scale
        results[component] = tmp_value 
        results['total'] += tmp_value
        if component in fluid_keys:
            results['fluid'] += tmp_value

        if component in particle_keys:
            results['particle'] += tmp_value

        if component in coupling_keys:
            results['coupling'] += tmp_value

        keys = m.keys()
        if 'N' in keys:
            tmp_value = m['N'] * N * iterations * scale
            results[component] += tmp_value
            results['total'] += tmp_value
            
            if component in fluid_keys:
                results['fluid'] += tmp_value

            if component in particle_keys:
                results['particle'] += tmp_value

            if component in coupling_keys:
                results['coupling'] += tmp_value

            
        if 'area' in keys:
            tmp_value = m['area'] * area * iterations * scale
            results[component] += tmp_value
            results['total'] += tmp_value

            if component in fluid_keys:
                results['fluid'] += tmp_value

            if component in particle_keys:
                results['particle'] += tmp_value

            if component in coupling_keys:
                results['coupling'] += tmp_value

        if 'RBCs' in keys:
            tmp_value = m['RBCs'] * RBCs * iterations * scale
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


def plot_analysis(analysis_df, balanced_df, models, total=True, machines=['das6', 'snellius']):

    for j, _ in enumerate(machines):
        new_data = []
        new_data_balanced = []
        for i in np.unique(analysis_df[j]['jobid']):
            tmp = analysis_df[j].loc[analysis_df[j]['jobid'] == i]
            idx = tmp.loc[tmp['component'] == 'advanceParticles']['total'].idxmax()
            tmp = tmp.loc[tmp['threadid'] == tmp.loc[idx]['threadid']]

            if(total):
                tmp = tmp.loc[tmp['component'].isin(particle_components + coupling_components + fluid_components)]
            else:
                tmp = tmp.loc[tmp['component'].isin(particle_components + coupling_components)]

            new_data.append([np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], np.sum(tmp['total']), np.array(tmp['H'])[0]])

        for i in np.unique(balanced_df[j]['jobid']):
            tmp = balanced_df[j].loc[balanced_df[j]['jobid'] == i]
            idx = tmp.loc[tmp['component'] == 'advanceParticles']['total'].idxmax()
            tmp = tmp.loc[tmp['threadid'] == tmp.loc[idx]['threadid']]
            if(total):
                tmp = tmp.loc[tmp['component'].isin(particle_components + coupling_components + fluid_components)]
            else:
                tmp = tmp.loc[tmp['component'].isin(particle_components + coupling_components)]
            new_data_balanced.append([np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], np.sum(tmp['total']), np.array(tmp['h'])[0]])
            
        analysis_df[j] = pd.DataFrame(new_data, columns=['largest_subdomain', 'RBCs', 'total', 'H']) 
        balanced_df[j] = pd.DataFrame(new_data_balanced, columns=['largest_subdomain', 'RBCs', 'total', 'H']) 

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharex=True)
    axis_label = []

    j = -1
    for i, m in enumerate(machines):
        larger_hemo = np.array(balanced_df[i].loc[balanced_df[i]['H'] == '018']['RBCs'])[0]
        for _, H in enumerate(['z25', 's25', 'z50', 's50', 'z100', 's100']):

            tmp = analysis_df[i].loc[analysis_df[i]['H'] == H]

            j += 1

            name = ""
            tmpstr = ""
            tmpstr += " 18\% / {}\% ".format(9 if H[0] == 's' else 0)
            
            # name += "Snelius: "
            # scale = (1, 1)

            if H in ["z25"]:
                tmpstr += "& {} / {} ".format(8, 120)
                name += '8/120, '
                scale = (5 / 18, 5 / 26)
            if H in ["s25"]:
                tmpstr += "& {} / {} ".format(8, 120)
                name += '8/120, '
                scale = (12 / 18, 8 / 26)

            if H in ["z50"]:
                name += '16/112, '
                tmpstr += "& {} / {} ".format(16, 112)
                scale = (8 / 18, 8 / 26)
            if H in ["s50"]:
                name += '16/112, '
                tmpstr += "& {} / {} ".format(16, 112)
                scale = (12 / 18, 8 / 26)

            if H in ["z100"]:
                tmpstr += "& {} / {} ".format(32, 92)
                name += '32/92, '
                scale = (12 / 18, 16 / 26)
            if H in ["s100"]:
                tmpstr += "& {} / {} ".format(32, 92)
                name += '32/92, '
                scale = (1, 13 / 26)
            
            if H[0] == 's':
                name += '9\%'
            if H[0] == 'z':
                name += '0\%'
            
            
            # scale = (0, 0)

            # tmpstr += "&$ \\frac{{ {} }}{{18}} $& $\\frac{{ {} }}{{26}}$".format(scale[0] * 18, scale[1] * 26)
            tmpstr += "&$  {} / 18 $& ${}/26$".format(int(scale[0] * 18), int(scale[1] * 26))
 
            model_res_naive = run_model(models[i], np.array(tmp['largest_subdomain'])[0], larger_hemo, iterations)
            model_res = run_model(models[i], np.array(tmp['largest_subdomain'])[0], larger_hemo, iterations, comm_scale=scale)

            if not total:
                model_res_naive['total'] = model_res_naive['particle'] + model_res_naive['coupling']
                model_res['total'] = model_res['particle'] + model_res['coupling']
            
            ax1.errorbar(j, np.mean(tmp['total']), yerr=np.std(tmp['total']), ms=35, color=CB_color_cycle[1], fmt=".", capsize=5, lw=1, zorder=1)
            ax1.plot(j, model_res_naive['total'], ms=30, color=CB_color_cycle[1], marker="x", lw=0)
            ax1.plot(j, model_res['total'], ms=30, color=CB_color_cycle[2], marker="*", lw=0) 

            pred_error = np.abs(model_res_naive['total'] - np.mean(tmp['total'])) * (100 / np.mean(tmp['total']))
            ax2.plot(j, pred_error, 'X', color=CB_color_cycle[1], ms=20)

            pred_error = np.abs(model_res['total'] - np.mean(tmp['total'])) * (100 / np.mean(tmp['total']))
            ax2.plot(j, pred_error, '*', color=CB_color_cycle[2], ms=20)
 
            axis_label.append(name)


            # Print table
            pred = model_res['total']
            res = np.mean(tmp['total'])
            std = np.std(tmp['total'])
            err = np.abs(pred - res) * (100 / res)

            pred_old = model_res_naive['total']
            res_old = np.mean(tmp['total'])
            std_old = np.std(tmp['total'])
            err_old = np.abs(pred_old - res_old ) * (100 / res_old)

            # tmpstr += "{}\% ".format(H)
            tmpstr += "& $\\num{{{0:.2f}}}".format(res)
            tmpstr += "\pm \\num{{{0:.2f}}}$".format(std)

            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred_old)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(err_old)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(err)

            tmpstr += '\\\\'

            print(tmpstr)


    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.tick_params(axis='both', which='minor', labelsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='both', which='minor', labelsize=25)

    legend_handels = []
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='.', ms=10, label='Imbalanced Results'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='x', ms=10, label='Naive Prediction'))        
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[2], lw=0, marker='*', ms=10, label='New Prediction'))

    legend_handels2 = [] 
    legend_handels2.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='X', ms=10, label='Naive Prediction Error'))
    legend_handels2.insert(0, Line2D([0], [0], color=CB_color_cycle[2], lw=0, marker='X', ms=10, label='New Prediction Error'))

    ax1.legend(handles=legend_handels, loc='upper left', prop={'size': 15})
    ax2.legend(handles=legend_handels2, loc='upper right', prop={'size': 15})
    ax2.set_ylim(0, 30)
    if(total):
        ax1.set_ylabel("Time in seconds", fontsize=25)
        ax1.set_ylim(0, 700)
    else:
        ax1.set_ylabel("Particle and coupling time in seconds", fontsize=25)
        ax1.set_ylim(0, 120)
        
    ax2.set_ylabel("Prediction error [\%]", fontsize=25)
    ax1.set_xticks(np.arange(len(axis_label)), axis_label, rotation=45, fontsize=25)
    ax2.set_xticks(np.arange(len(axis_label)), axis_label, rotation=45, fontsize=25)
    plt.tight_layout()
    plt.savefig(results_dir + fig_name + "-{}".format("total" if total else "partial") + ".pdf", bbox_inches='tight')

if __name__ == "__main__":
    results_df_snel, exp_df_snel = model_generation.load_data(datadir.format("snellius"))
    analysis_df_snel = pd.merge(results_df_snel, exp_df_snel, on=['jobid'], how="left")

    results_df_snel, exp_df_snel = model_generation.load_data('../results/snellius/imbalance-hemo/output')
    exp_df_snel = exp_df_snel.loc[((exp_df_snel['H'] == 'z50') | (exp_df_snel['H'] == 's50'))]
    analysis_df_snel = pd.concat([analysis_df_snel, pd.merge(results_df_snel, exp_df_snel, on=['jobid'], how="right")])

    
    results_bal_df_snel, exp_bal_df_snel = model_generation.load_data(datadir_bal.format("snellius"))
    exp_bal_df_snel = exp_bal_df_snel.loc[exp_bal_df_snel['largest_subdomain'] == (100,100,100)]
    analysis_bal_df_snel = pd.merge(results_bal_df_snel, exp_bal_df_snel, on=['jobid'], how="right")

    plot_analysis([analysis_df_snel], [analysis_bal_df_snel], [model_snel], machines=['snellius'], total=True)

    print()
    
    plot_analysis([analysis_df_snel], [analysis_bal_df_snel], [model_snel], machines=['snellius'], total=False)