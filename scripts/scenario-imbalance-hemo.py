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
fig_name= "scenario_imbalance_hemo"
datadir = "../results/{}/imbalance-hemo/output/"
iterations = 500
large_hemo = 250

model_das  =  {"collideAndStream": {"offset": 0.007586970083640878, "N": 2.4457985521310744e-07}, "setExternalVector": {"offset": -0.00017923400279257784, "N": 2.1311086952690122e-08}, "collideAndStream_comm": {"offset": 0.0009759617046640274, "area": 2.158405734410927e-07}, "syncEnvelopes_comm": {"offset": 0.00040164204605704534, "area": 1.591426069236932e-08, "RBCs": 9.369049445897329e-06}, "syncEnvelopes": {"offset": 0.00010545500938155938, "RBCs": 3.4645093627640815e-05}, "advanceParticles": {"offset": 0.0005292307787171433, "RBCs": 8.230150369077469e-05}, "applyConstitutiveModel": {"offset": -3.4074229702742984e-05, "RBCs": 2.794886344327952e-05}, "deleteNonLocalParticles": {"offset": 1.5278511490172454e-05, "RBCs": 7.705223394898288e-06}, "spreadParticleForce": {"offset": 0.0011070458465703972, "RBCs": 0.0002540657002879167}, "interpolateFluidVelocity": {"offset": 0.0002617313321741244, "RBCs": 4.1357788939953166e-05}}
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


def plot_analysis(analysis_df, models, total=True, machines=["das6", "Snellius"]):

    if total:
        for j, _ in enumerate(machines):
            analysis_df[j] = analysis_df[j].loc[analysis_df[0]['component'] == 'total']
            analysis_df[j] = analysis_df[j].loc[analysis_df[0].groupby('jobid', sort=False)['total'].idxmax()]
    else:
        for j, _ in enumerate(machines):
            new_data = []
            for i in np.unique(analysis_df[j]['jobid']):
                tmp = analysis_df[j].loc[analysis_df[j]['jobid'] == i]
                idx = tmp.loc[tmp['component'] == 'advanceParticles']['total'].idxmax()
                tmp = tmp.loc[tmp['threadid'] == tmp.loc[idx]['threadid']]
                tmp = tmp.loc[tmp['component'].isin(particle_components + coupling_components)]
                new_data.append([np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], np.sum(tmp['total']), np.array(tmp['H'])[0]])
                
            analysis_df[j] = pd.DataFrame(new_data, columns=['largest_subdomain', 'RBCs', 'total', 'H'])



    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(2, 1, 1)
    axis_label = []
    for i, m in enumerate(machines):
        for j, H in enumerate(pd.unique(analysis_df[i]['H'])):
            tmp = analysis_df[i].loc[analysis_df[i]['H'] == H]
            
            if total:
                model_res_naive = run_model(models[i], np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], iterations)
                model_res = run_model(models[i], np.array(tmp['largest_subdomain'])[0], large_hemo, iterations)
                plt.errorbar(i * 2 + j, np.mean(tmp['total']), yerr=np.std(tmp['total']), ms=30, color='k', fmt=".", capsize=5, lw=1)
                plt.errorbar(i * 2 + j, model_res_naive['total'], yerr=0, ms=20, color=CB_color_cycle[1], fmt="X", capsize=5, lw=1)
                plt.errorbar(i * 2 + j, model_res['total'], yerr=0, ms=20, color=CB_color_cycle[2], fmt="X", capsize=5, lw=1)
            else:
                model_res_naive = run_model(models[i], np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], iterations)
                model_res = run_model(models[i], np.array(tmp['largest_subdomain'])[0], large_hemo, iterations)
                plt.errorbar(i * 2 + j, np.mean(tmp['total']), yerr=np.std(tmp['total']), ms=30, color='k', fmt=".", capsize=5, lw=1, zorder=1)
                plt.plot(i * 2 + j, model_res_naive['particle'] +  model_res_naive['coupling'], ms=20, color=CB_color_cycle[1], marker="x", lw=0)
                plt.plot(i * 2 + j, model_res['particle'] + model_res['coupling'], ms=20, color=CB_color_cycle[2], marker="x", lw=0) 

                if H in ['z50', 's50']:
                    plt.plot(i * 2 + j, model_res['particle'] + model_res['coupling'] - model_res['syncEnvelopes_comm'] / 2 - model_res['collideAndStream_comm'] / 4,
                                ms=20, color=CB_color_cycle[3], marker="x", lw=0)
                    
            name = ""
            
            if m == 'das6':
                name += "DAS6: 12/12, "
                if H[0] == 's':
                    name += '18\%/9\%'
                if H[0] == 'z':
                    name += '18\%/0\%'
            
            if m == 'Snellius':
                
                name += "Snelius: "

                if H[1] == '5':
                    name += '16/112, '
                if H[1] == '2':
                    name += '64/64, '
                    
                if H[0] == 's':
                    name += '18\%/9\%'
                if H[0] == 'z':
                    name += '18\%/0\%'
                
                
            axis_label.append(name)


    legend_handels = []
    legend_handels.insert(0, Line2D([0], [0], color='k', lw=0, marker='.', ms=10, label='Results'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[1], lw=0, marker='x', ms=10, label='P1'))        
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[2], lw=0, marker='x', ms=10, label='P2'))
    legend_handels.insert(0, Line2D([0], [0], color=CB_color_cycle[3], lw=0, marker='x', ms=10, label='P3'))

    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.weight': 'bold'})

    plt.legend(handles=legend_handels)
    # ax.set_yscale('log')
    plt.ylim(10)
    if total:
        plt.ylabel("Time in Seconds")
    else:
        plt.ylabel("Particle + Coupling time in seconds")
    plt.xlabel("Experiment")
    # plt.title("Hematocrit imbalance")
    plt.xticks(np.arange(len(axis_label)), axis_label, rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir + fig_name + "-{}".format("total" if total else "partial") + ".pdf", bbox_inches='tight')


def print_latex_table(analysis_df, models, total=True, machines=["das6", "Snellius"]):
    if total:
        for j, _ in enumerate(machines):
            analysis_df[j] = analysis_df[j].loc[analysis_df[j]['component'] == 'total']
            analysis_df[j] = analysis_df[j].loc[analysis_df[j].groupby('jobid', sort=False)['total'].idxmax()]
    else:
        for j, _ in enumerate(machines):
            new_data = []
            for i in np.unique(analysis_df[j]['jobid']):
                tmp = analysis_df[j].loc[analysis_df[j]['jobid'] == i]
                idx = tmp.loc[tmp['component'] == 'advanceParticles']['total'].idxmax()
                tmp = tmp.loc[tmp['threadid'] == tmp.loc[idx]['threadid']]
                tmp = tmp.loc[tmp['component'].isin(particle_components + coupling_components)]
                new_data.append([np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], np.sum(tmp['total']), np.array(tmp['H'])[0]])
                
            analysis_df[j] = pd.DataFrame(new_data, columns=['largest_subdomain', 'RBCs', 'total', 'H']) 


    for i, m in enumerate(machines):
        for j, H in enumerate(pd.unique(analysis_df[i]['H'])):
            tmp = analysis_df[i].loc[analysis_df[i]['H'] == H]
            
            model_res_naive = run_model(models[i], np.array(tmp['largest_subdomain'])[0], np.array(tmp['RBCs'])[0], iterations)
            model_res = run_model(models[i], np.array(tmp['largest_subdomain'])[0], large_hemo, iterations)
            if not total:
                model_res_naive['total'] = model_res_naive['particle'] + model_res_naive['coupling']
                model_res['total'] = model_res['particle'] + model_res['coupling']

            tmpstr = m 
            tmpstr += "& 18\% / {}\% ".format(9 if H[0] == 's' else 0)

            if H in ["s200", "z200"]:
                tmpstr += "& {} / {} ".format(64, 64)
            else:
                if m == "das6":
                    tmpstr += "& {} / {} ".format(12, 12)
                elif m == "Snellius":
                    tmpstr += "& {} / {} ".format(16, 112)

            pred = model_res_naive['total']
            res = np.mean(tmp['total'])
            std = np.std(tmp['total'])
            err = np.abs(pred - res) * (100 / res)
            tmpstr += "& $\\num{{{0:.2f}}}".format(res)
            tmpstr += "\pm \\num{{{0:.2f}}}$".format(std)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(err)


            pred = model_res['total']
            err = np.abs(pred - res) * (100 / res)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(pred)
            tmpstr += "& $\\num{{{0:.2f}}}$".format(err)


            pred = model_res['total']
            if H in ['z50', 's50']:
                pred -= model_res['syncEnvelopes_comm'] / 2
                pred -= model_res['collideAndStream_comm'] / 4
                err = np.abs(pred - res) * (100 / res)
                tmpstr += "& $\\num{{{0:.2f}}}$".format(pred)
                tmpstr += "& $\\num{{{0:.2f}}}$\\\\".format(err)
            else:
                tmpstr += " & - & -\\"
            print(tmpstr)


if __name__ == "__main__":
    results_df_das, exp_df_das = model_generation.load_data(datadir.format("das6"))
    results_df_snel, exp_df_snel = model_generation.load_data(datadir.format("snellius"))
    analysis_df_das = pd.merge(results_df_das, exp_df_das, on=['jobid'], how="left")
    analysis_df_snel = pd.merge(results_df_snel, exp_df_snel, on=['jobid'], how="left")

    plot_analysis([analysis_df_das, analysis_df_snel], [model_das, model_snel], total=False)
    print_latex_table([analysis_df_das, analysis_df_snel], [model_das, model_snel], total=False)