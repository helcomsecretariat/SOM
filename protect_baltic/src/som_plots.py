"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import multiprocessing
from utilities import display_progress


def plot_total_pressure_load_levels(area, res, data, out_dir, progress, lock):
    """
    Plots TPL
    """
    # create new directory for the plots
    area_name = data['area'].loc[data['area']['ID'] == area, 'area'].values[0]
    out_path = os.path.join(out_dir, f'{area}_{area_name}', f'{area}_{area_name}_TotalPressureLoadLevels.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # plot settings
    marker = 's'
    markersize = 5
    markercolor = 'black'
    capsize = 3
    capthick = 1
    elinewidth = 1
    ecolor = 'salmon'
    label_angle = 60
    char_limit = 25

    fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)

    # adjust data
    suffixes = ('_mean', '_error')
    df = pd.merge(res['TPL']['Mean'].loc[:, ['ID', area]], res['TPL']['Error'].loc[:, ['ID', area]], on='ID', suffixes=suffixes)
    x_vals = data['state'].loc[:, 'state'].values
    x_vals = np.array([x[:char_limit]+'...' if len(x) > char_limit else x for x in x_vals])     # limit characters to char_limit
    y_vals = df[str(area)+'_mean'] * 100    # convert to %
    y_err = df[str(area)+'_error'] * 100    # conver to %

    # create plot
    ax.errorbar(np.arange(len(x_vals)), y_vals, yerr=y_err, linestyle='None', marker=marker, capsize=capsize, capthick=capthick, elinewidth=elinewidth, markersize=markersize, color=markercolor, ecolor=ecolor)
    ax.set_xlabel('Environmental State')
    ax.set_ylabel('Level (%)')
    ax.set_title(f'Total Pressure Load on Environmental States\n({area_name})')
    ax.set_xticks(np.arange(len(x_vals)), x_vals, rotation=label_angle, ha='right')
    ax.yaxis.grid(True, linestyle='--', color='lavender')

    # adjust axis limits
    x_lim = [- 0.5, len(x_vals) - 0.5]
    ax.set_xlim(x_lim)
    y_lim = [-5, 105]
    ax.set_ylim(y_lim)

    # export
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    
    with lock:
        progress.current += 1
        display_progress(progress.current / progress.total, text='\t\tTPL: ')


def plot_pressure_levels(area, res, data, out_dir, progress, lock):
    """
    Plots pressures
    """
    # create new directory for the plots
    area_name = data['area'].loc[data['area']['ID'] == area, 'area'].values[0]
    out_path = os.path.join(out_dir, f'{area}_{area_name}', f'{area}_{area_name}_PressureLevels.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # plot settings
    marker = 's'
    markersize = 5
    markercolor = 'black'
    capsize = 3
    capthick = 1
    elinewidth = 1
    ecolor = 'salmon'
    label_angle = 60
    char_limit = 25
    
    fig, ax = plt.subplots(figsize=(25, 12), constrained_layout=True)

    # adjust data
    suffixes = ('_mean', '_error')
    df = pd.merge(res['Pressure']['Mean'].loc[:, ['ID', area]], res['Pressure']['Error'].loc[:, ['ID', area]], on='ID', suffixes=suffixes)
    x_vals = data['pressure'].loc[:, 'pressure'].values
    x_vals = np.array([x[:char_limit]+'...' if len(x) > char_limit else x for x in x_vals])     # limit characters to char_limit
    y_vals = df[str(area)+'_mean'] * 100    # convert to %
    y_err = df[str(area)+'_error'] * 100    # conver to %

    # create plot
    ax.errorbar(np.arange(len(x_vals)), y_vals, yerr=y_err, linestyle='None', marker=marker, capsize=capsize, capthick=capthick, elinewidth=elinewidth, markersize=markersize, color=markercolor, ecolor=ecolor)
    ax.set_xlabel('Pressure')
    ax.set_ylabel('Level (%)')
    ax.set_title(f'Pressure Levels\n({area_name})')
    ax.set_xticks(np.arange(len(x_vals)), x_vals, rotation=label_angle, ha='right')
    ax.yaxis.grid(True, linestyle='--', color='lavender')

    # adjust axis limits
    x_lim = [- 0.5, len(x_vals) - 0.5]
    ax.set_xlim(x_lim)
    y_lim = [-5, 105]
    ax.set_ylim(y_lim)

    # export
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    
    with lock:
        progress.current += 1
        display_progress(progress.current / progress.total, text='\t\tPressures: ')


def plot_thresholds(area, res, data, out_dir, progress, lock):
    """
    Plots thresholds comparison
    """
    # create new directory for the plots
    area_name = data['area'].loc[data['area']['ID'] == area, 'area'].values[0]
    out_path = os.path.join(out_dir, f'{area}_{area_name}', f'{area}_{area_name}_Thresholds.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # plot settings
    capsize = 3
    capthick = 1
    elinewidth = 1
    ecolor = 'salmon'
    label_angle = 60
    char_limit = 25
    bar_width = 0.4
    bar_color_1 = 'turquoise'
    bar_color_2 = 'seagreen'
    edge_color = 'black'

    fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)

    # adjust data
    x_labels = np.array([x[:char_limit]+'...' if len(x) > char_limit else x for x in data['state'].loc[:, 'state'].values])     # limit characters to char_limit
    x_vals = np.arange(len(x_labels))
    suffixes = ('_mean', '_error')
    df = pd.merge(res['TPLRed']['Mean'].loc[:, ['ID', area]], res['TPLRed']['Error'].loc[:, ['ID', area]], on='ID', suffixes=suffixes)
    y_vals_tpl = df[str(area)+'_mean'] * 100    # convert to %
    y_err_tpl = df[str(area)+'_error'] * 100    # conver to %
    df = pd.merge(res['Thresholds']['Mean'].loc[:, ['ID', area]], res['Thresholds']['Error'].loc[:, ['ID', area]], on='ID', suffixes=suffixes)
    y_vals_ges = df[str(area)+'_mean'] * 100    # convert to %
    y_err_ges = df[str(area)+'_error'] * 100    # convert to %

    # create plot
    label_tpl = 'Reduction with measures'
    ax.bar(x_vals-bar_width/2, y_vals_tpl, width=bar_width, align='center', color=bar_color_1, label=label_tpl, edgecolor=edge_color)
    ax.errorbar(x_vals-bar_width/2, y_vals_tpl, yerr=y_err_tpl, linestyle='None', marker='None', capsize=capsize, capthick=capthick, elinewidth=elinewidth, ecolor=ecolor)
    label_ges = 'GES'
    ax.bar(x_vals+bar_width/2, y_vals_ges, width=bar_width, align='center', color=bar_color_2, label=label_ges, edgecolor=edge_color)
    ax.errorbar(x_vals+bar_width/2, y_vals_ges, yerr=y_err_ges, linestyle='None', marker='None', capsize=capsize, capthick=capthick, elinewidth=elinewidth, ecolor=ecolor)
    ax.set_xlabel('Environmental State')
    ax.set_ylabel('Reduction (%)')
    ax.set_title(f'Total Pressure Load Reduction vs. GES Reduction Thresholds\n({area_name})')
    ax.set_xticks(x_vals, x_labels, rotation=label_angle, ha='right')
    ax.yaxis.grid(True, linestyle='--', color='lavender')
    ax.legend()

    # adjust axis limits
    x_lim = [- 0.5, len(x_vals) - 0.5]
    ax.set_xlim(x_lim)
    y_lim = [0, 100]
    ax.set_ylim(y_lim)

    # export
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    
    with lock:
        progress.current += 1
        display_progress(progress.current / progress.total, text='\t\tThresholds: ')


def build_display(res: dict[str, dict[str, pd.DataFrame]], data: dict[str, pd.DataFrame], out_dir: str, use_parallel_processing: bool = False):
    """
    Constructs plots to visualize results.
    """
    areas = data['area']['ID']

    cpu_count = multiprocessing.cpu_count()     # available cpu cores
    with multiprocessing.Manager() as manager:
        progress = manager.Namespace()
        progress.current = 0
        progress.total = len(areas)
        lock = manager.Lock()
        if use_parallel_processing:
            with multiprocessing.Pool(processes=(min(cpu_count - 2, len(areas)))) as pool:
                jobs = [(area, res, data, out_dir, progress, lock) for area in areas]
                display_progress(progress.current / progress.total, text='\t\tTPL: ')
                progress.current = 0
                pool.starmap(plot_total_pressure_load_levels, jobs)
                display_progress(progress.current / progress.total, text='\t\tTPL: ')
                display_progress(progress.current / progress.total, text='\n\t\tPressures: ')
                progress.current = 0
                pool.starmap(plot_pressure_levels, jobs)
                display_progress(progress.current / progress.total, text='\t\tPressures: ')
                display_progress(progress.current / progress.total, text='\n\t\tThresholds: ')
                progress.current = 0
                pool.starmap(plot_thresholds, jobs)
                display_progress(progress.current / progress.total, text='\t\tThresholds: ')
        else:
            progress.current = 0
            display_progress(progress.current / progress.total, text='\t\tTPL: ')
            for area in areas:
                plot_total_pressure_load_levels(area, res, data, out_dir, progress, lock)
            display_progress(progress.current / progress.total, text='\t\tTPL: ')
            progress.current = 0
            display_progress(progress.current / progress.total, text='\n\t\tPressures: ')
            for area in areas:
                plot_pressure_levels(area, res, data, out_dir, progress, lock)
            display_progress(progress.current / progress.total, text='\t\tPressures: ')
            progress.current = 0
            display_progress(progress.current / progress.total, text='\n\t\tThresholds: ')
            for area in areas:
                plot_thresholds(area, res, data, out_dir, progress, lock)
            display_progress(progress.current / progress.total, text='\t\tThresholds: ')

    #
    # Measure effects
    #

    print('\n\t\tMeasure effects...')

    # plot settings
    capsize = 3
    capthick = 1
    elinewidth = 1
    ecolor = 'salmon'
    label_angle = 60
    char_limit = 25
    bar_width = 0.4
    edge_color = 'black'

    fig, ax = plt.subplots(figsize=(100, 14), constrained_layout=True)

    bar_width = 0.8
    edge_color = 'black'
    activity_font_size = 8

    # adjust data
    df = res['MeasureEffects']['Mean'].merge(res['MeasureEffects']['Error'], on=['measure', 'pressure', 'state', 'activity'], how='left', suffixes=('_mean', '_error'))
    df = df.sort_values(by=['measure', 'pressure', 'state', 'activity'])
    suffixes = ('', '_name')
    for col in ['measure', 'activity', 'pressure', 'state']:
        df = df.merge(data[col].loc[:, [col, 'ID']], left_on=col, right_on='ID', how='left', suffixes=suffixes)
        df = df.drop(columns=[col, 'ID'])
        df = df.rename(columns={col+'_name': col})
        df.loc[:, col] = np.array([(x[:char_limit]+'...' if len(x) > char_limit else x) if type(x) == str else 'All' for x in df.loc[:, col].values])
    df['index'] = np.arange(len(df))
    x_ticks = {x: df[df['measure'] == x]['index'].mean() for x in df['measure'].unique()}

    # set colors
    df['color_key'] = df['pressure'].astype(str) + '_' + df['state'].astype(str)
    unique_keys = df['color_key'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_keys)))
    color_map = {key: colors[i] for i, key in enumerate(unique_keys)}

    # create plot
    for key in unique_keys:
        subset = df[df['color_key'] == key]
        bars = ax.bar(subset['index'], subset['reduction_mean'] * 100, width=bar_width, color=color_map[key], label=key if key not in ax.get_legend_handles_labels()[1] else '', edgecolor=edge_color)
        ax.errorbar(subset['index'], subset['reduction_mean'] * 100, yerr=subset['reduction_error'] * 100, linestyle='None', marker='None', capsize=capsize, capthick=capthick, elinewidth=elinewidth, ecolor=ecolor)
        for bar, (_, row) in zip(bars, subset.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, str(row['activity']), 
                    ha='center', va='center', rotation=90, fontsize=activity_font_size, color='white')

    ax.set_xlabel('Measure')
    ax.set_ylabel('Reduction effect (%)')
    ax.set_title(f'Measure Reduction Effects')
    ax.set_xticks(list(x_ticks.values()), list(x_ticks.keys()), rotation=label_angle, ha='right')
    ax.yaxis.grid(True, linestyle='--', color='lavender')
    ax.legend(title='Pressure/State', bbox_to_anchor=(1.05, 1), loc='upper left')

    # adjust axis limits
    x_lim = [- 0.5, len(df) - 0.5]
    ax.set_xlim(x_lim)
    y_lim = [0, 100]
    ax.set_ylim(y_lim)

    # export
    for area in areas:
        area_name = data['area'].loc[areas == area, 'area'].values[0]
        temp_dir = os.path.join(out_dir, f'{area}_{area_name}')
        plt.savefig(os.path.join(temp_dir, f'{area}_{area_name}_MeasureEffects.png'), dpi=200)

    plt.close(fig)


