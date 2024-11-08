import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load
from astropy.io.fits import open as fopen
from warnings import filterwarnings
import pandas as pd
import os
from fnmatch import fnmatch
from scipy.stats import binned_statistic as bstat
from uv import read_uv


def radplot(stat, std, object_name) -> None:
    bin_means, bin_edges, _ = stat
    bin_width = (bin_edges[1] - bin_edges[0])/2
    # print(type(std))
    fig = plt.figure()
    ax = fig.add_subplot()
    # print((bin_edges[1] - bin_edges[0]) * 1e-6)
    # ax.set_yscale('log')
    ax.errorbar(
        bin_edges[:-1] * 1e-6, bin_means, 
        xerr=bin_width*1e-6, yerr=std,
        marker='.', color='k', ecolor='r',
        linestyle='none'
    )
    ax.set_ylim((0, 2))
    ax.set_title(object_name, loc='center')
    ax.set_xlabel(r'$R = \sqrt{U^2 + V^2}$ Baseline radius (M$\lambda$)')
    ax.set_ylabel('Amplitude, Jy')
    plt.savefig(f'figures2/{object_name}_radplot.png', dpi=500)
    plt.close(fig)

def get_freq_band(freq: float) -> str:
    freq_bands = {
        'L': (1, 1.8), 'S': (1.8, 2.8), 'C': (2.8, 7), 'X': (7, 9), 
        'U': (9, 17), 'K': (17, 26), 'Q': (26, 50), 'W': (50, 100), 
        'G': (100, 250)
    }
    for key, value in freq_bands.items():
        if value[0] <= freq and freq <= value[1]:
            return key
    
def get_filepaths(data_path: str, df: pd.DataFrame) -> list[str]:
    res = []
    
    for name, freq, date in zip(df.Name, df.Freq, df.Epoch):
        filename = '_'.join([name, get_freq_band(freq), '_'.join(date.split('-'))])
        for file in os.listdir(f'{data_path}/{name}'):
            if fnmatch(file, f'{filename}_*_vis.fits'):
                res.append(f'{name}/{file}')
    return res

filterwarnings('ignore')

with open('config.yaml') as f:
    cfg = safe_load(f)
    fits_path = cfg['fits_path']

# df = pd.read_csv('top_20.csv')
# files = get_filepaths(fits_path, df)

with open('potential_sources.txt') as f:
    files = f.read().splitlines()

bins = 25
for i, file in enumerate(files):
    with fopen(f'{fits_path}/{file}') as f:
        f.verify('fix')
        hdulist = f
        freq_header, freq_data = hdulist['AIPS FQ'].header, hdulist['AIPS FQ'].data
        uv_header, data = hdulist['PRIMARY'].header, hdulist['PRIMARY'].data
    
    uv = read_uv(hdulist, freq_header, freq_data, uv_header, data)
    # print(uv.shape)
    r = np.sqrt(np.square(uv[0]) + np.square(uv[1]))
    mean_stat = bstat(r, uv[2], statistic='mean', bins=bins)
    std, _, _ = bstat(r, uv[2], statistic='std', bins=bins)
    # x, y = mean_stat.bin_edges[:-1], mean_stat.statistic
    radplot(mean_stat, std, f'{file.split("/")[1].split(".")[0]}')