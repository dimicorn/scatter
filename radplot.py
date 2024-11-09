import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load
from astropy.io.fits import open as fopen
from warnings import filterwarnings
import pandas as pd
from os import listdir
from fnmatch import fnmatch
from scipy.stats import binned_statistic as bstat
from uv import read_uv


sqr = lambda x : x * x

def raw_radplot(X: np.array, Y: np.array, coresize: float, object_name: str) -> None:
    coresize = coresize / 3.6e6 * np.pi / 180
    sigma = coresize / (2 * np.sqrt(2 * np.log(2)))
    a = 1 / (2 * sqr(sigma))
    y = np.exp(-sqr(np.pi * X) / a) * Y[np.argmin(X)]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X * 1e-6, Y, color='k', marker='.')
    ax.scatter(X * 1e-6, y, color='r', marker='.')
    # ax.vlines(np.sqrt(a) / np.pi * np.sqrt(np.log(2)) * 1e-6, 0, 3, color='g')
    # ax.set_ylim((0, np.log(2)))
    ax.set_title(object_name, loc='center')
    ax.set_xlabel(r'$R = \sqrt{U^2 + V^2}$ Baseline radius (M$\lambda$)')
    ax.set_ylabel('Amplitude, Jy')
    plt.savefig(f'raw_figures/{object_name}_radplot.png', dpi=500)
    plt.close(fig)

def radplot(stat: tuple[np.array], std: np.array, object_name: str) -> None:
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

def new_radplot(X: np.array, Y: np.array, coresize: float, glat: float, glon: float, object_name: str) -> None:
    coresize = coresize / 3.6e6 * np.pi / 180
    sigma = coresize / (2 * np.sqrt(2 * np.log(2)))
    a = 1 / (2 * sqr(sigma))
    y = np.exp(-sqr(np.pi * X) / a) * Y[np.argmin(X)]
    hwhm = np.sqrt(a) / np.pi * np.sqrt(np.log(2))
    bins = 10
    if len(np.where(X < hwhm)[0]) == 0:
        return
    l = X[np.where(X > hwhm)].min()
    bin_means, bin_edges, _ = bstat(X[np.where(X <= l)], Y[np.where(X <= l)], statistic='mean', bins=bins)
    std, _, _ = bstat(X[np.where(X <= l)], Y[np.where(X <= l)], statistic='std', bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])/2

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.errorbar(
        bin_edges[:-1] * 1e-6, bin_means, 
        xerr=bin_width * 1e-6, yerr=std,
        marker='.', color='k', ecolor='r',
        linestyle='none'
    )
    bins = 20
    bin_means, bin_edges, _ = bstat(X[np.where(X > l)], Y[np.where(X > l)], statistic='mean', bins=bins)
    std, _, _ = bstat(X[np.where(X > l)], Y[np.where(X > l)], statistic='std', bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])/2

    ax.errorbar(
        bin_edges[:-1] * 1e-6, bin_means, 
        xerr=bin_width * 1e-6, yerr=std,
        marker='.', color='k', ecolor='r',
        linestyle='none'
    )
    ax.scatter(X * 1e-6, y, marker='.', color='b')
    ax.set_ylim((0, np.mean(Y) + 3 * np.std(Y)))
    ax.set_title(object_name[:-8], loc='right')
    ax.set_title(f'GLAT = {glat}, GLON = {glon}', loc='left')
    ax.set_xlabel(r'$R = \sqrt{U^2 + V^2}$ Baseline radius (M$\lambda$)')
    ax.set_ylabel('Amplitude, Jy')
    plt.savefig(f'figures3/{object_name}_radplot.png')
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
        for file in listdir(f'{data_path}/{name}'):
            if fnmatch(file, f'{filename}_*_vis.fits'):
                res.append(f'{name}/{file}')
    return res

filterwarnings('ignore')

with open('config.yaml') as f:
    cfg = safe_load(f)
    fits_path = cfg['fits_path']

df = pd.read_csv('top_20_2.csv')
files = get_filepaths(fits_path, df)

# with open('potential_sources.txt') as f:
#     files = f.read().split()

# files = [(files[i], float(files[i+1])) for i, _ in enumerate(files) if i % 2 == 0]

# bins = 25
for core, glat, glon, file in zip(df.Sizecore, df.GLAT, df.GLON, files):
    with fopen(f'{fits_path}/{file}') as f:
# for i, file in enumerate(files):
#     with fopen(f'{fits_path}/{file[0]}') as f:
        f.verify('fix')
        hdulist = f
        freq_header, freq_data = hdulist['AIPS FQ'].header, hdulist['AIPS FQ'].data
        uv_header, data = hdulist['PRIMARY'].header, hdulist['PRIMARY'].data
    
    uv = read_uv(hdulist, freq_header, freq_data, uv_header, data)
    # print(uv.shape)
    r = np.sqrt(np.square(uv[0]) + np.square(uv[1]))

    # mean_stat = bstat(r, uv[2], statistic='mean', bins=bins)
    # std, _, _ = bstat(r, uv[2], statistic='std', bins=bins)

    # x, y = mean_stat.bin_edges[:-1], mean_stat.statistic
    # radplot(mean_stat, std, f'{file.split("/")[1].split(".")[0]}')
    # raw_radplot(r, uv[2], file[1], f'{file[0].split("/")[1].split(".")[0]}')
    new_radplot(r, uv[2], core, glat, glon, f'{file.split("/")[1].split(".")[0]}')