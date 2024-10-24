import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load
from astropy.io.fits import open as fopen
from warnings import filterwarnings
from os import listdir
from scipy.stats import binned_statistic as bstat
from uv import read_uv


def radplot(x, y, object_name) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x * 1e-6, y, marker='.')
    ax.set_title(object_name, loc='center')
    ax.set_xlabel(r'$R = \sqrt{U^2 + V^2}$ Baseline radius (M$\lambda$)')
    ax.set_ylabel('Amplitude, Jy')
    plt.savefig(f'{object_name}_radplot.png', dpi=500)
    plt.close(fig)

filterwarnings('ignore')

with open('config.yaml') as f:
    cfg = safe_load(f)
    fits_path = cfg['fits_path']

obj = 'J2007+4029' # 'J1851+0035'
# files = [
#     f for f in listdir(f'{fits_path}/{obj}') 
#     if f[-8:-5] == 'vis' and f.split('_')[1] == 'S'
# ]
files = ['J2007+4029_S_2018_08_10_pet_vis.fits']

for i, file in enumerate(files):
    with fopen(f'{fits_path}/{obj}/{file}') as f:
        f.verify('fix')
        hdulist = f
        freq_header, freq_data = hdulist['AIPS FQ'].header, hdulist['AIPS FQ'].data
        uv_header, data = hdulist['PRIMARY'].header, hdulist['PRIMARY'].data
    
    uv = read_uv(hdulist, freq_header, freq_data, uv_header, data)
    print(uv.shape)
    r = np.sqrt(np.square(uv[0]) + np.square(uv[1]))
    mean_stat = bstat(r, uv[2], statistic='mean', bins=250)
    x, y = mean_stat.bin_edges[:-1], mean_stat.statistic
    radplot(x, y, f'{obj}_{i}')