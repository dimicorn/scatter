import numpy as np


UU, VV = 'UU', 'VV'

def get_freq(hdu) -> float:
    # FIXME: Refactor this plz
    header = hdu['PRIMARY'].header
    for i in range(1, header['NAXIS'] + 1):
        try:
            if header[f'CTYPE{i}'] == 'FREQ':
                return header[f'CRVAL{i}']
        except KeyError:
            ...
    print('No CTYPE_i == FREQ was found')
    exit(1)

def read_uv(hdu, freq_header, freq_data, uv_header, data) -> np.array:
    '''Reading UV data'''

    freq = get_freq(hdu)
    gcount = uv_header['GCOUNT']
    if_nums = freq_header['NO_IF']
    if_freq = freq_data['IF FREQ']

    uu, vv = [], []
    try:
        data[UU], data[VV]
        uu_key, vv_key = UU, VV
    except KeyError:
        try:
            data['UU--'], data['VV--']
            uu_key, vv_key = 'UU--', 'VV--'
        except KeyError:
            try:
                data['UU---SIN'], data['VV---SIN']
                uu_key, vv_key = 'UU---SIN', 'VV---SIN'
            except KeyError: 
                print(
                f'Caution: File has weird UU and VV keys'
            )

    if if_nums == 1:
        for ind in range(gcount):
            for if_num in range(if_nums):
                u = data[uu_key][ind] * (freq + if_freq[if_num])
                v = data[vv_key][ind] * (freq + if_freq[if_num])
                uu.append(u)
                vv.append(v)
    elif if_nums > 1:
        for ind in range(gcount):
            for if_num in range(if_nums):
                u = (data[uu_key][ind] * 
                        (freq + if_freq[0][if_num]))
                v = (data[vv_key][ind] * 
                        (freq + if_freq[0][if_num]))
                uu.append(u)
                vv.append(v)
    
    vis = (data.data[:, 0, 0, :, 0, 0, 0] + 
            data.data[:, 0, 0, :, 0, 0, 1] * 1j)
    ampl = np.absolute(vis).flatten()
    phase = np.angle(vis).flatten()

    X = np.array([np.array(uu), np.array(vv), 
                    np.array(ampl), np.array(phase)])
    return X
    # X_sym = np.copy(X)
    # X_sym[0] = -1 * X_sym[0]
    # X_sym[1] = -1 * X_sym[1]

    # return np.append(X.T, X_sym.T, axis=0).T