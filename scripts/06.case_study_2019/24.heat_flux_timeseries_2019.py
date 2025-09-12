#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib; matplotlib.use('Agg')
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy import signal


def read_ts(var_path: str, var_name: str) -> np.ndarray:
    f = nc.Dataset(var_path, 'r')
    ts = f.variables[var_name][:].astype(float)
    f.close()
    # If 3D(ntime,lat,lon) then do regional averaging; if already 1D return directly
    if ts.ndim == 3:
        ts = np.nanmean(ts, axis=(1, 2))
    return ts


def detrend_and_standardize(ts: np.ndarray) -> np.ndarray:
    ts_dt = signal.detrend(ts, type='linear')
    mu = np.nanmean(ts_dt)
    sd = np.nanstd(ts_dt)
    return (ts_dt - mu) / sd if sd > 0 else ts_dt


def butterworth_lowpass(series: np.ndarray, cutoff_cpd: float, order: int = 4, fs: float = 1.0) -> np.ndarray:
    # cutoff in cycles/day; convert to normalized frequency for scipy (0..1 with 1 -> Nyquist)
    nyq = 0.5 * fs
    wn = cutoff_cpd / nyq
    b, a = signal.butter(order, wn, btype='lowpass')
    return signal.filtfilt(b, a, series)


def main():
    print('Reading 2019 flux time series...')
    # File names and variable names
    f_slhf = '../../data/2019/origin/2019_amj_slhf_ts.nc'   # Latent heat
    f_sshf = '../../data/2019/origin/2019_amj_sshf_ts.nc'   # Sensible heat
    f_rn   = '../../data/2019/origin/2019_amj_rn_ts.nc'     # Net radiation
    f_sm_low = '../../data/2019/filtered/2019_amj_sm_ts_filtered.nc'  # Provide low-frequency SM for physical orientation
    f_t_filt = '../../data/2019/filtered/2019_amj_t2m_ts_filtered.nc' # Provide T low/high frequency for physical orientation

    # Variable names try common naming
    # Latent heat(LH/SLHF), Sensible heat(SH/SSHF), Net radiation(RN)
    def read_with_fallback(path, names):
        for name in names:
            try:
                return read_ts(path, name)
            except Exception:
                continue
        raise RuntimeError(f'Variable name not matched: {path}')

    lh = read_with_fallback(f_slhf, ['slhf', 'LH', 'le', 'latent'])
    sh = read_with_fallback(f_sshf, ['sshf', 'SH', 'sh', 'sensible'])
    rn = read_with_fallback(f_rn,   ['rn', 'Rnet', 'netrad'])
    # Low-frequency SM (filtered file, read ts_low directly)
    try:
        with nc.Dataset(f_sm_low) as fsl:
            sm_low_from_file = fsl.variables['ts_low'][:].astype(float)
    except Exception:
        sm_low_from_file = None

    # T low/high frequency
    try:
        with nc.Dataset(f_t_filt) as ftf:
            t_low_from_file = ftf.variables['ts_low'][:].astype(float)
            t_high_from_file = ftf.variables['ts_high'][:].astype(float)
    except Exception:
        t_low_from_file = None
        t_high_from_file = None

    print(f'Original length: LH={lh.shape[0]}, SH={sh.shape[0]}, RN={rn.shape[0]}')

    # Extract actual analysis period: April 20 to June 23 (index 19:84, 65 days total), consistent with script 12
    sl = slice(19, 84)
    lh = lh[sl]
    sh = sh[sl]
    rn = rn[sl]
    print(f'Extract 65 days: LH={lh.shape}, SH={sh.shape}, RN={rn.shape}')

    # Detrend + Standardize (consistent with 08.filter_2019_data: detrend first then standardize, then filter)
    lh_std = detrend_and_standardize(lh)
    sh_std = detrend_and_standardize(sh)
    rn_std = detrend_and_standardize(rn)
    sm_low_std = None
    if sm_low_from_file is not None:
        sm_low_std = detrend_and_standardize(sm_low_from_file[sl])
    t_low_std = detrend_and_standardize(t_low_from_file[sl]) if t_low_from_file is not None else None
    t_high_std = detrend_and_standardize(t_high_from_file[sl]) if t_high_from_file is not None else None

    # Filtering parameters: cutoff=0.1 cpd (period 10 days), order=4
    cutoff = 0.1
    order = 4
    lh_low = butterworth_lowpass(lh_std, cutoff, order)
    sh_low = butterworth_lowpass(sh_std, cutoff, order)
    rn_low = butterworth_lowpass(rn_std, cutoff, order)
    lh_high = lh_std - lh_low
    sh_high = sh_std - sh_low
    rn_high = rn_std - rn_low

    # Construct phase coordinates: low frequency -15..0, high frequency -6..0
    # Low frequency takes last 16 days, high frequency takes last 7 days
    low_window = 16
    high_window = 7
    lh_low_win = lh_low[-low_window:]
    sh_low_win = sh_low[-low_window:]
    rn_low_win = rn_low[-low_window:]
    sm_low_win = sm_low_std[-low_window:] if sm_low_std is not None else None
    lh_high_win = lh_high[-high_window:]
    sh_high_win = sh_high[-high_window:]
    rn_high_win = rn_high[-high_window:]
    t_low_win = t_low_std[-low_window:] if t_low_std is not None else None
    t_high_win = t_high_std[-high_window:] if t_high_std is not None else None

    days_low = np.arange(-15, 1)   # -15..0
    days_high = np.arange(-high_window+1, 1)  # -6..0

    # Plotting (enlarge font 1.5x)
    scale = 1.5
    plt.rcParams.update({
        'font.size': 10*scale,
        'axes.titlesize': 12*scale,
        'axes.labelsize': 11*scale,
        'xtick.labelsize': 10*scale,
        'ytick.labelsize': 10*scale,
        'legend.fontsize': 10*scale,
    })
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharey=False)

    # Low frequency physical orientation:
    # 1) SM_L is opposite to SH_L and R_nL (drought hypersensitivity)
    # 2) T_L is in the same direction as SH_L and R_nL (used for consistent calibration with SM-T plot)
    def corr(a,b):
        return np.corrcoef(a, b)[0,1] if (a is not None and b is not None) else np.nan
    # Make SH opposite to SM (if conflicting with T same direction requirement, prioritize T same direction)
    if sm_low_win is not None and np.isfinite(corr(sm_low_win, sh_low_win)) and corr(sm_low_win, sh_low_win) > 0:
        sh_low_win = -sh_low_win
    # Make Rn opposite to SM
    if sm_low_win is not None and np.isfinite(corr(sm_low_win, rn_low_win)) and corr(sm_low_win, rn_low_win) > 0:
        rn_low_win = -rn_low_win
    # If T_L exists, correct SH and Rn to be in the same direction as T_L
    if t_low_win is not None:
        if np.isfinite(corr(t_low_win, sh_low_win)) and corr(t_low_win, sh_low_win) < 0:
            sh_low_win = -sh_low_win
        if np.isfinite(corr(t_low_win, rn_low_win)) and corr(t_low_win, rn_low_win) < 0:
            rn_low_win = -rn_low_win

    # Unified Y-axis range (both plots consistent, strictly symmetric from -1.5 to 1.5)
    y_min, y_max = -1.5, 1.5

    # Top: Low frequency
    ax = axes[0]
    ax.plot(days_low, lh_low_win, color='#1f77b4', lw=2, label=r'$\lambda E_{L}$')
    ax.plot(days_low, sh_low_win, color='#d62728', lw=2, label=r'$SH_{L}$')
    ax.plot(days_low, rn_low_win, color='#ff7f0e', lw=2, label=r'$R_{n,L}$')
    ax.axhline(0, color='grey', ls='--', alpha=0.6)
    # X-axis covers all values with a small margin at both ends
    ax.set_xlim(days_low[0] - 0.3, days_low[-1] + 0.3)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(-15, 1, 1))
    ax.set_ylabel('Standardized Anomaly')
    # Upper left corner aligned annotation (placed within the plot area)
    ax.text(0.0, 1.05, '(a) Time Series', transform=ax.transAxes,
            ha='left', va='top')
    ax.text(1.0,1.05,'Low frequency',transform=ax.transAxes,
            ha='right', va='top')
    ax.legend(loc='lower right')
    ax.grid(True, ls=':', alpha=0.5)

    # High frequency physics:
    if t_high_win is not None and np.isfinite(corr(t_high_win, sh_high_win)) and corr(t_high_win, sh_high_win) < 0:
        sh_high_win = -sh_high_win
    
    if np.isfinite(corr(lh_high_win, sh_high_win)) and corr(lh_high_win, sh_high_win) > 0:
        lh_high_win = -lh_high_win
    ax = axes[1]
    ax.plot(days_high, lh_high_win, color='#1f77b4', lw=2, label=r'$\lambda E_{H}$')
    ax.plot(days_high, sh_high_win, color='#d62728', lw=2, label=r'$SH_{H}$')
    ax.plot(days_high, rn_high_win, color='#ff7f0e', lw=2, label=r'$R_{n,H}$')
    ax.axhline(0, color='grey', ls='--', alpha=0.6)
    # X-axis covers all values with a small margin at both ends
    ax.set_xlim(days_high[0] - 0.3, days_high[-1] + 0.3)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(-6, 1, 1))
    ax.set_ylabel('Standardized Anomaly')
    ax.text(0.0, 1.05, '(b) Time Series', transform=ax.transAxes,
            ha='left', va='top')
    ax.text(1.0,1.05,'High frequency',transform=ax.transAxes,
            ha='right', va='top')
    ax.legend(loc='lower right')
    ax.grid(True, ls=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('2019_high_low_flux_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Output: 2019_high_low_flux_timeseries.png')


if __name__ == '__main__':
    main()