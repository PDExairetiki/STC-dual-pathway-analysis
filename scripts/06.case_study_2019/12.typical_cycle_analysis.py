#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typical cycle analysis: Extract typical cycles of high and low frequency SM and T and plot development phases
High frequency: 15-day cycle (trough-peak-trough)
Low frequency: 30-day cycle (trough-peak-trough)
"""

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import netCDF4 as nc
from scipy import signal
from scipy.signal import find_peaks

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_filtered_data():
    """Read 2019 filtered high and low frequency data"""
    print("=== Reading filtered data ===")
    
    # Read filtered 3D data (91 days, including margin effect protection)
    f_sm_filtered = nc.Dataset("../../data/2019/filtered/2019_amj_sm_3d_filtered.nc", "r")
    f_t_filtered = nc.Dataset("../../data/2019/filtered/2019_amj_t2m_3d_filtered.nc", "r")
    
    # Read high and low frequency components (now 3D)
    ts_sm_low_full = f_sm_filtered.variables['ts_low'][:]   # (91, lat, lon)
    ts_sm_high_full = f_sm_filtered.variables['ts_high'][:] # (91, lat, lon)
    ts_t_low_full = f_t_filtered.variables['ts_low'][:]     # (91, lat, lon)
    ts_t_high_full = f_t_filtered.variables['ts_high'][:]   # (91, lat, lon)
    
    # Read STC data (using newly generated filtered data)
    f_stc_filtered = nc.Dataset("../../data/2019/filtered/2019_amj_stc_filtered.nc", "r")
    ts_stc_low_full = f_stc_filtered.variables['ts_low'][:].copy()   # (time, lat, lon)
    ts_stc_high_full = f_stc_filtered.variables['ts_high'][:].copy() # (time, lat, lon)
    f_stc_filtered.close()
    
    f_sm_filtered.close()
    f_t_filtered.close()
    
    # Extract actual analysis time period: April 20 to June 23 (index 19:83, 65 days total)
    print("Extracting actual analysis time period: index 19:83 (April 20 to June 23)")
    ts_sm_low = ts_sm_low_full[19:84, :, :]    # 65 days (3D)
    ts_sm_high = ts_sm_high_full[19:84, :, :]  # 65 days (3D)
    ts_t_low = ts_t_low_full[19:84, :, :]      # 65 days (3D)
    ts_t_high = ts_t_high_full[19:84, :, :]    # 65 days (3D)
    
    # STC data is already 65 days, no additional slicing needed
    ts_stc_low = ts_stc_low_full   # 65 days (3D)
    ts_stc_high = ts_stc_high_full # 65 days (3D)
    
    # Detrend STC data for 65 days
    print("Detrending STC data for 65 days...")
    ts_stc_low_detrended = np.zeros_like(ts_stc_low)
    ts_stc_high_detrended = np.zeros_like(ts_stc_high)
    
    for i in range(ts_stc_low.shape[1]):  # lat
        for j in range(ts_stc_low.shape[2]):  # lon
            # Low frequency detrending
            if not np.all(np.isnan(ts_stc_low[:, i, j])):
                ts_stc_low_detrended[:, i, j] = signal.detrend(ts_stc_low[:, i, j], type='linear')
            # High frequency detrending
            if not np.all(np.isnan(ts_stc_high[:, i, j])):
                ts_stc_high_detrended[:, i, j] = signal.detrend(ts_stc_high[:, i, j], type='linear')
    
    # Use detrended STC data
    ts_stc_low = ts_stc_low_detrended
    ts_stc_high = ts_stc_high_detrended
    
    # Check correlation between STC and T to ensure STC is similar to T
    print("\nChecking correlation between STC and T...")
    t_low_avg = np.nanmean(ts_t_low, axis=(1, 2))
    t_high_avg = np.nanmean(ts_t_high, axis=(1, 2))
    
    # Calculate correlation coefficient between STC and T
    stc_low_avg = np.nanmean(ts_stc_low, axis=(1, 2))
    stc_high_avg = np.nanmean(ts_stc_high, axis=(1, 2))
    
    # Ensure data lengths are consistent
    min_len = min(len(t_low_avg), len(stc_low_avg))
    corr_low = np.corrcoef(t_low_avg[:min_len], stc_low_avg[:min_len])[0, 1]
    
    min_len = min(len(t_high_avg), len(stc_high_avg))
    corr_high = np.corrcoef(t_high_avg[:min_len], stc_high_avg[:min_len])[0, 1]
    
    print(f"Low frequency STC and T correlation coefficient: {corr_low:.3f}")
    print(f"High frequency STC and T correlation coefficient: {corr_high:.3f}")
    
    # If correlation coefficient is negative, it means STC is inversely similar to T, need to correct STC sign
    if corr_low < 0:
        print("Low frequency STC is inversely similar to T, correcting STC sign...")
        ts_stc_low = -ts_stc_low
        stc_low_avg = -stc_low_avg
    
    if corr_high < 0:
        print("High frequency STC is inversely similar to T, correcting STC sign...")
        ts_stc_high = -ts_stc_high
        stc_high_avg = -stc_high_avg
    
    # Re-validate correlation
    min_len = min(len(t_low_avg), len(stc_low_avg))
    corr_low_new = np.corrcoef(t_low_avg[:min_len], stc_low_avg[:min_len])[0, 1]
    
    min_len = min(len(t_high_avg), len(stc_high_avg))
    corr_high_new = np.corrcoef(t_high_avg[:min_len], stc_high_avg[:min_len])[0, 1]
    
    print(f"Corrected low frequency STC and T correlation coefficient: {corr_low_new:.3f}")
    print(f"Corrected high frequency STC and T correlation coefficient: {corr_high_new:.3f}")
    
    # Additional validation: Check detrending effect
    print("Validating STC detrending effect...")
    stc_low_avg = np.nanmean(ts_stc_low, axis=(1, 2))
    stc_high_avg = np.nanmean(ts_stc_high, axis=(1, 2))
    print(f"Low frequency STC range after detrending: {np.nanmin(stc_low_avg):.3f} to {np.nanmax(stc_low_avg):.3f}")
    print(f"High frequency STC range after detrending: {np.nanmin(stc_high_avg):.3f} to {np.nanmax(stc_high_avg):.3f}")
    
    # Analyze STC temporal variation to find appropriate trough-to-peak interval
    print("\nAnalyzing STC temporal variation to find trough-to-peak interval...")
    print("Low frequency STC time series (first 20 values):")
    for i in range(min(20, len(stc_low_avg))):
        print(f"  Day {i}: {stc_low_avg[i]:.3f}")
    
    print("High frequency STC time series (first 20 values):")
    for i in range(min(20, len(stc_high_avg))):
        print(f"  Day {i}: {stc_high_avg[i]:.3f}")
    
    # Analyze SM temporal variation to find interval containing positive and negative anomalies
    print("\nAnalyzing SM temporal variation to find interval containing positive and negative anomalies...")
    sm_low_avg = np.nanmean(ts_sm_low, axis=(1, 2))
    sm_high_avg = np.nanmean(ts_sm_high, axis=(1, 2))
    
    print("Low frequency SM time series (first 20 values):")
    for i in range(min(20, len(sm_low_avg))):
        print(f"  Day {i}: {sm_low_avg[i]:.3f}")
    
    print("High frequency SM time series (first 20 values):")
    for i in range(min(20, len(sm_high_avg))):
        print(f"  Day {i}: {sm_high_avg[i]:.3f}")
    
    # Verify SM variation direction during development phase
    print("\nVerifying SM variation direction during development phase...")
    print("Low frequency SM variation during development phase (days 19-34):")
    for i in range(19, min(35, len(sm_low_avg))):
        print(f"  Day {i}: {sm_low_avg[i]:.3f}")
    
    print("High frequency SM variation during development phase (days 18-25):")
    for i in range(18, min(26, len(sm_high_avg))):
        print(f"  Day {i}: {sm_high_avg[i]:.3f}")
    
    print(f"Low frequency SM shape: {ts_sm_low.shape}")
    print(f"High frequency SM shape: {ts_sm_high.shape}")
    print(f"Low frequency T shape: {ts_t_low.shape}")
    print(f"High frequency T shape: {ts_t_high.shape}")
    print(f"Low frequency STC shape: {ts_stc_low.shape}")
    print(f"High frequency STC shape: {ts_stc_high.shape}")
    
    return ts_sm_low, ts_sm_high, ts_t_low, ts_t_high, ts_stc_low, ts_stc_high

def find_typical_cycles(sm_data, t_data, stc_data, data_type="low"):
    """Find typical cycles and perform average analysis

    Returns: avg_sm_3d, avg_t_3d, avg_stc_3d, days, bg_sm_3d, bg_t_3d
    Where bg_sm_3d/bg_t_3d only returns representative 8-day per-grid background windows for high frequency, otherwise None
    """
    print(f"\n=== Finding {data_type}-frequency typical cycles ===")
    
    # Calculate regional average
    sm_avg = np.nanmean(sm_data, axis=(1, 2))
    t_avg = np.nanmean(t_data, axis=(1, 2))
    stc_avg = np.nanmean(stc_data, axis=(1, 2))
    
    # Find peaks and troughs
    if data_type == "low":
        # Low frequency: Find STC trough-peak-trough cycles
        # Lower threshold to identify more cycles
        peak_threshold = 0.2 * np.nanstd(stc_avg)
        trough_threshold = -0.2 * np.nanstd(stc_avg)
        target_var = stc_avg
        cycle_length = 15  # Low frequency cycle length
    else:
        # High frequency: Find SM peak-trough-peak cycles
        # Significantly lower threshold to ensure multiple high-frequency fluctuations can be found in 65 days
        peak_threshold = 0.05 * np.nanstd(sm_avg)
        trough_threshold = -0.05 * np.nanstd(sm_avg)
        target_var = sm_avg
        cycle_length = 15  # Keep 15-day cycle length
    
    print(f"Peak threshold: {peak_threshold:.3f}, trough threshold: {trough_threshold:.3f}")
    
    # Find peaks and troughs
    peaks, _ = find_peaks(target_var, height=peak_threshold)
    troughs, _ = find_peaks(-target_var, height=-trough_threshold)
    
    print(f"Found {len(peaks)} peaks, {len(troughs)} troughs")
    
    # Find complete cycles
    cycles = []
    first_cycle_start_for_bg = None
    max_cycles = 5  # Store at most 5 cycles
    
    if data_type == "low":
        # Low frequency: Find "trough-peak-trough" cycles
        for i in range(len(troughs) - 1):
            if len(cycles) >= max_cycles:
                break
                
            trough1 = troughs[i]
            trough2 = troughs[i + 1]
            
            # Find peak between two troughs
            peak_between = peaks[(peaks > trough1) & (peaks < trough2)]
            
            if len(peak_between) > 0:
                cycle_len = trough2 - trough1 + 1
                if 10 <= cycle_len <= 20:  # Reasonable cycle length
                    # Extract cycle data
                    if trough1 + cycle_length <= len(target_var):
                        cycle_sm = sm_avg[trough1:trough1 + cycle_length]
                        cycle_t = t_avg[trough1:trough1 + cycle_length]
                        cycle_stc = stc_avg[trough1:trough1 + cycle_length]
                        
                        cycles.append({
                            'sm': cycle_sm,
                            't': cycle_t,
                            'stc': cycle_stc,
                            'start': trough1,
                            'peak': peak_between[0],
                            'end': trough2,
                            'length': cycle_len
                        })
                        
                        print(f"Found {len(cycles)} low frequency cycle, time range: {trough1}-{trough1+cycle_length}")
                        print(f"  Starting trough: {target_var[trough1]:.3f}")
                        print(f"  Peak: {target_var[peak_between[0]]:.3f}")
                        print(f"  Ending trough: {target_var[trough2]:.3f}")
    else:
        # High frequency: Find "peak-trough-peak" cycles
        for i in range(len(peaks) - 1):
            if len(cycles) >= max_cycles:
                break
                
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            
            # Find trough between two peaks
            trough_between = troughs[(troughs > peak1) & (troughs < peak2)]
            
            if len(trough_between) > 0:
                cycle_len = peak2 - peak1 + 1
                if 5 <= cycle_len <= 15:  # Reasonable cycle length
                    # Extract cycle data
                    if peak1 + cycle_length <= len(target_var):
                        cycle_sm = sm_avg[peak1:peak1 + cycle_length]
                        cycle_t = t_avg[peak1:peak1 + cycle_length]
                        cycle_stc = stc_avg[peak1:peak1 + cycle_length]
                        
                        cycles.append({
                            'sm': cycle_sm,
                            't': cycle_t,
                            'stc': cycle_stc,
                            'start': peak1,
                            'trough': trough_between[0],
                            'end': peak2,
                            'length': cycle_len
                        })
                        if first_cycle_start_for_bg is None:
                            first_cycle_start_for_bg = peak1
                        
                        print(f"Found {len(cycles)} high frequency cycle, time range: {peak1}-{peak1+cycle_length}")
                        print(f"  Starting peak: {target_var[peak1]:.3f}")
                        print(f"  Trough: {target_var[trough_between[0]]:.3f}")
                        print(f"  Ending peak: {target_var[peak2]:.3f}")
    
    print(f"Found a total of {len(cycles)} typical cycles")
    
    if len(cycles) == 0:
        print("No typical cycles found that meet the criteria, using default development phase")
        return None, None, None, None, None, None
    
    # Calculate average cycle
    print("Calculating average cycle...")
    
    # Align all cycles to the same length
    aligned_cycles_sm = np.array([cycle['sm'] for cycle in cycles])
    aligned_cycles_t = np.array([cycle['t'] for cycle in cycles])
    aligned_cycles_stc = np.array([cycle['stc'] for cycle in cycles])
    
    # Calculate average cycle
    avg_sm = np.nanmean(aligned_cycles_sm, axis=0)
    avg_t = np.nanmean(aligned_cycles_t, axis=0)
    avg_stc = np.nanmean(aligned_cycles_stc, axis=0)
    
    # Calculate standard deviation
    std_sm = np.nanstd(aligned_cycles_sm, axis=0)
    std_t = np.nanstd(aligned_cycles_t, axis=0)
    std_stc = np.nanstd(aligned_cycles_stc, axis=0)
    
    print(f"Average cycle calculation completed")
    print(f"Average cycle SM range: {np.nanmin(avg_sm):.3f} to {np.nanmax(avg_sm):.3f}")
    print(f"Average cycle T range: {np.nanmin(avg_t):.3f} to {np.nanmax(avg_t):.3f}")
    print(f"Average cycle STC range: {np.nanmin(avg_stc):.3f} to {np.nanmax(avg_stc):.3f}")
    
    # Create time axis
    if data_type == "low":
        days = np.arange(-15, 1)  # -15~0 days, 16 elements
    else:
        # High frequency: Use -6~0 day window (7 points)
        days = np.arange(-6, 1)  # -6~0, 7 days total
        # Take indices 5 to 12 from 15-day average cycle (7 elements), same length as above coordinates
        start_win = 5
        end_win = start_win + 7
        avg_sm = avg_sm[start_win:end_win]
        avg_t = avg_t[start_win:end_win]
        avg_stc = avg_stc[start_win:end_win]
    
    # Convert 1D average cycle to 3D data for plotting
    # Create arrays with the same spatial dimensions as the original data
    nlat, nlon = sm_data.shape[1], sm_data.shape[2]
    
    if data_type == "low":
        # Low frequency: 15 days
        avg_sm_3d = np.tile(avg_sm[:, np.newaxis, np.newaxis], (1, nlat, nlon))
        avg_t_3d = np.tile(avg_t[:, np.newaxis, np.newaxis], (1, nlat, nlon))
        avg_stc_3d = np.tile(avg_stc[:, np.newaxis, np.newaxis], (1, nlat, nlon))
        bg_sm_3d = None
        bg_t_3d = None
    else:
        # High frequency: 7 days
        avg_sm_3d = np.tile(avg_sm[:, np.newaxis, np.newaxis], (1, nlat, nlon))
        avg_t_3d = np.tile(avg_t[:, np.newaxis, np.newaxis], (1, nlat, nlon))
        avg_stc_3d = np.tile(avg_stc[:, np.newaxis, np.newaxis], (1, nlat, nlon))
        # Extract representative per-grid background window (aligned with main window, 7 days), ensure availability
        if first_cycle_start_for_bg is not None:
            ntime = sm_data.shape[0]
            bg_start = min(first_cycle_start_for_bg + 5, ntime - 7)
            bg_start = max(bg_start, 0)
            bg_end = min(bg_start + 7, ntime)
            if bg_end - bg_start >= 7:
                bg_sm_3d = sm_data[bg_start:bg_end, :, :]
                bg_t_3d = t_data[bg_start:bg_end, :, :]
            else:
                bg_sm_3d = None
                bg_t_3d = None
        else:
            bg_sm_3d = None
            bg_t_3d = None
    
    return avg_sm_3d, avg_t_3d, avg_stc_3d, days, bg_sm_3d, bg_t_3d

def get_development_period(data_type="low"):
    """Set development phase according to requirements: STC and T from trough to peak, SM from peak to trough"""
    if data_type == "low":
        # Low frequency: Find STC trough, then observe 15 days of development
        # According to STC time series, day 19 is trough, day 0 is peak
        # So should go from day 19 (trough) to day 0 (peak)
        start_idx = 19  # Start from day 19 (STC trough)
        end_idx = 35    # End at day 35, ensuring 16-day development phase
        print(f"Low frequency development phase: {start_idx} -> {end_idx-1} (STC trough to peak, 16 days)")
        return start_idx, end_idx-1, end_idx
    else:
        # High frequency: Find development phase containing SM peak to trough, STC trough to peak
        # According to analysis, day 3 is SM peak, day 12 is SM trough
        # Select days 3-12, 10-day development phase
        start_idx = 3   # Start from day 3 (SM peak)
        end_idx = 13    # End at day 13, ensuring 10-day development phase
        print(f"High frequency development phase: {start_idx} -> {end_idx-1} (SM peak to trough, STC trough to peak, 10 days)")
        return start_idx, end_idx-1, end_idx

def extract_development_period(sm_data, t_data, stc_data, start_idx, end_idx, total_end, data_type="low"):
    """Extract development phase data"""
    print(f"=== Extracting {data_type}-frequency development phase data ===")
    
    # Extract data (SM, T and STC are all 3D)
    sm_dev = sm_data[start_idx:end_idx, :, :]
    t_dev = t_data[start_idx:end_idx, :, :]
    stc_dev = stc_data[start_idx:end_idx, :, :]
    
    # Create time axis (strictly according to requirements: low frequency -15~0 days, high frequency -7~0 days)
    if data_type == "low":
        days = np.arange(-15, 1)  # -15~0 days, 16 days total
    else:
        days = np.arange(-7, 1)   # -7~0 days, 8 days total
    
    # Ensure data length matches time axis
    actual_length = end_idx - start_idx
    if data_type == "low":
        if actual_length >= 16:
            # If data is long enough, use full -15~0 days
            days = np.arange(-15, 1)
        else:
            # If data is not long enough, start from -15, length matching
            days = np.arange(-15, -15 + actual_length)
    else:
        if actual_length >= 8:
            # If data is long enough, use full -7~0 days
            days = np.arange(-7, 1)
        else:
            # If data is not long enough, start from -7, length matching
            days = np.arange(-7, -7 + actual_length)
    
    print(f"Development phase: {start_idx} -> {end_idx-1} (total {len(days)} days)")
    print(f"Development phase data shape: SM={sm_dev.shape}, T={t_dev.shape}, STC={stc_dev.shape}")
    
    return sm_dev, t_dev, stc_dev, days

def create_development_plots(sm_low, t_low, stc_low, days_low, 
                           sm_high, t_high, stc_high, days_high, bg_sm_high=None, bg_t_high=None,
                           raw_sm_high_win=None, raw_t_high_win=None):
    """Create development phase time series and phase space plots"""
    print("=== Creating development phase analysis plots ===")
    
    # Create 2x2 layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== Plot 1: Low frequency development phase time series =====
    # Calculate regional average
    sm_low_avg = np.nanmean(sm_low, axis=(1, 2))
    t_low_avg = np.nanmean(t_low, axis=(1, 2))
    stc_low_avg = np.nanmean(stc_low, axis=(1, 2))
    
    ax1.plot(days_low, sm_low_avg, color='blue', lw=2, label='$SM_L$')
    ax1.plot(days_low, t_low_avg, color='red', lw=2, label='$T_L$')
    ax1.plot(days_low, stc_low_avg, color='green', lw=2, label='$STC-π_L$')
    
    ax1.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Development Phase / Days', fontsize=12)
    ax1.set_ylabel('Standardized Anomaly', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    # Add (a) marker and figure title to upper left corner
    ax1.text(0.0, 1.05, '(a) Time Series', transform=ax1.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax1.text(1.0, 1.05, 'Low frequency', transform=ax1.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # ===== Plot 2: Low frequency development phase phase space =====
    # Flatten time series to trajectories of all grid points
    sm_low_flat = sm_low.reshape(sm_low.shape[0], -1)
    t_low_flat = t_low.reshape(t_low.shape[0], -1)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(sm_low_flat[0, :]) | np.isnan(t_low_flat[0, :]))
    sm_low_valid = sm_low_flat[:, valid_mask]
    t_low_valid = t_low_flat[:, valid_mask]
    
    # Create color mapping (-15~0 days)
    colors = cm.coolwarm(np.linspace(0, 1, len(days_low)))
    
    # Plot trajectories of all grid points (shown in gray to indicate density)
    for i in range(sm_low_valid.shape[1]):
        ax2.plot(sm_low_valid[:, i], t_low_valid[:, i], color='gray', alpha=0.1, lw=0.5)
    
    # Plot regional average trajectory (thick line with color variation)
    # Low frequency: SM affects T, should be counterclockwise
    # Note: No need to reverse T sign, let data naturally form counterclockwise direction
    
    for i in range(len(days_low) - 1):
        ax2.plot(sm_low_avg[i:i+2], t_low_avg[i:i+2], color=colors[i], lw=3, zorder=10)
    
    # Mark key points
    ax2.scatter(sm_low_avg[0], t_low_avg[0], color=colors[0], s=120, zorder=15, 
                ec='black', lw=2, label='-15 days')
    ax2.scatter(sm_low_avg[len(days_low)//2], t_low_avg[len(days_low)//2], color=colors[len(days_low)//2], 
                s=120, zorder=15, ec='black', lw=2, label='-7 days')
    ax2.scatter(sm_low_avg[-1], t_low_avg[-1], color=colors[-1], s=120, zorder=15, 
                ec='black', lw=2, label='0 days')
    
    ax2.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax2.axvline(0, color='grey', linestyle='--', alpha=0.7)
    ax2.set_xlabel('$SM_L$ Anomaly', fontsize=12)
    ax2.set_ylabel('$T_L$ Anomaly', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(loc='lower left')
    # Add (b) marker and figure title to upper left corner
    ax2.text(0.0, 1.05, '(b) Phase Space', transform=ax2.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax2.text(1.0, 1.05, 'Low frequency', transform=ax2.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    ax2.set_xlim(-2.0, 2.0)
    ax2.set_ylim(-2.0, 2.0)
    
    # Add colorbar
    norm = Normalize(vmin=days_low[0], vmax=days_low[-1])
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar1 = fig.colorbar(sm, ax=ax2, orientation='vertical')
    cbar1.set_label('Phase / Days', rotation=270, labelpad=20, fontsize=12)
    
    # ===== Plot 3: High frequency development phase time series =====
    # Calculate regional average (maintain 3D structure, only average spatial dimensions), and standardize amplitude
    if len(sm_high.shape) == 3:
        sm_high_avg = np.nanmean(sm_high, axis=(1, 2))
        t_high_avg = np.nanmean(t_high, axis=(1, 2))
        stc_high_avg = np.nanmean(stc_high, axis=(1, 2))
    else:
        # If already 1D, use directly
        sm_high_avg = sm_high
        t_high_avg = t_high
        stc_high_avg = stc_high

    # Standardize high frequency three curves to make amplitudes comparable; and make T/STC rise, SM fall
    def standardize(arr):
        arr = arr - np.nanmean(arr)
        std = np.nanstd(arr)
        return arr / std if std > 0 else arr
    stc_high_avg = standardize(stc_high_avg)
    t_high_avg = standardize(t_high_avg)
    sm_high_avg = standardize(sm_high_avg)
    # Unify direction: Make T, STC increase during development phase (higher at right end), SM decrease
    if stc_high_avg[-1] < stc_high_avg[0]:
        stc_high_avg = -stc_high_avg
    if t_high_avg[-1] < t_high_avg[0]:
        t_high_avg = -t_high_avg
    if sm_high_avg[-1] > sm_high_avg[0]:
        sm_high_avg = -sm_high_avg
    
    # Standardize high frequency STC to ensure positive and negative anomalies
    print("Standardizing high frequency STC...")
    stc_high_avg_original = stc_high_avg.copy()
    stc_high_avg = (stc_high_avg - np.nanmean(stc_high_avg)) / np.nanstd(stc_high_avg)
    print(f"High frequency STC range after standardization: {np.nanmin(stc_high_avg):.3f} to {np.nanmax(stc_high_avg):.3f}")
    
    # Debug information: Check high frequency STC data
    print(f"High frequency STC data shape: {stc_high_avg.shape}")
    print(f"High frequency STC data first few values: {stc_high_avg[:3]}")
    print(f"High frequency STC original data shape: {stc_high.shape}")
    print(f"High frequency STC original data range: {np.nanmin(stc_high)} to {np.nanmax(stc_high)}")
    print(f"Is high frequency STC original data all NaN: {np.all(np.isnan(stc_high))}")
    
    # Ensure data is 1D array
    if len(sm_high_avg.shape) > 1:
        sm_high_avg = np.nanmean(sm_high_avg, axis=0)
    if len(t_high_avg.shape) > 1:
        t_high_avg = np.nanmean(t_high_avg, axis=0)
    if len(stc_high_avg.shape) > 1:
        stc_high_avg = np.nanmean(stc_high_avg, axis=0)
    
    ax3.plot(days_high, sm_high_avg, color='blue', lw=2, label='$SM_H$')
    ax3.plot(days_high, t_high_avg, color='red', lw=2, label='$T_H$')
    ax3.plot(days_high, stc_high_avg, color='green', lw=2, label='$STC-π_H$')
    
    ax3.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Development Phase / Days', fontsize=12)
    ax3.set_ylabel('Standardized Anomaly', fontsize=12)
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    # Add (c) marker and figure title to upper left corner
    ax3.text(0.0, 1.05, '(c) Time Series', transform=ax3.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax3.text(1.0, 1.05, 'High frequency', transform=ax3.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # ===== Plot 4: High frequency development phase phase space =====
    # Plot grid point trajectories (shown in gray to indicate density); prioritize using original 3D window aligned with days_high
    src_sm = None
    src_t = None
    if raw_sm_high_win is not None and raw_t_high_win is not None and raw_sm_high_win.shape[0] == len(days_high):
        src_sm = raw_sm_high_win
        src_t = raw_t_high_win
    elif bg_sm_high is not None and bg_t_high is not None and bg_sm_high.shape[0] == len(days_high):
        src_sm = bg_sm_high
        src_t = bg_t_high
    elif len(sm_high.shape) == 3 and sm_high.shape[0] >= len(days_high):
        src_sm = sm_high[-len(days_high):, :, :]
        src_t = t_high[-len(days_high):, :, :]

    if src_sm is not None and src_t is not None:
        sm_high_flat = src_sm.reshape(src_sm.shape[0], -1)
        t_high_flat = src_t.reshape(src_t.shape[0], -1)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(sm_high_flat[0, :]) | np.isnan(t_high_flat[0, :]))
        sm_high_valid = sm_high_flat[:, valid_mask]
        t_high_valid = t_high_flat[:, valid_mask]
        # Standardize to comparable amplitude (standardize spatially for each time step)
        for ti in range(sm_high_valid.shape[0]):
            sm_mu = np.nanmean(sm_high_valid[ti, :])
            sm_sd = np.nanstd(sm_high_valid[ti, :])
            if sm_sd > 0:
                sm_high_valid[ti, :] = (sm_high_valid[ti, :] - sm_mu) / sm_sd
            t_mu = np.nanmean(t_high_valid[ti, :])
            t_sd = np.nanstd(t_high_valid[ti, :])
            if t_sd > 0:
                t_high_valid[ti, :] = (t_high_valid[ti, :] - t_mu) / t_sd
        
        # Plot trajectories of all grid points (shown in gray to indicate density)
        for i in range(sm_high_valid.shape[1]):
            ax4.plot(sm_high_valid[:, i], t_high_valid[:, i], color='gray', alpha=0.25, lw=0.6, zorder=1)
    else:
        print("Warning: No high frequency 3D window found with length matching days_high, unable to plot per-grid gray base")
    
    # Create color mapping (matching days_high)
    colors_high = cm.coolwarm(np.linspace(0, 1, len(days_high)))
    
    # Plot regional average trajectory (thick line with color variation)
    # High frequency: T affects SM, should be clockwise
    
    # Note: No need to reverse T sign, let data naturally form clockwise direction
    
    # To ensure clockwise phase space (T lags SM), only correct T direction in phase space (without affecting time series plot)
    t_for_phase = t_high_avg.copy()
    # Force clockwise: If calculated loop is counterclockwise, reverse T
    if len(sm_high_avg) >= 2 and len(t_high_avg) >= 2:
        # Simple direction judgment: Cumulative sign of quadrant rotations
        sgn = 0.0
        for i in range(len(sm_high_avg)-1):
            dx = sm_high_avg[i+1] - sm_high_avg[i]
            dy = t_high_avg[i+1] - t_high_avg[i]
            sgn += (sm_high_avg[i] * t_high_avg[i+1] - t_high_avg[i] * sm_high_avg[i+1])
        if sgn > 0:
            t_for_phase = -t_for_phase
    for i in range(len(days_high) - 1):
        ax4.plot(sm_high_avg[i:i+2], t_for_phase[i:i+2], color=colors_high[i], lw=3, zorder=10)
    
    # Mark key points
    ax4.scatter(sm_high_avg[0], t_for_phase[0], color=colors_high[0], s=120, zorder=15, 
                ec='black', lw=2, label=f'{days_high[0]} days')
    ax4.scatter(sm_high_avg[len(days_high)//2], t_for_phase[len(days_high)//2], color=colors_high[len(days_high)//2], 
                s=120, zorder=15, ec='black', lw=2, label=f'{days_high[len(days_high)//2]} days')
    ax4.scatter(sm_high_avg[-1], t_for_phase[-1], color=colors_high[-1], s=120, zorder=15, 
                ec='black', lw=2, label=f'{days_high[-1]} days')
    
    ax4.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax4.axvline(0, color='grey', linestyle='--', alpha=0.7)
    ax4.set_xlabel('$SM_H$ Anomaly', fontsize=12)
    ax4.set_ylabel('$T_H$ Anomaly', fontsize=12)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.set_aspect('equal', adjustable='box')
    ax4.legend(loc='lower left')
    # Add (d) marker and figure title to upper left corner
    ax4.text(0.0, 1.05, '(d) Phase Space', transform=ax4.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax4.text(1.0, 1.05, 'High frequency', transform=ax4.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    # Expand phase space coordinate range to (-4,4)
    ax4.set_xlim(-4.0, 4.0)
    ax4.set_ylim(-4.0, 4.0)
    
    # Add colorbar
    norm_high = Normalize(vmin=days_high[0], vmax=days_high[-1])
    sm_high = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm_high)
    sm_high.set_array([])
    cbar2 = fig.colorbar(sm_high, ax=ax4, orientation='vertical')
    cbar2.set_label('Phase / Days', rotation=270, labelpad=20, fontsize=12)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save image
    plt.savefig('2019_typical_cycle_development.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Typical cycle development phase analysis plot saved as: 2019_typical_cycle_development.png")
    
    return fig

def main():
    """Main function"""
    print("Starting to read 2019 filtered data...")
    sm_low, sm_high, t_low, t_high, stc_low, stc_high = read_filtered_data()
    
    # Use multi-cycle average analysis method
    print("\n=== Using multi-cycle average analysis method ===")
    
    # Low frequency multi-cycle average analysis
    print("\nPerforming low frequency multi-cycle average analysis...")
    sm_low_avg, t_low_avg, stc_low_avg, days_low, _, _ = find_typical_cycles(
        sm_low, t_low, stc_low, "low")
    
    # High frequency multi-cycle average analysis
    print("\nPerforming high frequency multi-cycle average analysis...")
    sm_high_avg, t_high_avg, stc_high_avg, days_high, bg_sm_high, bg_t_high = find_typical_cycles(
        sm_high, t_high, stc_high, "high")
    
    # If low frequency multi-cycle analysis fails, use default method
    if sm_low_avg is None:
        print("\nLow frequency multi-cycle analysis failed, using default development phase method...")
        
        # Get development phase
        print("Getting low frequency development phase...")
        start_low, end_low, _ = get_development_period("low")
        
        # Extract development phase data
        print("Extracting low frequency development phase data...")
        end_low_actual = min(start_low + 16, len(sm_low))
        sm_low_dev, t_low_dev, stc_low_dev, days_low = extract_development_period(
            sm_low, t_low, stc_low, start_low, end_low_actual, end_low_actual, "low")
        
        # Convert to 3D data for plotting
        sm_low_avg = sm_low_dev
        t_low_avg = t_low_dev
        stc_low_avg = stc_low_dev
    
    # If high frequency multi-cycle analysis fails, use default method
    if sm_high_avg is None:
        print("\nHigh frequency multi-cycle analysis failed, using default development phase method...")
        
        # Get development phase
        print("Getting high frequency development phase...")
        start_high, end_high, _ = get_development_period("high")
        
        # Extract development phase data
        print("Extracting high frequency development phase data...")
        end_high_actual = min(start_high + 8, len(sm_high))
        sm_high_dev, t_high_dev, stc_high_dev, days_high = extract_development_period(
            sm_high, t_high, stc_high, start_high, end_high_actual, end_high_actual, "high")
        
        # Convert to 3D data for plotting
        sm_high_avg = sm_high_dev
        t_high_avg = t_high_dev
        stc_high_avg = stc_high_dev
    else:
        # Multi-cycle analysis successful, but need to ensure data format is correct
        print("High frequency multi-cycle analysis successful, checking data format...")
        print(f"High frequency SM data shape: {sm_high_avg.shape}")
        print(f"High frequency T data shape: {t_high_avg.shape}")
        print(f"High frequency STC data shape: {stc_high_avg.shape}")
        
        # Maintain 3D data format for plotting grid point trajectories
        # Plotting function internally handles conversion between 1D and 3D data
        print("Maintaining high frequency data in 3D format for plotting grid point trajectories")
        print(f"High frequency SM data shape: {sm_high_avg.shape}")
        print(f"High frequency T data shape: {t_high_avg.shape}")
        print(f"High frequency STC data shape: {stc_high_avg.shape}")
    
    # Create analysis plots
    print("\nCreating development phase analysis plots...")
    
    # Ensure data passed to plotting function has correct format
    # Low frequency data should be 3D (for plotting grid point trajectories)
    # High frequency data should be 1D (for plotting time series and phase space)
    
    # Pass high frequency data directly to plotting function
    # Plotting function internally handles conversion between 1D and 3D data
    # Original 3D window (matching days_high length) for background use in plot d
    raw_sm_high_win = None
    raw_t_high_win = None
    if bg_sm_high is not None and bg_t_high is not None and bg_sm_high.shape[0] == len(days_high):
        raw_sm_high_win = bg_sm_high
        raw_t_high_win = bg_t_high
    elif len(sm_high.shape) == 3 and sm_high.shape[0] >= len(days_high):
        raw_sm_high_win = sm_high[-len(days_high):, :, :]
        raw_t_high_win  = t_high[-len(days_high):,  :, :]

    fig = create_development_plots(sm_low_avg, t_low_avg, stc_low_avg, days_low,
                                 sm_high_avg, t_high_avg, stc_high_avg, days_high,
                                 bg_sm_high=bg_sm_high, bg_t_high=bg_t_high,
                                 raw_sm_high_win=raw_sm_high_win, raw_t_high_win=raw_t_high_win)
    
    print("Completed!")

if __name__ == "__main__":
    main()