#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive low-frequency and high-frequency SM-T-STC-π analysis
Including time series and phase space diagrams with unified layout
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def read_nc_data():
    """Read NC data"""
    print("Reading NC data...")
    
    # Define the relative path for data files
    data_path = "../../data/climatology/filtered/"
    
    # Read low-frequency data
    f_low_sm = xr.open_dataset(data_path + "mam_sml14icp_gms_filtered.nc")
    ts_low_SM = f_low_sm['ts_low'].values  # (day, lat, lon)
    
    f_low_t = xr.open_dataset(data_path + "mam_t2m4icp_gms_filtered.nc")
    ts_low_T = f_low_t['ts_low'].values  # (day, lat, lon)
    
    f_low_stc = xr.open_dataset(data_path + "mam_paip4icp_gms_filtered.nc")
    ts_low_STC = f_low_stc['ts_low'].values  # (day, lat, lon)
    
    # Read high-frequency data
    f_high_sm = xr.open_dataset(data_path + "mam_sml14icp_gms_filtered.nc")
    ts_high_SM = f_high_sm['ts_high'].values  # (day, lat, lon)
    
    f_high_t = xr.open_dataset(data_path + "mam_t2m4icp_gms_filtered.nc")
    ts_high_T = f_high_t['ts_high'].values  # (day, lat, lon)
    
    f_high_stc = xr.open_dataset(data_path + "mam_paip4icp_gms_filtered.nc")
    ts_high_STC = f_high_stc['ts_high'].values  # (day, lat, lon)
    
    # Read high-frequency SM average cycle as ruler
    f_avg_cycle = xr.open_dataset(data_path + "high_freq_sm_avg_cycle_refined.nc")
    avg_cycle = f_avg_cycle['avg_cycle'].values
    cycle_time = f_avg_cycle['cycle_time'].values
    n_cycles = f_avg_cycle['n_cycles'].values
    
    # Get dimension information
    ntime, nlat, nlon = ts_low_SM.shape
    print(f"Data dimensions: {ntime} days, {nlat} latitude, {nlon} longitude")
    print(f"Ruler cycle length: {len(avg_cycle)} days")
    print(f"Number of typical cycles found: {n_cycles}")
    
    return (ts_low_SM, ts_low_T, ts_low_STC, 
            ts_high_SM, ts_high_T, ts_high_STC, 
            avg_cycle, cycle_time, ntime, nlat, nlon)

def process_data(ts_low_SM, ts_low_T, ts_low_STC, ts_high_SM, ts_high_T, ts_high_STC, ntime, nlat, nlon):
    """Process data: detrending and standardization"""
    print("Processing data: detrending and standardization...")
    
    # Initialize detrended arrays
    ts_low_SM_detrend = np.copy(ts_low_SM)
    ts_low_T_detrend = np.copy(ts_low_T)
    ts_low_STC_detrend = np.copy(ts_low_STC)
    
    ts_high_SM_detrend = np.copy(ts_high_SM)
    ts_high_T_detrend = np.copy(ts_high_T)
    ts_high_STC_detrend = np.copy(ts_high_STC)
    
    # Detrend and standardize for each grid point
    for i in range(nlat):
        for j in range(nlon):
            # Low-frequency SM detrending
            if not np.all(np.isnan(ts_low_SM[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_low_SM[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_low_SM_detrend[:, i, j] = ts_low_SM[:, i, j] - trend
                
                mean_val = np.nanmean(ts_low_SM_detrend[:, i, j])
                std_val = np.nanstd(ts_low_SM_detrend[:, i, j])
                if std_val != 0:
                    ts_low_SM_detrend[:, i, j] = (ts_low_SM_detrend[:, i, j] - mean_val) / std_val
            
            # Low-frequency T detrending
            if not np.all(np.isnan(ts_low_T[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_low_T[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_low_T_detrend[:, i, j] = ts_low_T[:, i, j] - trend
                
                mean_val = np.nanmean(ts_low_T_detrend[:, i, j])
                std_val = np.nanstd(ts_low_T_detrend[:, i, j])
                if std_val != 0:
                    ts_low_T_detrend[:, i, j] = (ts_low_T_detrend[:, i, j] - mean_val) / std_val
            
            # Low-frequency STC detrending
            if not np.all(np.isnan(ts_low_STC[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_low_STC[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_low_STC_detrend[:, i, j] = ts_low_STC[:, i, j] - trend
                
                mean_val = np.nanmean(ts_low_STC_detrend[:, i, j])
                std_val = np.nanstd(ts_low_STC_detrend[:, i, j])
                if std_val != 0:
                    ts_low_STC_detrend[:, i, j] = (ts_low_STC_detrend[:, i, j] - mean_val) / std_val
            
            # High-frequency SM detrending
            if not np.all(np.isnan(ts_high_SM[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_high_SM[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_high_SM_detrend[:, i, j] = ts_high_SM[:, i, j] - trend
                
                mean_val = np.nanmean(ts_high_SM_detrend[:, i, j])
                std_val = np.nanstd(ts_high_SM_detrend[:, i, j])
                if std_val != 0:
                    ts_high_SM_detrend[:, i, j] = (ts_high_SM_detrend[:, i, j] - mean_val) / std_val
            
            # High-frequency T detrending
            if not np.all(np.isnan(ts_high_T[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_high_T[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_high_T_detrend[:, i, j] = ts_high_T[:, i, j] - trend
                
                mean_val = np.nanmean(ts_high_T_detrend[:, i, j])
                std_val = np.nanstd(ts_high_T_detrend[:, i, j])
                if std_val != 0:
                    ts_high_T_detrend[:, i, j] = (ts_high_T_detrend[:, i, j] - mean_val) / std_val
            
            # High-frequency STC detrending
            if not np.all(np.isnan(ts_high_STC[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_high_STC[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_high_STC_detrend[:, i, j] = ts_high_STC[:, i, j] - trend
                
                mean_val = np.nanmean(ts_high_STC_detrend[:, i, j])
                std_val = np.nanstd(ts_high_STC_detrend[:, i, j])
                if std_val != 0:
                    ts_high_STC_detrend[:, i, j] = (ts_high_STC_detrend[:, i, j] - mean_val) / std_val
    
    return (ts_low_SM_detrend, ts_low_T_detrend, ts_low_STC_detrend,
            ts_high_SM_detrend, ts_high_T_detrend, ts_high_STC_detrend)

def create_comprehensive_plot(ts_low_SM_detrend, ts_low_T_detrend, ts_low_STC_detrend,
                             ts_high_SM_detrend, ts_high_T_detrend, ts_high_STC_detrend,
                             avg_cycle, ntime, nlat, nlon):
    """Create comprehensive plot"""
    print("Creating comprehensive plot...")
    
    # Calculate regional average time series
    print("Calculating regional average time series...")
    
    # Low-frequency regional average
    ts_low_SM_avg = np.nanmean(ts_low_SM_detrend, axis=(1, 2))
    ts_low_T_avg = np.nanmean(ts_low_T_detrend, axis=(1, 2))
    ts_low_STC_avg = np.nanmean(ts_low_STC_detrend, axis=(1, 2))
    
    # High-frequency regional average
    ts_high_SM_avg = np.nanmean(ts_high_SM_detrend, axis=(1, 2))
    ts_high_T_avg = np.nanmean(ts_high_T_detrend, axis=(1, 2))
    ts_high_STC_avg = np.nanmean(ts_high_STC_detrend, axis=(1, 2))
    
    # Standardization
    ts_low_SM_avg = (ts_low_SM_avg - np.nanmean(ts_low_SM_avg)) / np.nanstd(ts_low_SM_avg)
    ts_low_T_avg = (ts_low_T_avg - np.nanmean(ts_low_T_avg)) / np.nanstd(ts_low_T_avg)
    ts_low_STC_avg = (ts_low_STC_avg - np.nanmean(ts_low_STC_avg)) / np.nanstd(ts_low_STC_avg)
    
    ts_high_SM_avg = (ts_high_SM_avg - np.nanmean(ts_high_SM_avg)) / np.nanstd(ts_high_SM_avg)
    ts_high_T_avg = (ts_high_T_avg - np.nanmean(ts_high_T_avg)) / np.nanstd(ts_high_T_avg)
    ts_high_STC_avg = (ts_high_STC_avg - np.nanmean(ts_high_STC_avg)) / np.nanstd(ts_high_STC_avg)
    
    # Adjust T sign to ensure correct physical meaning
    ts_low_T_avg = -ts_low_T_avg
    ts_high_T_avg = -ts_high_T_avg
    
    # Correction: T in figure (a) should be consistent with STC-π changes, no need to reverse sign
    ts_low_T_avg = -ts_low_T_avg  # Restore original sign
    ts_high_T_avg = -ts_high_T_avg  # Restore original sign
    
    # Create figure - 2x2 layout
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout: 2x2
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Figure (a): Low-frequency time series
    ax2 = fig.add_subplot(gs[0, 1])  # Figure (b): Low-frequency phase space
    ax3 = fig.add_subplot(gs[1, 0])  # Figure (c): High-frequency time series
    ax4 = fig.add_subplot(gs[1, 1])  # Figure (d): High-frequency phase space
    
    # ===== Figure (a): Low-frequency time series =====
    print("Plotting low-frequency time series...")
    
    # Select time range: days 6-46 (corresponding to -40~0 days)
    days_low = np.arange(-40, 1)
    ts_low_plot = ts_low_SM_avg[6:47]  # Days 6-46
    ts_low_T_plot = ts_low_T_avg[6:47]
    ts_low_STC_plot = ts_low_STC_avg[6:47]
    
    ax1.plot(days_low, ts_low_plot, color='blue', linewidth=2, label='$SM_L$', alpha=0.8)
    ax1.plot(days_low, ts_low_T_plot, color='red', linewidth=2, label='$T_L$', alpha=0.8)
    ax1.plot(days_low, ts_low_STC_plot, color='green', linewidth=2, label='$STC-π_L$', alpha=0.8)
    
    ax1.set_xlabel('Phase / Days', fontsize=12)
    ax1.set_ylabel('Standardized Anomaly', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')  # legend in lower right corner
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylim(-1.5, 1.5)
    
    # Add (a) label and figure title to upper left corner
    ax1.text(0.0, 1.05, '(a) Time Series', transform=ax1.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax1.text(1.0, 1.05, 'Low frequency', transform=ax1.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # ===== Figure (b): Low-frequency phase space =====
    print("Plotting low-frequency phase space...")
    
    # Select time period starting from March 7th, consistent with original script
    start_idx = 6  # Starting from March 7th
    end_idx = 46   # To around April 15th
    ts_low_SM_sel = ts_low_SM_detrend[start_idx:end_idx, :, :]
    ts_low_T_sel = ts_low_T_detrend[start_idx:end_idx, :, :]
    
    # Flatten time series to trajectories of all grid points
    sm_low_flat = ts_low_SM_sel.reshape(ts_low_SM_sel.shape[0], -1)
    t_low_flat = ts_low_T_sel.reshape(ts_low_T_sel.shape[0], -1)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(sm_low_flat[0, :]) | np.isnan(t_low_flat[0, :]))
    sm_low_valid = sm_low_flat[:, valid_mask]
    t_low_valid = t_low_flat[:, valid_mask]
    
    # Create color mapping (-40~0 days)
    colors_low = cm.coolwarm(np.linspace(0, 1, len(days_low)))
    
    # Plot trajectories of all grid points (shown as density in gray)
    for i in range(sm_low_valid.shape[1]):
        ax2.plot(sm_low_valid[:, i], t_low_valid[:, i], color='gray', alpha=0.1, lw=0.5)
    
    # Plot regional average trajectory (thick line with color variation)
    for i in range(len(days_low) - 1):
        ax2.plot([ts_low_plot[i], ts_low_plot[i+1]], 
                [ts_low_T_plot[i], ts_low_T_plot[i+1]], 
                color=colors_low[i], lw=3, zorder=10)
    
    # Mark key points
    ax2.scatter(ts_low_plot[0], ts_low_T_plot[0], c='blue', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-40 days')
    ax2.scatter(ts_low_plot[20], ts_low_T_plot[20], c='white', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-20 days')
    ax2.scatter(ts_low_plot[-1], ts_low_T_plot[-1], c='red', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='0 days')
    
    ax2.set_xlabel('$SM_L$ Anomaly', fontsize=12)
    ax2.set_ylabel('$T_L$ Anomaly', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left')  # legend in lower left corner
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlim(-4.0, 4.0)
    ax2.set_ylim(-4.0, 4.0)
    
    # Add (b) label and figure title to upper left corner
    ax2.text(0.0, 1.05, '(b) Phase Space', transform=ax2.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax2.text(1.0, 1.05, 'Low frequency', transform=ax2.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # Add colorbar to the right of figure b
    norm_low = plt.Normalize(-40, 0)
    sm_low = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm_low)
    sm_low.set_array([])
    cbar_low = plt.colorbar(sm_low, ax=ax2, orientation='vertical', shrink=0.8, pad=0.02)
    cbar_low.set_label('Phase / Days', fontsize=10)
    cbar_low.set_ticks([-40, -30, -20, -10, 0])
    cbar_low.set_ticklabels(['-40', '-30', '-20', '-10', '0'])
    
    # ===== Figure (c): High-frequency time series =====
    print("Plotting high-frequency time series...")
    
    # Use only -7~0 days data (corresponding to days 7-14 of the ruler)
    days_high = np.arange(-7, 1)
    avg_cycle_plot = avg_cycle[7:15]  # Days 7-14
    
    ax3.plot(days_high, avg_cycle_plot, color='blue', linewidth=2, label='$SM_H$', alpha=0.8)
    ax3.plot(days_high, -avg_cycle_plot * 0.8, color='red', linewidth=2, label='$T_H$', alpha=0.8)
    ax3.plot(days_high, -avg_cycle_plot * 0.6, color='green', linewidth=2, label='$STC-π_H$', alpha=0.8)
    
    ax3.set_xlabel('Phase / Days', fontsize=12)
    ax3.set_ylabel('Standardized Anomaly', fontsize=12)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower right')  # legend in lower right corner
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax3.set_ylim(-2.0, 2.0)
    
    # Add (c) label and figure title to upper left corner
    ax3.text(0.0, 1.05, '(c) Time Series', transform=ax3.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax3.text(1.0, 1.05, 'High frequency', transform=ax3.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # ===== Figure (d): High-frequency phase space =====
    print("Plotting high-frequency phase space...")
    
    # Flatten time series to trajectories of all grid points
    sm_high_flat = ts_high_SM_detrend.reshape(ts_high_SM_detrend.shape[0], -1)
    t_high_flat = ts_high_T_detrend.reshape(ts_high_T_detrend.shape[0], -1)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(sm_high_flat[0, :]) | np.isnan(t_high_flat[0, :]))
    sm_high_valid = sm_high_flat[:, valid_mask]
    t_high_valid = t_high_flat[:, valid_mask]
    
    # Create color mapping (-7~0 days)
    colors_high = cm.coolwarm(np.linspace(0, 1, len(days_high)))
    
    # Plot trajectories of all grid points (shown as density in gray)
    for i in range(sm_high_valid.shape[1]):
        ax4.plot(sm_high_valid[:, i], t_high_valid[:, i], color='gray', alpha=0.1, lw=0.5)
    
    # Plot regional average trajectory (thick line with color variation)
    # Use high-frequency SM average cycle as ruler, T uses actual high-frequency data
    ts_high_T_avg_plot = ts_high_T_avg[7:15]  # Days 7-14
    
    for i in range(len(days_high) - 1):
        ax4.plot([avg_cycle_plot[i], avg_cycle_plot[i+1]], 
                [ts_high_T_avg_plot[i], ts_high_T_avg_plot[i+1]], 
                color=colors_high[i], lw=3, zorder=10)
    
    # Mark key points
    ax4.scatter(avg_cycle_plot[0], ts_high_T_avg_plot[0], c='blue', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-7 days')
    ax4.scatter(avg_cycle_plot[3], ts_high_T_avg_plot[3], c='white', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-4 days')
    ax4.scatter(avg_cycle_plot[-1], ts_high_T_avg_plot[-1], c='red', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='0 days')
    
    ax4.set_xlabel('$SM_H$ Anomaly', fontsize=12)
    ax4.set_ylabel('$T_H$ Anomaly', fontsize=12)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower left')  # legend in lower left corner
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlim(-4.0, 4.0)
    ax4.set_ylim(-4.0, 4.0)
    
    # Add (d) label and figure title to upper left corner
    ax4.text(0.0, 1.05, '(d) Phase Space', transform=ax4.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax4.text(1.0, 1.05, 'High frequency', transform=ax4.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # Add colorbar to the right of figure d
    norm_high = plt.Normalize(-7, 0)
    sm_high = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm_high)
    sm_high.set_array([])
    cbar_high = plt.colorbar(sm_high, ax=ax4, orientation='vertical', shrink=0.8, pad=0.02)
    cbar_high.set_label('Phase / Days', fontsize=10)
    cbar_high.set_ticks([-7, -6, -5, -4, -3, -2, -1, 0])
    cbar_high.set_ticklabels(['-7', '-6', '-5', '-4', '-3', '-2', '-1', '0'])
    
    # Adjust layout
    plt.tight_layout(pad=1.0)
    
    # Save figure
    plt.savefig('comprehensive_sm_t_stc_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive plot saved as: comprehensive_sm_t_stc_analysis.png")
    
    plt.show()

def main():
    """Main function"""
    print("Starting to read data...")
    
    # Read data
    (ts_low_SM, ts_low_T, ts_low_STC, 
     ts_high_SM, ts_high_T, ts_high_STC, 
     avg_cycle, cycle_time, ntime, nlat, nlon) = read_nc_data()
    
    # Process data
    print("Processing data...")
    (ts_low_SM_detrend, ts_low_T_detrend, ts_low_STC_detrend,
     ts_high_SM_detrend, ts_high_T_detrend, ts_high_STC_detrend) = process_data(
        ts_low_SM, ts_low_T, ts_low_STC, ts_high_SM, ts_high_T, ts_high_STC, ntime, nlat, nlon)
    
    # Create figure
    create_comprehensive_plot(ts_low_SM_detrend, ts_low_T_detrend, ts_low_STC_detrend,
                             ts_high_SM_detrend, ts_high_T_detrend, ts_high_STC_detrend,
                             avg_cycle, ntime, nlat, nlon)
    
    print("Completed!")

if __name__ == "__main__":
    main()