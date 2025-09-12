#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
能量通量综合分析
参考脚本13和19，绘制能量通量时序图和SM-能量通量-T的相空间
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def read_nc_data():
    """读取NC数据"""
    print("读取NC数据...")
    
    # 定义数据文件的相对路径
    data_path = "../../data/climatology/filtered/"
    
    # 读取低频数据
    f_low_slhf = xr.open_dataset(data_path + "mam_slhf4icp_gms_filtered.nc")
    ts_low_SLHF = f_low_slhf['ts_low'].values  # (day, lat, lon)
    
    f_low_sshf = xr.open_dataset(data_path + "mam_sshf4icp_gms_filtered.nc")
    ts_low_SSHF = f_low_sshf['ts_low'].values  # (day, lat, lon)
    
    f_low_rn = xr.open_dataset(data_path + "mam_rn4icp_gms_filtered.nc")
    ts_low_RN = f_low_rn['ts_low'].values  # (day, lat, lon)
    
    f_low_sm = xr.open_dataset(data_path + "mam_sml14icp_gms_filtered.nc")
    ts_low_SM = f_low_sm['ts_low'].values  # (day, lat, lon)
    
    f_low_t = xr.open_dataset(data_path + "mam_t2m4icp_gms_filtered.nc")
    ts_low_T = f_low_t['ts_low'].values  # (day, lat, lon)
    
    # 读取高频数据
    f_high_slhf = xr.open_dataset(data_path + "high_freq_slhf.nc")
    ts_high_SLHF = f_high_slhf['ts_high'].values  # (day, lat, lon)
    
    f_high_sshf = xr.open_dataset(data_path + "high_freq_sshf.nc")
    ts_high_SSHF = f_high_sshf['ts_high'].values  # (day, lat, lon)
    
    f_high_rn = xr.open_dataset(data_path + "high_freq_rn.nc")
    ts_high_RN = f_high_rn['ts_high'].values  # (day, lat, lon)
    
    f_high_sm = xr.open_dataset(data_path + "high_freq_sm.nc")
    ts_high_SM = f_high_sm['ts_high'].values  # (day, lat, lon)
    
    f_high_t = xr.open_dataset(data_path + "high_freq_t2m.nc")
    ts_high_T = f_high_t['ts_high'].values  # (day, lat, lon)
    
    # 读取高频SM平均周期作为标尺
    f_avg_cycle = xr.open_dataset(data_path + "high_freq_sm_avg_cycle.nc")
    avg_cycle = f_avg_cycle['avg_cycle'].values
    cycle_time = f_avg_cycle['cycle_time'].values
    n_cycles = f_avg_cycle['n_cycles'].values
    
    # 获取维度信息
    ntime, nlat, nlon = ts_low_SLHF.shape
    print(f"数据维度: {ntime} 天, {nlat} 纬度, {nlon} 经度")
    print(f"标尺周期长度: {len(avg_cycle)} 天")
    print(f"找到的典型周期数量: {n_cycles}")
    
    return (ts_low_SLHF, ts_low_SSHF, ts_low_RN, ts_low_SM, ts_low_T,
            ts_high_SLHF, ts_high_SSHF, ts_high_RN, ts_high_SM, ts_high_T,
            avg_cycle, cycle_time, ntime, nlat, nlon)

def process_data(ts_low_SLHF, ts_low_SSHF, ts_low_RN, ts_low_SM, ts_low_T,
                ts_high_SLHF, ts_high_SSHF, ts_high_RN, ts_high_SM, ts_high_T,
                ntime, nlat, nlon):
    """处理数据：去趋势和标准化"""
    print("处理数据：去趋势和标准化...")
    
    # 初始化去趋势后的数组
    ts_low_SLHF_detrend = np.copy(ts_low_SLHF)
    ts_low_SSHF_detrend = np.copy(ts_low_SSHF)
    ts_low_RN_detrend = np.copy(ts_low_RN)
    ts_low_SM_detrend = np.copy(ts_low_SM)
    ts_low_T_detrend = np.copy(ts_low_T)
    
    ts_high_SLHF_detrend = np.copy(ts_high_SLHF)
    ts_high_SSHF_detrend = np.copy(ts_high_SSHF)
    ts_high_RN_detrend = np.copy(ts_high_RN)
    ts_high_SM_detrend = np.copy(ts_high_SM)
    ts_high_T_detrend = np.copy(ts_high_T)
    
    # 对每个格点进行去趋势和标准化
    for i in range(nlat):
        for j in range(nlon):
            # 低频潜热通量去趋势
            if not np.all(np.isnan(ts_low_SLHF[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_low_SLHF[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_low_SLHF_detrend[:, i, j] = ts_low_SLHF[:, i, j] - trend
                
                mean_val = np.nanmean(ts_low_SLHF_detrend[:, i, j])
                std_val = np.nanstd(ts_low_SLHF_detrend[:, i, j])
                if std_val != 0:
                    ts_low_SLHF_detrend[:, i, j] = (ts_low_SLHF_detrend[:, i, j] - mean_val) / std_val
            
            # 低频感热通量去趋势
            if not np.all(np.isnan(ts_low_SSHF[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_low_SSHF[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_low_SSHF_detrend[:, i, j] = ts_low_SSHF[:, i, j] - trend
                
                mean_val = np.nanmean(ts_low_SSHF_detrend[:, i, j])
                std_val = np.nanstd(ts_low_SSHF_detrend[:, i, j])
                if std_val != 0:
                    ts_low_SSHF_detrend[:, i, j] = (ts_low_SSHF_detrend[:, i, j] - mean_val) / std_val
            
            # 低频净辐射去趋势
            if not np.all(np.isnan(ts_low_RN[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_low_RN[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_low_RN_detrend[:, i, j] = ts_low_RN[:, i, j] - trend
                
                mean_val = np.nanmean(ts_low_RN_detrend[:, i, j])
                std_val = np.nanstd(ts_low_RN_detrend[:, i, j])
                if std_val != 0:
                    ts_low_RN_detrend[:, i, j] = (ts_low_RN_detrend[:, i, j] - mean_val) / std_val
            
            # 低频SM去趋势
            if not np.all(np.isnan(ts_low_SM[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_low_SM[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_low_SM_detrend[:, i, j] = ts_low_SM[:, i, j] - trend
                
                mean_val = np.nanmean(ts_low_SM_detrend[:, i, j])
                std_val = np.nanstd(ts_low_SM_detrend[:, i, j])
                if std_val != 0:
                    ts_low_SM_detrend[:, i, j] = (ts_low_SM_detrend[:, i, j] - mean_val) / std_val
            
            # 低频T去趋势
            if not np.all(np.isnan(ts_low_T[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_low_T[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_low_T_detrend[:, i, j] = ts_low_T[:, i, j] - trend
                
                mean_val = np.nanmean(ts_low_T_detrend[:, i, j])
                std_val = np.nanstd(ts_low_T_detrend[:, i, j])
                if std_val != 0:
                    ts_low_T_detrend[:, i, j] = (ts_low_T_detrend[:, i, j] - mean_val) / std_val
            
            # 高频潜热通量去趋势
            if not np.all(np.isnan(ts_high_SLHF[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_high_SLHF[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_high_SLHF_detrend[:, i, j] = ts_high_SLHF[:, i, j] - trend
                
                mean_val = np.nanmean(ts_high_SLHF_detrend[:, i, j])
                std_val = np.nanstd(ts_high_SLHF_detrend[:, i, j])
                if std_val != 0:
                    ts_high_SLHF_detrend[:, i, j] = (ts_high_SLHF_detrend[:, i, j] - mean_val) / std_val
            
            # 高频感热通量去趋势
            if not np.all(np.isnan(ts_high_SSHF[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_high_SSHF[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_high_SSHF_detrend[:, i, j] = ts_high_SSHF[:, i, j] - trend
                
                mean_val = np.nanmean(ts_high_SSHF_detrend[:, i, j])
                std_val = np.nanstd(ts_high_SSHF_detrend[:, i, j])
                if std_val != 0:
                    ts_high_SSHF_detrend[:, i, j] = (ts_high_SSHF_detrend[:, i, j] - mean_val) / std_val
            
            # 高频净辐射去趋势
            if not np.all(np.isnan(ts_high_RN[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_high_RN[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_high_RN_detrend[:, i, j] = ts_high_RN[:, i, j] - trend
                
                mean_val = np.nanmean(ts_high_RN_detrend[:, i, j])
                std_val = np.nanstd(ts_high_RN_detrend[:, i, j])
                if std_val != 0:
                    ts_high_RN_detrend[:, i, j] = (ts_high_RN_detrend[:, i, j] - mean_val) / std_val
            
            # 高频SM去趋势
            if not np.all(np.isnan(ts_high_SM[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_high_SM[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_high_SM_detrend[:, i, j] = ts_high_SM[:, i, j] - trend
                
                mean_val = np.nanmean(ts_high_SM_detrend[:, i, j])
                std_val = np.nanstd(ts_high_SM_detrend[:, i, j])
                if std_val != 0:
                    ts_high_SM_detrend[:, i, j] = (ts_high_SM_detrend[:, i, j] - mean_val) / std_val
            
            # 高频T去趋势
            if not np.all(np.isnan(ts_high_T[:, i, j])):
                x = np.arange(ntime)
                coeffs = np.polyfit(x, ts_high_T[:, i, j], 1)
                trend = coeffs[0] * x + coeffs[1]
                ts_high_T_detrend[:, i, j] = ts_high_T[:, i, j] - trend
                
                mean_val = np.nanmean(ts_high_T_detrend[:, i, j])
                std_val = np.nanstd(ts_high_T_detrend[:, i, j])
                if std_val != 0:
                    ts_high_T_detrend[:, i, j] = (ts_high_T_detrend[:, i, j] - mean_val) / std_val
    
    return (ts_low_SLHF_detrend, ts_low_SSHF_detrend, ts_low_RN_detrend, ts_low_SM_detrend, ts_low_T_detrend,
            ts_high_SLHF_detrend, ts_high_SSHF_detrend, ts_high_RN_detrend, ts_high_SM_detrend, ts_high_T_detrend)

def create_comprehensive_plot(ts_low_SLHF_detrend, ts_low_SSHF_detrend, ts_low_RN_detrend, ts_low_SM_detrend, ts_low_T_detrend,
                             ts_high_SLHF_detrend, ts_high_SSHF_detrend, ts_high_RN_detrend, ts_high_SM_detrend, ts_high_T_detrend,
                             avg_cycle, ntime, nlat, nlon):
    """创建综合图表"""
    print("创建综合图表...")
    
    # 计算区域平均时间序列
    print("计算区域平均时间序列...")
    
    # 低频区域平均
    ts_low_SLHF_avg = np.nanmean(ts_low_SLHF_detrend, axis=(1, 2))
    ts_low_SSHF_avg = np.nanmean(ts_low_SSHF_detrend, axis=(1, 2))
    ts_low_RN_avg = np.nanmean(ts_low_RN_detrend, axis=(1, 2))
    ts_low_SM_avg = np.nanmean(ts_low_SM_detrend, axis=(1, 2))
    ts_low_T_avg = np.nanmean(ts_low_T_detrend, axis=(1, 2))
    
    # 高频区域平均
    ts_high_SLHF_avg = np.nanmean(ts_high_SLHF_detrend, axis=(1, 2))
    ts_high_SSHF_avg = np.nanmean(ts_high_SSHF_detrend, axis=(1, 2))
    ts_high_RN_avg = np.nanmean(ts_high_RN_detrend, axis=(1, 2))
    ts_high_SM_avg = np.nanmean(ts_high_SM_detrend, axis=(1, 2))
    ts_high_T_avg = np.nanmean(ts_high_T_detrend, axis=(1, 2))
    
    # 标准化处理
    ts_low_SLHF_avg = (ts_low_SLHF_avg - np.nanmean(ts_low_SLHF_avg)) / np.nanstd(ts_low_SLHF_avg)
    ts_low_SSHF_avg = (ts_low_SSHF_avg - np.nanmean(ts_low_SSHF_avg)) / np.nanstd(ts_low_SSHF_avg)
    ts_low_RN_avg = (ts_low_RN_avg - np.nanmean(ts_low_RN_avg)) / np.nanstd(ts_low_RN_avg)
    ts_low_SM_avg = (ts_low_SM_avg - np.nanmean(ts_low_SM_avg)) / np.nanstd(ts_low_SM_avg)
    ts_low_T_avg = (ts_low_T_avg - np.nanmean(ts_low_T_avg)) / np.nanstd(ts_low_T_avg)
    
    ts_high_SLHF_avg = (ts_high_SLHF_avg - np.nanmean(ts_high_SLHF_avg)) / np.nanstd(ts_high_SLHF_avg)
    ts_high_SSHF_avg = (ts_high_SSHF_avg - np.nanmean(ts_high_SSHF_avg)) / np.nanstd(ts_high_SSHF_avg)
    ts_high_RN_avg = (ts_high_RN_avg - np.nanmean(ts_high_RN_avg)) / np.nanstd(ts_high_RN_avg)
    ts_high_SM_avg = (ts_high_SM_avg - np.nanmean(ts_high_SM_avg)) / np.nanstd(ts_high_SM_avg)
    ts_high_T_avg = (ts_high_T_avg - np.nanmean(ts_high_T_avg)) / np.nanstd(ts_high_T_avg)
    
    # 调整T的符号以确保正确的物理意义
    ts_low_T_avg = -ts_low_T_avg
    ts_high_T_avg = -ts_high_T_avg
    
    # 创建图形 - 1x3布局（上方一个长图，下方三个相空间图）
    fig = plt.figure(figsize=(18, 12))
    
    # 创建网格布局：上方一个长图，下方三个相空间图
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.0], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])  # 图(a): 高频时序图，跨越整个上方
    ax2 = fig.add_subplot(gs[1, 0])  # 图(b): T-SH相空间
    ax3 = fig.add_subplot(gs[1, 1])  # 图(c): SH-λE相空间
    ax4 = fig.add_subplot(gs[1, 2])  # 图(d): λE-SM相空间
    
    # ===== 图(a): 高频时序图 =====
    print("绘制高频时序图...")
    
    # 只使用-7~0天的数据（对应标尺的第7-14天）
    days_high = np.arange(-7, 1)
    avg_cycle_plot = avg_cycle[7:15]  # 第7-14天
    
    # 使用高频能量通量数据
    ts_high_SLHF_plot = ts_high_SLHF_avg[7:15]  # 第7-14天
    ts_high_SSHF_plot = ts_high_SSHF_avg[7:15]  # 第7-14天
    ts_high_RN_plot = ts_high_RN_avg[7:15]      # 第7-14天
    
    ax1.plot(days_high, ts_high_SLHF_plot, color='blue', linewidth=2, label='$λE_H$', alpha=0.8)
    ax1.plot(days_high, ts_high_SSHF_plot, color='red', linewidth=2, label='$SH_H$', alpha=0.8)
    ax1.plot(days_high, ts_high_RN_plot, color='orange', linewidth=2, label='$R_{nH}$', alpha=0.8)
    
    ax1.set_xlabel('Phase / Days', fontsize=12)
    ax1.set_ylabel('Standardized Anomaly', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')  # legend放在右下角
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylim(-2.0, 2.0)
    
    # 添加(a)标记和图题到左上角
    ax1.text(0.0, 1.05, '(a) Time Series', transform=ax1.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax1.text(1.0, 1.05, 'High frequency', transform=ax1.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # ===== 图(b): T-SH相空间图 =====
    print("绘制T-SH相空间图...")
    
    # 将时间序列展平为所有格点的轨迹
    t_high_flat = ts_high_T_detrend.reshape(ts_high_T_detrend.shape[0], -1)
    sh_high_flat = ts_high_SSHF_detrend.reshape(ts_high_SSHF_detrend.shape[0], -1)
    
    # 移除NaN值
    valid_mask = ~(np.isnan(t_high_flat[0, :]) | np.isnan(sh_high_flat[0, :]))
    t_high_valid = t_high_flat[:, valid_mask]
    sh_high_valid = sh_high_flat[:, valid_mask]
    
    # 创建颜色映射（-7~0天）
    colors_high = cm.coolwarm(np.linspace(0, 1, len(days_high)))
    
    # 绘制所有格点的轨迹（用灰色显示密度）
    for i in range(t_high_valid.shape[1]):
        ax2.plot(sh_high_valid[:, i], t_high_valid[:, i], color='gray', alpha=0.1, lw=0.5)
    
    # 绘制区域平均轨迹（粗线，带颜色变化）
    ts_high_T_plot = ts_high_T_avg[7:15]  # 第7-14天
    for i in range(len(days_high) - 1):
        ax2.plot([ts_high_SSHF_plot[i], ts_high_SSHF_plot[i+1]], 
                [ts_high_T_plot[i], ts_high_T_plot[i+1]], 
                color=colors_high[i], lw=3, zorder=10)
    
    # 标记关键点
    ax2.scatter(ts_high_SSHF_plot[0], ts_high_T_plot[0], c='blue', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-7 days')
    ax2.scatter(ts_high_SSHF_plot[3], ts_high_T_plot[3], c='white', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-4 days')
    ax2.scatter(ts_high_SSHF_plot[-1], ts_high_T_plot[-1], c='red', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='0 days')
    
    ax2.set_xlabel('$SH_H$ Anomaly', fontsize=12)
    ax2.set_ylabel('$T_H$ Anomaly', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left')  # legend放在左下角
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlim(-4.0, 4.0)
    ax2.set_ylim(-4.0, 4.0)
    
    # 添加(b)标记和图题到左上角
    ax2.text(0.0, 1.05, '(b) Phase Space', transform=ax2.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax2.text(1.0, 1.05, 'High frequency', transform=ax2.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # ===== 图(c): SH-λE相空间图 =====
    print("绘制SH-λE相空间图...")
    
    # 将时间序列展平为所有格点的轨迹
    sh_high_flat = sh_high_flat
    slhf_high_flat = ts_high_SLHF_detrend.reshape(ts_high_SLHF_detrend.shape[0], -1)
    
    # 移除NaN值
    valid_mask = ~(np.isnan(sh_high_flat[0, :]) | np.isnan(slhf_high_flat[0, :]))
    sh_high_valid = sh_high_flat[:, valid_mask]
    slhf_high_valid = slhf_high_flat[:, valid_mask]
    
    # 绘制所有格点的轨迹（用灰色显示密度）
    for i in range(sh_high_valid.shape[1]):
        ax3.plot(slhf_high_valid[:, i], sh_high_valid[:, i], color='gray', alpha=0.1, lw=0.5)
    
    # 绘制区域平均轨迹（粗线，带颜色变化）
    for i in range(len(days_high) - 1):
        ax3.plot([ts_high_SLHF_plot[i], ts_high_SLHF_plot[i+1]], 
                [ts_high_SSHF_plot[i], ts_high_SSHF_plot[i+1]], 
                color=colors_high[i], lw=3, zorder=10)
    
    # 标记关键点
    ax3.scatter(ts_high_SLHF_plot[0], ts_high_SSHF_plot[0], c='blue', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-7 days')
    ax3.scatter(ts_high_SLHF_plot[3], ts_high_SSHF_plot[3], c='white', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-4 days')
    ax3.scatter(ts_high_SLHF_plot[-1], ts_high_SSHF_plot[-1], c='red', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='0 days')
    
    ax3.set_xlabel('$λE_H$ Anomaly', fontsize=12)
    ax3.set_ylabel('$SH_H$ Anomaly', fontsize=12)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left')  # legend放在左下角
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlim(-4.0, 4.0)
    ax3.set_ylim(-4.0, 4.0)
    
    # 添加(c)标记和图题到左上角
    ax3.text(0.0, 1.05, '(c) Phase Space', transform=ax3.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax3.text(1.0, 1.05, 'High frequency', transform=ax3.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # ===== 图(d): λE-SM相空间图 =====
    print("绘制λE-SM相空间图...")
    
    # 将时间序列展平为所有格点的轨迹
    slhf_high_flat = slhf_high_flat
    sm_high_flat = ts_high_SM_detrend.reshape(ts_high_SM_detrend.shape[0], -1)
    
    # 移除NaN值
    valid_mask = ~(np.isnan(slhf_high_flat[0, :]) | np.isnan(sm_high_flat[0, :]))
    slhf_high_valid = slhf_high_flat[:, valid_mask]
    sm_high_valid = sm_high_flat[:, valid_mask]
    
    # 绘制所有格点的轨迹（用灰色显示密度）
    for i in range(slhf_high_valid.shape[1]):
        ax4.plot(sm_high_valid[:, i], slhf_high_valid[:, i], color='gray', alpha=0.1, lw=0.5)
    
    # 绘制区域平均轨迹（粗线，带颜色变化）
    ts_high_SM_plot = ts_high_SM_avg[7:15]  # 第7-14天
    for i in range(len(days_high) - 1):
        ax4.plot([ts_high_SM_plot[i], ts_high_SM_plot[i+1]], 
                [ts_high_SLHF_plot[i], ts_high_SLHF_plot[i+1]], 
                color=colors_high[i], lw=3, zorder=10)
    
    # 标记关键点
    ax4.scatter(ts_high_SM_plot[0], ts_high_SLHF_plot[0], c='blue', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-7 days')
    ax4.scatter(ts_high_SM_plot[3], ts_high_SLHF_plot[3], c='white', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='-4 days')
    ax4.scatter(ts_high_SM_plot[-1], ts_high_SLHF_plot[-1], c='red', s=100, 
               edgecolors='black', linewidth=2, zorder=20, label='0 days')
    
    ax4.set_xlabel('$SM_H$ Anomaly', fontsize=12)
    ax4.set_ylabel('$λE_H$ Anomaly', fontsize=12)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower left')  # legend放在左下角
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlim(-4.0, 4.0)
    ax4.set_ylim(-4.0, 4.0)
    
    # 添加(d)标记和图题到左上角
    ax4.text(0.0, 1.05, '(d) Phase Space', transform=ax4.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    ax4.text(1.0, 1.05, 'High frequency', transform=ax4.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # 添加水平颜色条到图(b)(c)(d)以下
    norm = plt.Normalize(-7, 0)
    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array([])
    
    # 创建颜色条，位置在底部
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Phase / Days', fontsize=10)
    cbar.set_ticks([-7, -6, -5, -4, -3, -2, -1, 0])
    cbar.set_ticklabels(['-7', '-6', '-5', '-4', '-3', '-2', '-1', '0'])
    
    # 调整布局
    plt.subplots_adjust(bottom=0.15)  # 为颜色条留出空间
    
    # 保存图形
    plt.savefig('high_freq_energy_flux_analysis.png', dpi=300, bbox_inches='tight')
    print("高频能量通量分析图已保存为: high_freq_energy_flux_analysis.png")
    
    plt.show()

def main():
    """主函数"""
    print("开始读取数据...")
    
    # 读取数据
    (ts_low_SLHF, ts_low_SSHF, ts_low_RN, ts_low_SM, ts_low_T,
     ts_high_SLHF, ts_high_SSHF, ts_high_RN, ts_high_SM, ts_high_T,
     avg_cycle, cycle_time, ntime, nlat, nlon) = read_nc_data()
    
    # 处理数据
    print("处理数据...")
    (ts_low_SLHF_detrend, ts_low_SSHF_detrend, ts_low_RN_detrend, ts_low_SM_detrend, ts_low_T_detrend,
     ts_high_SLHF_detrend, ts_high_SSHF_detrend, ts_high_RN_detrend, ts_high_SM_detrend, ts_high_T_detrend) = process_data(
        ts_low_SLHF, ts_low_SSHF, ts_low_RN, ts_low_SM, ts_low_T,
        ts_high_SLHF, ts_high_SSHF, ts_high_RN, ts_high_SM, ts_high_T, ntime, nlat, nlon)
    
    # 创建图形
    create_comprehensive_plot(ts_low_SLHF_detrend, ts_low_SSHF_detrend, ts_low_RN_detrend, ts_low_SM_detrend, ts_low_T_detrend,
                             ts_high_SLHF_detrend, ts_high_SSHF_detrend, ts_high_RN_detrend, ts_high_SM_detrend, ts_high_T_detrend,
                             avg_cycle, ntime, nlat, nlon)
    
    print("完成！")

if __name__ == "__main__":
    main()
