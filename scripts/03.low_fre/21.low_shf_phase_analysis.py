#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Space Analysis for Low-Frequency Heat Flux Processes
低频潜热、感热和净辐射的相空间分析
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import netCDF4 as nc
import os
from scipy import signal

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_nc_data():
    """读取滤波后的低频分量数据"""
    # 定义数据文件的相对路径
    data_path = "../../data/climatology/filtered/"
    
    # 读取滤波后的低频分量
    f_slhf = nc.Dataset(data_path + "mam_slhf4icp_gms_filtered.nc", "r")
    f_sshf = nc.Dataset(data_path + "mam_sshf4icp_gms_filtered.nc", "r")
    f_rn = nc.Dataset(data_path + "mam_rn4icp_gms_filtered.nc", "r")
    f_sm = nc.Dataset(data_path + "mam_sml14icp_gms_filtered.nc", "r")
    f_t2m = nc.Dataset(data_path + "mam_t2m4icp_gms_filtered.nc", "r")
    
    # 读取低频分量（保留空间维度）
    ts_low_slhf = f_slhf.variables['ts_low'][:]  # 潜热 (time, lat, lon)
    ts_low_sshf = f_sshf.variables['ts_low'][:]  # 感热 (time, lat, lon)
    ts_low_rn = f_rn.variables['ts_low'][:]      # 净辐射 (time, lat, lon)
    ts_low_sm = f_sm.variables['ts_low'][:]      # 土壤湿度 (time, lat, lon)
    ts_low_t2m = f_t2m.variables['ts_low'][:]    # 地表温度 (time, lat, lon)
    
    # 获取维度信息
    lat = f_slhf.variables['lat'][:]
    lon = f_slhf.variables['lon'][:]
    
    # 关闭文件
    f_slhf.close()
    f_sshf.close()
    f_rn.close()
    f_sm.close()
    f_t2m.close()
    
    return ts_low_slhf, ts_low_sshf, ts_low_rn, ts_low_sm, ts_low_t2m, lat, lon

def apply_mask_and_process(data, lat, lon):
    """应用掩膜并进行去趋势和标准化处理"""
    # 创建陆地掩膜（简单方法：移除海洋区域）
    # 这里假设NaN值表示海洋区域
    masked_data = data.copy()
    
    # 对每个格点进行去趋势和标准化
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if not np.isnan(data[0, i, j]):
                # 去趋势
                masked_data[:, i, j] = signal.detrend(data[:, i, j])
                # 标准化
                masked_data[:, i, j] = (masked_data[:, i, j] - np.mean(masked_data[:, i, j])) / np.std(masked_data[:, i, j])
            else:
                masked_data[:, i, j] = np.nan
    
    return masked_data

def select_time_period(data, start_idx=6, end_idx=46):
    """选择特定的时间段（-40到0天）"""
    if data.shape[0] >= end_idx:
        return data[start_idx:end_idx, :, :]
    else:
        return data

def create_phase_space_plot(slhf_low, sshf_low, rn_low, sm_low, t2m_low, lat, lon):
    """创建相空间分析图"""
    # 创建时间轴（-40到0天）
    days = np.arange(-40, 0)
    
    print(f"=== 选择的时间段 ===")
    print(f"时间段: {days[0]} 到 {days[-1]} 天")
    print(f"数据形状: {slhf_low.shape}")
    
    # 创建图形 - 2x2布局，图a为长方形在上方，图bcd为正方形在下方
    fig = plt.figure(figsize=(16, 14))  # 增加图形高度
    
    # 创建网格布局 - 3行：图a、图bcd、色标
    gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 1.0, 0.2], width_ratios=[1, 1, 1], 
                          hspace=0.005, wspace=0.2)  # 进一步减小垂直间距，调整色标区域
    
    # 图a：时间序列（跨越整个宽度）
    ax1 = fig.add_subplot(gs[0, :])
    
    # 图bcd：相空间图（正方形）
    ax2 = fig.add_subplot(gs[1, 0])  # SM-潜热
    ax3 = fig.add_subplot(gs[1, 1])  # 潜热-感热
    ax4 = fig.add_subplot(gs[1, 2])  # 感热-T
    
    # 色标区域（跨越整个宽度）
    ax_cbar = fig.add_subplot(gs[2, :])
    
    # ===== 图a: 时间序列图 =====
    # 计算区域平均
    slhf_avg = np.nanmean(slhf_low, axis=(1, 2))
    sshf_avg = np.nanmean(sshf_low, axis=(1, 2))
    rn_avg = np.nanmean(rn_low, axis=(1, 2))
    
    ax1.plot(days, slhf_avg, color='blue', lw=2, label='$λE_L$')
    ax1.plot(days, sshf_avg, color='red', lw=2, label='$SH_L$')
    ax1.plot(days, rn_avg, color='orange', lw=2, label='$R_{nL}$')
    
    ax1.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Phase / Days', fontsize=12)
    ax1.set_ylabel('Standardized Anomaly', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(days[0]-2, days[-1]+2)  # 扩大X轴范围
    ax1.set_ylim(-1.5, 1.5)  # 设置Y轴范围为-1.5到1.5
    
    # 添加(a)标记
    ax1.text(0.0, 1.05, '(a) Time Series', transform=ax1.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='left')
    
    # 添加Low frequency标记到右上角
    ax1.text(1.0, 1.05, 'Low frequency', transform=ax1.transAxes, fontsize=14, 
             verticalalignment='top', horizontalalignment='right')
    
    # 设置x轴刻度
    ax1.set_xticks([-40, -30, -20, -10, 0])
    ax1.set_xticklabels(['-40', '-30', '-20', '-10', '0'])
    
    # ===== 图b: SM-潜热相空间 =====
    create_single_phase_space(ax2, sm_low, slhf_low, days, '$SM_L$', '$λE_L$', '(b) Phase Space')
    
    # ===== 图c: 潜热-感热相空间 =====
    create_single_phase_space(ax3, slhf_low, sshf_low, days, '$λE_L$', '$SH_L$', '(c) Phase Space')
    
    # ===== 图d: 感热-T相空间 =====
    create_single_phase_space(ax4, sshf_low, t2m_low, days, '$SH_L$', '$T_L$', '(d) Phase Space')
    
    # ===== 添加水平色标 =====
    # 隐藏色标区域的坐标轴
    ax_cbar.set_visible(False)
    
    # 创建水平色标
    norm = Normalize(vmin=days[0], vmax=days[-1])
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array([])
    
    # 在色标区域添加水平色标
    cbar = fig.colorbar(sm, ax=ax_cbar, orientation='horizontal', 
                       ticks=[-40, -30, -20, -10, 0], shrink=0.8, pad=0.05)
    cbar.set_label('Phase / Days', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # 调整布局 - 进一步减小pad以让色标更接近图形
    plt.tight_layout(pad=0.1)
    
    # 保存图片
    plt.savefig('low_freq_heat_flux_phase_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("相空间分析图已保存为: low_freq_heat_flux_phase_analysis.png")
    
    return fig

def create_single_phase_space(ax, var1_data, var2_data, days, var1_name, var2_name, label):
    """创建单个相空间图"""
    # 计算区域平均
    var1_avg = np.nanmean(var1_data, axis=(1, 2))
    var2_avg = np.nanmean(var2_data, axis=(1, 2))
    
    # 使用所有格点的数据创建相空间轨迹
    var1_flat = var1_data.reshape(var1_data.shape[0], -1)
    var2_flat = var2_data.reshape(var2_data.shape[0], -1)
    
    # 移除NaN值
    valid_mask = ~(np.isnan(var1_flat[0, :]) | np.isnan(var2_flat[0, :]))
    var1_valid = var1_flat[:, valid_mask]
    var2_valid = var2_flat[:, valid_mask]
    
    # 创建颜色映射
    colors = cm.coolwarm(np.linspace(0, 1, len(days)))
    
    # 绘制所有格点的轨迹（用灰色显示密度）
    for i in range(var1_valid.shape[1]):  # 绘制所有格点轨迹
        ax.plot(var1_valid[:, i], var2_valid[:, i], color='gray', alpha=0.1, lw=0.5)
    
    # 绘制区域平均轨迹（粗线，带颜色变化）
    for i in range(len(days) - 1):
        ax.plot(var1_avg[i:i+2], var2_avg[i:i+2], color=colors[i], lw=3, zorder=10)
    
    # 添加关键点标记
    ax.scatter(var1_avg[0], var2_avg[0], color=colors[0], s=120, zorder=15, 
               ec='black', lw=2, label='-40 days Initial')
    ax.scatter(var1_avg[len(days)//2], var2_avg[len(days)//2], color=colors[len(days)//2], 
               s=120, zorder=15, ec='black', lw=2, label='-20 days Mid')
    ax.scatter(var1_avg[-1], var2_avg[-1], color=colors[-1], s=120, zorder=15, 
               ec='black', lw=2, label='0 days Peak')
    
    # 添加参考线
    ax.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.7)
    
    # 设置轴标签和标题
    ax.set_xlabel(f'{var1_name} Anomaly', fontsize=10)
    ax.set_ylabel(f'{var2_name} Anomaly', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower left', fontsize=8)
    
    # 添加标记
    ax.text(0.0, 1.05, label, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='left')
    
    # 添加Low frequency标记到右上角
    ax.text(1.0, 1.05, 'Low frequency', transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='right')
    
    # 设置轴范围
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    
    # 色标将在主函数中统一添加

def main():
    """主函数"""
    print("开始读取数据...")
    slhf_low, sshf_low, rn_low, sm_low, t2m_low, lat, lon = read_nc_data()
    
    print("应用掩膜和处理数据...")
    slhf_processed = apply_mask_and_process(slhf_low, lat, lon)
    sshf_processed = apply_mask_and_process(sshf_low, lat, lon)
    rn_processed = apply_mask_and_process(rn_low, lat, lon)
    sm_processed = apply_mask_and_process(sm_low, lat, lon)
    t2m_processed = apply_mask_and_process(t2m_low, lat, lon)
    
    print("选择时间段...")
    slhf_selected = select_time_period(slhf_processed)
    sshf_selected = select_time_period(sshf_processed)
    rn_selected = select_time_period(rn_processed)
    sm_selected = select_time_period(sm_processed)
    t2m_selected = select_time_period(t2m_processed)
    
    print("创建相空间图...")
    fig = create_phase_space_plot(slhf_selected, sshf_selected, rn_selected, 
                                 sm_selected, t2m_selected, lat, lon)
    
    print("完成！")

if __name__ == "__main__":
    main()
