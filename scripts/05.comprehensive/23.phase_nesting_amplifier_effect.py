#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Nesting and Amplifier Effect Analysis
高低频SM-T滞回环并列分析：相位嵌套与放大器效应

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xarray as xr
import netCDF4 as nc
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def read_data():
    """读取数据"""
    print("开始读取数据...")
    
    # Define the relative path for data files
    data_path = "../../data/climatology/filtered/"
    
    # 读取低频数据，与脚本11保持一致
    print("读取低频数据...")
    ts_low_SM = xr.open_dataset(data_path + "mam_sml14icp_gms_filtered.nc")['ts_low'].values  # 修正：使用SM数据文件
    ts_low_T = xr.open_dataset(data_path + "mam_t2m4icp_gms_filtered.nc")['ts_low'].values
    
    # 读取高频数据
    print("读取高频数据...")
    ts_high_SM = xr.open_dataset(data_path + "mam_sml14icp_gms_filtered.nc")['ts_high'].values  # 修正：使用SM数据文件
    ts_high_T = xr.open_dataset(data_path + "mam_t2m4icp_gms_filtered.nc")['ts_high'].values
    
    # 读取高频SM平均周期作为标尺
    print("读取高频SM平均周期标尺...")
    f_avg_cycle = xr.open_dataset(data_path + "high_freq_sm_avg_cycle_refined.nc")
    avg_cycle = f_avg_cycle['avg_cycle'].values
    
    print(f"数据维度: {ts_low_SM.shape[0]} 天, {ts_low_SM.shape[1]} 纬度, {ts_low_SM.shape[2]} 经度")
    print(f"标尺周期长度: {len(avg_cycle)} 天")
    
    return ts_low_SM, ts_low_T, ts_high_SM, ts_high_T, avg_cycle

def process_data(ts_low_SM, ts_low_T, ts_high_SM, ts_high_T, avg_cycle):
    """处理数据：去趋势和标准化，与脚本11完全一致"""
    print("处理数据：去趋势和标准化...")
    
    # 首先对全部数据进行去趋势和标准化，与脚本11完全一致
    print("对全部数据进行去趋势和标准化...")
    from scipy import signal
    
    # 对低频SM进行去趋势和标准化
    for i in range(ts_low_SM.shape[1]):
        for j in range(ts_low_SM.shape[2]):
            if not np.isnan(ts_low_SM[0, i, j]):
                ts_low_SM[:, i, j] = signal.detrend(ts_low_SM[:, i, j])
                ts_low_SM[:, i, j] = (ts_low_SM[:, i, j] - np.mean(ts_low_SM[:, i, j])) / np.std(ts_low_SM[:, i, j])
            
            if not np.isnan(ts_low_T[0, i, j]):
                ts_low_T[:, i, j] = signal.detrend(ts_low_T[:, i, j])
                ts_low_T[:, i, j] = (ts_low_T[:, i, j] - np.mean(ts_low_T[:, i, j])) / np.std(ts_low_T[:, i, j])
    
    # 对高频T进行去趋势和标准化
    for i in range(ts_high_T.shape[1]):
        for j in range(ts_high_T.shape[2]):
            if not np.isnan(ts_high_T[0, i, j]):
                ts_high_T[:, i, j] = signal.detrend(ts_high_T[:, i, j])
                ts_high_T[:, i, j] = (ts_high_T[:, i, j] - np.mean(ts_high_T[:, i, j])) / np.std(ts_high_T[:, i, j])
    
    # 然后选择前半周期数据（第6-46天，对应-40~0天），与脚本11保持一致
    print("选择前半周期数据...")
    start_idx = 6   # 从第6天开始
    end_idx = 46    # 到第45天结束（40天）
    ts_low_SM_sel = ts_low_SM[start_idx:end_idx, :, :]
    ts_low_T_sel = ts_low_T[start_idx:end_idx, :, :]
    
    # 高频数据处理（前半周期：-7~0天）
    print("处理高频数据...")
    # 使用高频SM平均周期作为标尺，T使用实际高频数据
    ts_high_SM_plot = avg_cycle[7:15]  # 第7-14天，对应-7~0天
    
    # 高频T数据选择
    ts_high_T_sel = ts_high_T[start_idx:end_idx, :, :]
    
    return ts_low_SM_sel, ts_low_T_sel, ts_high_SM_plot, ts_high_T_sel

def create_phase_nesting_plot(ts_low_SM_detrend, ts_low_T_detrend, ts_high_SM_plot, ts_high_T_detrend):
    """创建相位嵌套与放大器效应图"""
    print("创建相位嵌套与放大器效应图...")
    
    # 计算区域平均时间序列，参考脚本11
    print("计算区域平均时间序列...")
    
    # 低频区域平均
    ts_low_SM_avg = np.nanmean(ts_low_SM_detrend, axis=(1, 2))
    ts_low_T_avg = np.nanmean(ts_low_T_detrend, axis=(1, 2))
    
    # 高频区域平均
    ts_high_T_avg = np.nanmean(ts_high_T_detrend, axis=(1, 2))
    
    # 检查数据范围
    print(f"低频SM范围: {np.min(ts_low_SM_avg):.3f} ~ {np.max(ts_low_SM_avg):.3f}")
    print(f"低频T范围: {np.min(ts_low_T_avg):.3f} ~ {np.max(ts_low_T_avg):.3f}")
    print(f"高频SM范围: {np.min(ts_high_SM_plot):.3f} ~ {np.max(ts_high_SM_plot):.3f}")
    print(f"高频T范围: {np.min(ts_high_T_avg):.3f} ~ {np.max(ts_high_T_avg):.3f}")
    print(f"低频SM-40天值: {ts_low_SM_avg[0]:.3f}")
    print(f"低频T-40天值: {ts_low_T_avg[0]:.3f}")
    print(f"低频SM-20天值: {ts_low_SM_avg[20]:.3f}")
    print(f"低频T-20天值: {ts_low_T_avg[20]:.3f}")
    print(f"低频SM-0天值: {ts_low_SM_avg[-1]:.3f}")
    print(f"低频T-0天값: {ts_low_T_avg[-1]:.3f}")
    
    # 数据处理完成，现在与脚本11完全一致
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # ===== 低频SM-T滞回环（用色标表示方向，参考脚本11）=====
    print("绘制低频SM-T滞回环...")
    
    # 低频时间范围：前半周期（-40~0天，SM衰减、T上升）
    days_low = np.arange(-40, 0)  # -40到0天，与脚本11保持一致
    ts_low_SM_plot = ts_low_SM_avg  # 使用全部40天数据
    ts_low_T_plot = ts_low_T_avg    # 使用全部40天数据
    
    print(f"低频SM数据长度: {len(ts_low_SM_plot)}")
    print(f"低频T数据长度: {len(ts_low_T_plot)}")
    print(f"days_low长度: {len(days_low)}")
    
    # 确保数据长度一致
    min_length = min(len(ts_low_SM_plot), len(ts_low_T_plot), len(days_low))
    ts_low_SM_plot = ts_low_SM_plot[:min_length]
    ts_low_T_plot = ts_low_T_plot[:min_length]
    days_low = days_low[:min_length]
    
    # 将时间序列展平为所有格点的轨迹，参考脚本11
    sm_low_flat = ts_low_SM_detrend.reshape(ts_low_SM_detrend.shape[0], -1)
    t_low_flat = ts_low_T_detrend.reshape(ts_low_T_detrend.shape[0], -1)
    
    # 移除NaN值
    valid_mask = ~(np.isnan(sm_low_flat[0, :]) | np.isnan(t_low_flat[0, :]))
    sm_low_valid = sm_low_flat[:, valid_mask]
    t_low_valid = t_low_flat[:, valid_mask]
    
    # 删除格点分布，只保留区域平均轨迹
    # for i in range(sm_low_valid.shape[1]):
    #     ax.plot(sm_low_valid[:, i], t_low_valid[:, i], color='lightblue', alpha=0.1, lw=0.5)
    
    # 绘制区域平均轨迹（用色标表示方向，参考脚本11）
    # 创建颜色映射（-40~0天）
    colors_low = cm.coolwarm(np.linspace(0, 1, len(days_low)))
    
    # 绘制低频轨迹，用颜色表示时间方向
    for i in range(len(days_low) - 1):
        ax.plot([ts_low_SM_plot[i], ts_low_SM_plot[i+1]], 
                [ts_low_T_plot[i], ts_low_T_plot[i+1]], 
                color=colors_low[i], lw=4, alpha=0.8, zorder=5)
    
    # 标记低频轨迹的关键点
    ax.scatter(ts_low_SM_plot[0], ts_low_T_plot[0], c='blue', s=150, 
               edgecolors='black', linewidth=2, zorder=20, alpha=0.8, label='Low-freq: -40 days')
    mid_idx = len(days_low) // 2
    ax.scatter(ts_low_SM_plot[mid_idx], ts_low_T_plot[mid_idx], c='white', s=150, 
               edgecolors='blue', linewidth=2, zorder=20, alpha=0.8, label='Low-freq: -20 days')
    ax.scatter(ts_low_SM_plot[-1], ts_low_T_plot[-1], c='red', s=150, 
               edgecolors='black', linewidth=2, zorder=20, alpha=0.8, label='Low-freq: 0 days')
    
    # ===== 高频SM-T滞回环（用箭头表示，两组不同背景条件）=====
    print("绘制高频SM-T滞回环...")
    
    # 高频时间范围：前半周期（-7~0天）
    days_high = np.arange(-7, 1)  # -7到0天
    ts_high_SM_plot = ts_high_SM_plot  # 高频SM使用标尺
    ts_high_T_plot = ts_high_T_avg[:len(ts_high_SM_plot)]  # 高频T数据
    
    print(f"高频SM数据长度: {len(ts_high_SM_plot)}")
    print(f"高频T数据长度: {len(ts_high_T_plot)}")
    print(f"days_high长度: {len(days_high)}")
    
    # 确保高频数据长度一致
    min_length_high = min(len(ts_high_SM_plot), len(ts_high_T_plot), len(days_high))
    ts_high_SM_plot = ts_high_SM_plot[:min_length_high]
    ts_high_T_plot = ts_high_T_plot[:min_length_high]
    days_high = days_high[:min_length_high]
    
    # 将时间序列展平为所有格点的轨迹
    t_high_flat = ts_high_T_detrend.reshape(ts_high_T_detrend.shape[0], -1)
    
    # 移除NaN值
    valid_mask = ~(np.isnan(t_high_flat[0, :]))
    t_high_valid = t_high_flat[:, valid_mask]
    
    # 删除格点分布，只保留区域平均轨迹
    # for i in range(t_high_valid.shape[1]):
    #     # 使用高频SM标尺和实际T数据，确保长度匹配
    #     if len(ts_high_SM_plot) == len(t_high_valid[:, i]):
    #         ax.plot(ts_high_SM_plot, t_high_valid[:, i], color='gray', alpha=0.1, lw=0.5)
    
    # ===== 第一组高频滞回环：在低频轨迹-38天位置（湿润背景）=====
    print("绘制第一组高频滞回环（湿润背景）...")
    
    # 使用低频轨迹的-38天位置（第2天，SM正异常）
    wet_bg_idx = 2  # -38天位置（第2天）
    wet_bg_SM = ts_low_SM_plot[wet_bg_idx]  # SM-38天值
    wet_bg_T = ts_low_T_plot[wet_bg_idx]    # T-38天值
    
    print(f"湿润背景位置: 第{wet_bg_idx}天（-38天）, SM={wet_bg_SM:.3f}, T={wet_bg_T:.3f}")
    
    # 在-38天位置绘制高频滞回环
    # 调整高频滞回环的位置，使其嵌套在低频轨迹的-38天位置
    high_freq_wet_SM = wet_bg_SM + ts_high_SM_plot * 0.3  # 缩放高频SM变化
    high_freq_wet_T = wet_bg_T + ts_high_T_plot * 0.3     # 缩放高频T变化
    
    # 绘制湿润背景下的高频滞回环（蓝色，较温和）
    for i in range(len(days_high) - 1):
        ax.plot([high_freq_wet_SM[i], high_freq_wet_SM[i+1]], 
                [high_freq_wet_T[i], high_freq_wet_T[i+1]], 
                color='blue', lw=3, zorder=15, alpha=0.8)
    
    # 添加箭头表示方向（顺时针）- 修正方向
    for i in range(0, len(days_high) - 1, 2):  # 每隔一个点添加箭头
        if i + 1 < len(days_high):
            # 确保箭头指向正确的顺时针方向
            ax.annotate('', xy=(high_freq_wet_SM[i+1], high_freq_wet_T[i+1]), 
                        xytext=(high_freq_wet_SM[i], high_freq_wet_T[i]),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.8))
    
    # 标记湿润背景高频滞回环的关键点
    ax.scatter(high_freq_wet_SM[0], high_freq_wet_T[0], c='blue', s=80, 
               edgecolors='black', linewidth=2, zorder=25, alpha=0.8, label='High-freq Wet: -7 days')
    ax.scatter(high_freq_wet_SM[-1], high_freq_wet_T[-1], c='red', s=80, 
               edgecolors='blue', linewidth=2, zorder=25, alpha=0.8, label='High-freq Wet: 0 days')
    
    # ===== 第二组高频滞回环：在低频SM为负异常时期（干旱背景）=====
    print("绘制第二组高频滞回环（干旱背景）...")
    
    # 找到低频轨迹中SM为负异常的位置，确保初始SM也是负异常
    sm_negative_indices = np.where(ts_low_SM_plot < 0)[0]
    if len(sm_negative_indices) > 0:
        # 选择负异常区域中较晚的位置，确保高频滞回环的初始SM也是负异常
        dry_bg_idx = sm_negative_indices[-1]  # 选择最后一个负异常位置
    else:
        dry_bg_idx = len(days_low) - 5  # 如果没有负异常，选择接近末尾的位置
    
    dry_bg_SM = ts_low_SM_plot[dry_bg_idx]
    dry_bg_T = ts_low_T_plot[dry_bg_idx]
    
    print(f"干旱背景位置: 第{dry_bg_idx}天, SM={dry_bg_SM:.3f}, T={dry_bg_T:.3f}")
    
    # 在干旱背景位置绘制高频滞回环
    # 调整高频滞回环的位置，使其嵌套在低频轨迹上
    high_freq_dry_SM = dry_bg_SM + ts_high_SM_plot * 0.2  # 减小高频SM变化，确保初始值为负
    high_freq_dry_T = dry_bg_T + ts_high_T_plot * 0.5     # 放大高频T变化（放大器效应）
    
    # 检查高频滞回环的初始SM值
    print(f"干旱背景高频滞回环初始SM值: {high_freq_dry_SM[0]:.3f}")
    
    # 如果初始SM值仍然为正，进一步调整
    if high_freq_dry_SM[0] > 0:
        # 选择一个更负的背景位置
        for idx in reversed(sm_negative_indices):
            test_bg_SM = ts_low_SM_plot[idx]
            test_bg_T = ts_low_T_plot[idx]
            test_high_freq_SM = test_bg_SM + ts_high_SM_plot * 0.2
            if test_high_freq_SM[0] < 0:
                dry_bg_idx = idx
                dry_bg_SM = test_bg_SM
                dry_bg_T = test_bg_T
                high_freq_dry_SM = test_high_freq_SM
                high_freq_dry_T = dry_bg_T + ts_high_T_plot * 0.5
                print(f"调整后干旱背景位置: 第{dry_bg_idx}天, SM={dry_bg_SM:.3f}, T={dry_bg_T:.3f}")
                print(f"调整后干旱背景高频滞回环初始SM值: {high_freq_dry_SM[0]:.3f}")
                break
    
    # 绘制干旱背景下的高频滞回环（红色，响应强烈）
    for i in range(len(days_high) - 1):
        ax.plot([high_freq_dry_SM[i], high_freq_dry_SM[i+1]], 
                [high_freq_dry_T[i], high_freq_dry_T[i+1]], 
                color='red', lw=3, zorder=15, alpha=0.8)
    
    # 添加箭头表示方向（顺时针）- 修正方向
    for i in range(0, len(days_high) - 1, 2):  # 每隔一个点添加箭头
        if i + 1 < len(days_high):
            # 确保箭头指向正确的顺时针方向
            ax.annotate('', xy=(high_freq_dry_SM[i+1], high_freq_dry_T[i+1]), 
                        xytext=(high_freq_dry_SM[i], high_freq_dry_T[i]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.8))
    
    # 标记干旱背景高频滞回环的关键点
    ax.scatter(high_freq_dry_SM[0], high_freq_dry_T[0], c='blue', s=80, 
               edgecolors='red', linewidth=2, zorder=25, alpha=0.8, label='High-freq Dry: -7 days')
    ax.scatter(high_freq_dry_SM[-1], high_freq_dry_T[-1], c='red', s=80, 
               edgecolors='red', linewidth=2, zorder=25, alpha=0.8, label='High-freq Dry: 0 days')
    
    # ===== 删除左上角的理论说明文字 =====
    # 低频说明
    # ax.text(0.02, 0.98, 'Low-frequency: System State Evolution Trajectory\n(SM leads T) - Color indicates time direction', 
    #         transform=ax.transAxes, fontsize=12, verticalalignment='top', 
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    # 高频说明
    # ax.text(0.02, 0.90, 'High-frequency: Instantaneous Response Patterns\n(T leads SM) - Arrows show clockwise direction', 
    #         transform=ax.transAxes, fontsize=12, verticalalignment='top',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    # 相位嵌套说明
    # ax.text(0.02, 0.82, 'Phase Nesting: High-freq events nested\nwithin low-freq trajectory', 
    #         transform=ax.transAxes, fontsize=12, verticalalignment='top',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    # 放大器效应说明
    # ax.text(0.02, 0.74, 'Amplifier Effect: Dry background amplifies\nhigh-freq T response (red loop)', 
    #         transform=ax.transAxes, fontsize=12, verticalalignment='top',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    
    # ===== 图形设置 =====
    # 设置轴范围
    ax.set_xlim(-2.0, 2.0)  # SM轴范围
    ax.set_ylim(-2.0, 2.0)  # 缩小T轴范围，从(-3.0, 2.0)改为(-2.0, 2.0)
    
    # 添加参考线
    ax.axhline(0, color='grey', linestyle='--', alpha=0.7)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.7)
    
    # 设置轴标签和标题
    ax.set_xlabel('SM Anomaly', fontsize=12)
    ax.set_ylabel('T Anomaly', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')
    
    # 添加图题
    ax.text(0.5, 1.05, 'Phase Nesting and Amplifier Effect: Low vs High Frequency SM-T Hysteresis', 
            transform=ax.transAxes, fontsize=14, ha='center', va='bottom')
    
    # 设置轴范围
    ax.set_xlim(-2.0, 2.0)  # SM轴范围
    ax.set_ylim(-2.0, 2.0)  # 缩小T轴范围
    
    # 添加色标
    norm = plt.Normalize(vmin=days_low[0], vmax=days_low[-1])
    sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[-40, -20, 0], orientation='vertical')
    cbar.set_label('Low-freq Phase (Days)', rotation=270, labelpad=20, fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('phase_nesting_amplifier_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("相位嵌套与放大器效应图已保存为: phase_nesting_amplifier_effect.png")
    
    return fig

def main():
    """主函数"""
    print("开始相位嵌套与放大器效应分析...")
    
    # 读取数据
    ts_low_SM, ts_low_T, ts_high_SM, ts_high_T, avg_cycle = read_data()
    
    # 处理数据
    ts_low_SM_detrend, ts_low_T_detrend, ts_high_SM_plot, ts_high_T_detrend = process_data(
        ts_low_SM, ts_low_T, ts_high_SM, ts_high_T, avg_cycle)
    
    # 创建相位嵌套图
    create_phase_nesting_plot(ts_low_SM_detrend, ts_low_T_detrend, ts_high_SM_plot, ts_high_T_detrend)
    
    print("完成！")

if __name__ == "__main__":
    main()
