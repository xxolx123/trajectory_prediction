#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion 部署版（20260427）：可视化 trajectories_vis.csv
新增 type=road / type=target 的可选渲染（旧版列只多不少，向后兼容）。
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trajectories_vis.csv')

plt.figure(figsize=(10, 8))

hist = df[df['type'] == 'history']
if not hist.empty:
    plt.plot(hist['lon'], hist['lat'], 'ko-', label='历史轨迹',
             linewidth=2, markersize=6)

colors = ['r', 'b', 'm', 'g', 'c']
pred = df[df['type'] == 'prediction']
for mode in sorted(pred['mode'].unique()):
    pp = pred[pred['mode'] == mode]
    color = colors[mode % len(colors)]
    plt.plot(pp['lon'], pp['lat'], color + 'o--',
             label=f'预测轨迹 Mode {mode}', linewidth=1.5,
             markersize=4, alpha=0.7)

road = df[df['type'] == 'road']
for bi in sorted(road['mode'].unique()):
    rb = road[road['mode'] == bi]
    plt.plot(rb['lon'], rb['lat'], 's-', color='#1F77B4',
             label=f'路网 branch {bi}', linewidth=2.5, markersize=5,
             alpha=0.85)

tgt = df[df['type'] == 'target']
if not tgt.empty:
    plt.scatter(tgt['lon'], tgt['lat'], s=220, marker='*',
                color='red', edgecolors='black', linewidths=1.2,
                zorder=9, label='我方固定目标 position')

plt.xlabel('经度 (Lon)', fontsize=12)
plt.ylabel('纬度 (Lat)', fontsize=12)
plt.title('多模态轨迹预测可视化（fusion 部署版 20260427）', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('trajectories_vis.png', dpi=150, bbox_inches='tight')
print('[OK] 可视化图片已保存到: trajectories_vis.png')
plt.show()
