#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trajectories_vis.csv')

plt.figure(figsize=(10, 8))

# 绘制历史轨迹
hist = df[df['type'] == 'history']
plt.plot(hist['lon'], hist['lat'], 'ko-', label='历史轨迹', linewidth=2, markersize=6)

# 绘制预测轨迹（每个 mode）
colors = ['r', 'b', 'm', 'g', 'c']
for mode in sorted(df[df['type'] == 'prediction']['mode'].unique()):
    pred = df[(df['type'] == 'prediction') & (df['mode'] == mode)]
    color = colors[mode % len(colors)]
    plt.plot(pred['lon'], pred['lat'], color + 'o--', 
             label=f'预测轨迹 Mode {mode}', linewidth=1.5, markersize=4, alpha=0.7)

plt.xlabel('经度 (Lon)', fontsize=12)
plt.ylabel('纬度 (Lat)', fontsize=12)
plt.title('多模态轨迹预测可视化', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('trajectories_vis.png', dpi=150, bbox_inches='tight')
print('[OK] 可视化图片已保存到: trajectories_vis.png')
plt.show()
