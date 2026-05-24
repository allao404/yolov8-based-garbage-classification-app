#!/usr/bin/env python3
"""
对比YOLOv8s和YOLOv8m的评估指标(Precision, Recall, mAP)变化趋势
绘制精度、召回率、平均精度均值模型性能对比曲线图
"""

# 数据处理库
import pandas as pd
# 绘图可视化库
import matplotlib.pyplot as plt
import matplotlib
# 切换无图形界面可用的绘图后端，适配服务器环境
matplotlib.use('Agg')

# 全局字体配置，兼容中文与负号显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 两个模型训练日志文件路径
yolov8s_path = '/nas03/yixuh/garbage-classification/models/garbage_yolov8s/results.csv'
yolov8m_path = '/nas03/yixuh/garbage-classification/models/garbage_yolov8m/results.csv'

# 读取训练日志数据
df_s = pd.read_csv(yolov8s_path)
df_m = pd.read_csv(yolov8m_path)

# 清理列名多余空格，避免索引读取失败
df_s.columns = df_s.columns.str.strip()
df_m.columns = df_m.columns.str.strip()

# 创建2行2列子图画布，设定画布尺寸
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# 设置图表总标题
fig.suptitle('YOLOv8s vs YOLOv8m - Metrics Comparison', fontsize=16, fontweight='bold')

# 定义四项对比评估指标：字段名、图表标题
metrics = [
    ('metrics/precision(B)', 'Precision'),          # 精确率
    ('metrics/recall(B)', 'Recall'),                # 召回率
    ('metrics/mAP50(B)', 'mAP@0.5'),                # 0.5阈值平均精度
    ('metrics/mAP50-95(B)', 'mAP@0.5:0.95')         # 多阈值区间平均精度
]

# 遍历指标，逐个绘制对比曲线
for idx, (metric_name, title) in enumerate(metrics):
    # 计算子图行列位置
    ax = axes[idx // 2, idx % 2]

    # 数据清洗，剔除空值与无穷异常值
    df_s_clean = df_s[df_s[metric_name].notna() & ~df_s[metric_name].isin([float('inf'), float('-inf')])]
    df_m_clean = df_m[df_m[metric_name].notna() & ~df_m[metric_name].isin([float('inf'), float('-inf')])]

    # 分别绘制两个模型曲线，使用不同标记区分样式
    ax.plot(df_s_clean['epoch'], df_s_clean[metric_name], label='YOLOv8s',
            linewidth=2, alpha=0.8, marker='o', markersize=3, markevery=10)
    ax.plot(df_m_clean['epoch'], df_m_clean[metric_name], label='YOLOv8m',
            linewidth=2, alpha=0.8, marker='s', markersize=3, markevery=10)

    # 坐标轴与标题样式设置
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 标注模型最终迭代指标数值
    if len(df_s_clean) > 0:
        final_s = df_s_clean[metric_name].iloc[-1]
        ax.text(0.02, 0.98, f'YOLOv8s final: {final_s:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if len(df_m_clean) > 0:
        final_m = df_m_clean[metric_name].iloc[-1]
        ax.text(0.02, 0.88, f'YOLOv8m final: {final_m:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# 自动适配子图间距，防止元素重叠
plt.tight_layout()
# 定义图片保存路径
output_path = '/nas03/yixuh/garbage-classification/scripts/visulization/metrics_comparison.png'
# 保存高清图片
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Metrics comparison plot saved to: {output_path}")
# 关闭画布释放内存
plt.close()
