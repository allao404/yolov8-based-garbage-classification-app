# 肖晴标注
#!/usr/bin/env python3
"""
对比YOLOv8s和YOLOv8m的训练和验证损失变化趋势
功能：生成训练/验证损失对比图，直观对比两个模型的收敛效果
"""

# 导入依赖库
import pandas as pd       # 用于读取和处理训练日志CSV数据
import matplotlib.pyplot as plt  # 用于绘制图表
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在无图形界面环境下报错

# 配置matplotlib，支持中文显示和负号
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------------------- 数据读取与预处理 ----------------------
# 定义两个模型训练日志CSV文件的路径
yolov8s_path = '/nas03/yixuh/garbage-classification/models/garbage_yolov8s/results.csv'
yolov8m_path = '/nas03/yixuh/garbage-classification/models/garbage_yolov8m/results.csv'

# 读取CSV文件到DataFrame
df_s = pd.read_csv(yolov8s_path)  # YOLOv8s的训练数据
df_m = pd.read_csv(yolov8m_path)  # YOLOv8m的训练数据

# 去除列名中的多余空格（CSV文件导出时常见问题）
df_s.columns = df_s.columns.str.strip()
df_m.columns = df_m.columns.str.strip()

# ---------------------- 创建图表框架 ----------------------
# 创建2行3列的子图布局，设置整体尺寸
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# 设置图表总标题
fig.suptitle('YOLOv8s vs YOLOv8m - Loss Comparison', fontsize=16, fontweight='bold')

# ---------------------- 训练损失对比（第一行） ----------------------
# 定义需要对比的训练损失项：(列名, 图表标题)
losses = [
    ('train/box_loss', 'Training Box Loss'),      # 边界框回归损失
    ('train/cls_loss', 'Training Classification Loss'),  # 分类损失
    ('train/dfl_loss', 'Training DFL Loss')      # 分布焦点损失
]

# 遍历每个损失项，绘制对比曲线
for idx, (loss_name, title) in enumerate(losses):
    ax = axes[0, idx]  # 选择第1行第idx列的子图
    # 绘制YOLOv8s的损失曲线
    ax.plot(df_s['epoch'], df_s[loss_name], label='YOLOv8s', linewidth=2, alpha=0.8)
    # 绘制YOLOv8m的损失曲线
    ax.plot(df_m['epoch'], df_m[loss_name], label='YOLOv8m', linewidth=2, alpha=0.8)
    # 设置坐标轴标签和标题
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()  # 显示图例
    ax.grid(True, alpha=0.3)  # 添加网格线，提高可读性

# ---------------------- 验证损失对比（第二行） ----------------------
# 定义需要对比的验证损失项：(列名, 图表标题)
val_losses = [
    ('val/box_loss', 'Validation Box Loss'),      # 验证集边界框回归损失
    ('val/cls_loss', 'Validation Classification Loss'),  # 验证集分类损失
    ('val/dfl_loss', 'Validation DFL Loss')      # 验证集分布焦点损失
]

# 遍历每个验证损失项，绘制对比曲线
for idx, (loss_name, title) in enumerate(val_losses):
    ax = axes[1, idx]  # 选择第2行第idx列的子图
    
    # 数据清洗：过滤掉NaN和无穷值，避免绘图报错
    df_s_clean = df_s[df_s[loss_name].notna() & ~df_s[loss_name].isin([float('inf'), float('-inf')])]
    df_m_clean = df_m[df_m[loss_name].notna() & ~df_m[loss_name].isin([float('inf'), float('-inf')])]

    # 绘制清洗后的数据曲线
    ax.plot(df_s_clean['epoch'], df_s_clean[loss_name], label='YOLOv8s', linewidth=2, alpha=0.8)
    ax.plot(df_m_clean['epoch'], df_m_clean[loss_name], label='YOLOv8m', linewidth=2, alpha=0.8)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()  # 显示图例
    ax.grid(True, alpha=0.3)  # 添加网格线

# ---------------------- 图表保存与收尾 ----------------------
plt.tight_layout()  # 自动调整子图间距，避免重叠
# 定义输出文件路径
output_path = '/nas03/yixuh/garbage-classification/scripts/visulization/loss_comparison.png'
# 保存图片，设置分辨率和裁剪选项
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Loss comparison plot saved to: {output_path}")
plt.close()  # 关闭图表，释放内存
