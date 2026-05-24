#!/usr/bin/env python3
"""
一键运行所有可视化脚本
功能：批量调用多个可视化脚本，生成模型训练对比图表
"""

# 导入系统工具库
import subprocess  # 用于在Python中执行外部命令/脚本
import sys         # 用于获取系统参数和退出程序
import os          # 用于处理文件路径

# 定义需要批量运行的可视化脚本列表
scripts = [
    'plot_loss_comparison.py',      # 模型损失对比图脚本
    'plot_metrics_comparison.py',   # 模型指标对比图脚本
    'plot_training_overview.py'     # 训练过程总览图脚本
]

# 脚本文件所在的目录路径
script_dir = '/nas03/yixuh/garbage-classification/scripts/visulization'

# 打印分隔线，美化输出
print("="*60)
print("Running all visualization scripts...")
print("="*60)

# 遍历所有脚本，逐个执行
for script in scripts:
    # 拼接脚本的完整路径
    script_path = os.path.join(script_dir, script)
    print(f"\n>>> Running {script}...")
    
    try:
        # 调用python3命令运行脚本
        result = subprocess.run(
            ['python3', script_path],  # 执行命令：python3 脚本路径
            capture_output=True,        # 捕获脚本的标准输出和错误输出
            text=True,                  # 以文本形式（而非字节）返回输出
            check=True                  # 如果脚本执行出错，自动抛出异常
        )
        
        # 打印脚本的标准输出内容
        print(result.stdout)
        
        # 如果脚本有警告信息（stderr），也打印出来
        if result.stderr:
            print(f"Warnings: {result.stderr}")
    
    # 捕获脚本执行失败的异常
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}:")
        # 打印脚本的错误信息
        print(e.stderr)
        # 脚本出错时直接退出整个程序，避免后续脚本继续执行
        sys.exit(1)

# 所有脚本执行完成，打印结果汇总
print("\n" + "="*60)
print("All visualizations completed successfully!")
print("="*60)
print("\nGenerated plots:")
print("  1. loss_comparison.png - Loss comparison between models")
print("  2. metrics_comparison.png - Metrics comparison between models")
print("  3. yolov8s_training_overview.png - YOLOv8s detailed overview")
print("  4. yolov8m_training_overview.png - YOLOv8m detailed overview")
print(f"\nOutput directory: {script_dir}")
