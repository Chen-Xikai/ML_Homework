"""
generate_figures.py
生成实验报告所需的所有图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def create_all_figures(output_dir='./report_figures'):
    """生成所有图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成图表...")
    
    fig1_architecture(output_dir)
    fig2_dataset_stats(output_dir)
    fig3_training_dynamics(output_dir)
    fig4_margin_ablation(output_dir)
    fig5_convergence_speed(output_dir)
    fig6_training_time(output_dir)
    fig7_test_accuracy_trajectory(output_dir)
    fig8_system_architecture(output_dir)
    fig9_data_flow(output_dir)
    fig10_overfitting_analysis(output_dir)
    
    print(f"所有图表已保存到: {output_dir}")


# ============================================================
# Figure 1: 网络架构示意图（双分支孪生网络）
# ============================================================
def fig1_architecture(save_dir):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(5, 11.5, 'Figure 1: Siamese Contrastive Network Architecture',
            ha='center', va='center', fontsize=13, fontweight='bold')
    
    # ===== 左分支（输入1） =====
    # 输入1
    bbox_input = dict(boxstyle='round,pad=0.3', facecolor='#3498db', alpha=0.3, edgecolor='#3498db', linewidth=2)
    ax.text(2, 10.5, 'Input 1\n1x105x105', ha='center', va='center', fontsize=9, bbox=bbox_input)
    ax.annotate('', xy=(2, 9.8), xytext=(2, 10.2), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # CNN Backbone 1
    bbox_cnn = dict(boxstyle='round,pad=0.3', facecolor='#2ecc71', alpha=0.3, edgecolor='#2ecc71', linewidth=2)
    ax.text(2, 9.2, 'CNN\nBackbone', ha='center', va='center', fontsize=9, bbox=bbox_cnn)
    ax.annotate('', xy=(2, 8.5), xytext=(2, 8.9), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # FC层 1
    bbox_fc = dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.3, edgecolor='#e74c3c', linewidth=2)
    ax.text(2, 7.9, 'FC + L2\n9216->512->128', ha='center', va='center', fontsize=8, bbox=bbox_fc)
    ax.annotate('', xy=(2, 7.2), xytext=(2, 7.6), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # 嵌入1
    bbox_embed = dict(boxstyle='round,pad=0.3', facecolor='#9b59b6', alpha=0.3, edgecolor='#9b59b6', linewidth=2)
    ax.text(2, 6.6, 'Embedding 1\n(128-dim)', ha='center', va='center', fontsize=9, bbox=bbox_embed)
    
    # ===== 右分支（输入2） =====
    # 输入2
    ax.text(8, 10.5, 'Input 2\n1x105x105', ha='center', va='center', fontsize=9, bbox=bbox_input)
    ax.annotate('', xy=(8, 9.8), xytext=(8, 10.2), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # CNN Backbone 2（共享权重）
    ax.text(8, 9.2, 'CNN\nBackbone', ha='center', va='center', fontsize=9, bbox=bbox_cnn)
    ax.annotate('', xy=(8, 8.5), xytext=(8, 8.9), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # FC层 2
    ax.text(8, 7.9, 'FC + L2\n9216->512->128', ha='center', va='center', fontsize=8, bbox=bbox_fc)
    ax.annotate('', xy=(8, 7.2), xytext=(8, 7.6), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # 嵌入2
    ax.text(8, 6.6, 'Embedding 2\n(128-dim)', ha='center', va='center', fontsize=9, bbox=bbox_embed)
    
    # ===== 共享权重标注 =====
    ax.annotate('Weight Sharing', xy=(5, 9.2), xytext=(5, 9.8),
                ha='center', va='center', fontsize=10, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
    
    # 连接线到Loss
    ax.annotate('', xy=(5, 5.5), xytext=(2, 6.3), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(5, 5.5), xytext=(8, 6.3), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # ===== 对比损失 =====
    bbox_loss = dict(boxstyle='round,pad=0.3', facecolor='#f39c12', alpha=0.3, edgecolor='#f39c12', linewidth=2)
    ax.text(5, 4.8, 'Contrastive Loss\nL = (1-Y)xD^2 + Yx[max(0,a-D)]^2', ha='center', va='center', fontsize=9, bbox=bbox_loss)
    
    # ===== 说明文字 =====
    ax.text(5, 3.5, 'Key Properties:', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.0, '1. Weight Sharing: Both CNN branches share identical parameters', ha='center', va='center', fontsize=9)
    ax.text(5, 2.5, '2. L2 Normalization: Embeddings mapped to unit sphere', ha='center', va='center', fontsize=9)
    ax.text(5, 2.0, '3. Margin Control: alpha controls min distance for negative pairs', ha='center', va='center', fontsize=9)
    
    # 数据流标注
    ax.text(5, 1.0, 'Training: Minimize distance for same-class pairs, maximize for different-class pairs',
            ha='center', va='center', fontsize=8, style='italic', color='#666')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 2: 数据集划分统计
# ============================================================
def fig2_dataset_stats(save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：数据集大小
    datasets = ['Train', 'Validation', 'Test']
    pairs = [31500, 6102, 7362]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = axes[0].bar(datasets, pairs, color=colors, edgecolor='black', linewidth=1.2)
    for bar, pair in zip(bars, pairs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{pair:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_ylabel('Number of Pairs', fontsize=12)
    axes[0].set_title('(a) Dataset Size', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # 右图：书写者划分
    writers = ['Train\n(Writers 1-12)', 'Validation\n(Writers 13-16)', 'Test\n(Writers 17-20)']
    writer_counts = [12, 4, 4]
    
    bars2 = axes[1].bar(writers, writer_counts, color=colors, edgecolor='black', linewidth=1.2)
    for bar, count in zip(bars2, writer_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[1].set_xlabel('Dataset', fontsize=12)
    axes[1].set_ylabel('Number of Writers', fontsize=12)
    axes[1].set_title('(b) Writer Partition', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    fig.suptitle('Figure 2: Dataset Statistics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_dataset_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 3: 训练动态曲线
# ============================================================
def fig3_training_dynamics(save_dir):
    epochs = np.arange(1, 25)
    train_loss = [0.0281, 0.0190, 0.0151, 0.0124, 0.0104, 0.0087, 0.0075, 0.0064,
                  0.0057, 0.0050, 0.0044, 0.0040, 0.0027, 0.0023, 0.0021, 0.0019,
                  0.0018, 0.0017, 0.0012, 0.0010, 0.0010, 0.0009, 0.0009, 0.0008]
    val_acc = [86.56, 91.08, 90.94, 91.54, 91.38, 91.20, 90.33, 89.87,
               89.99, 88.82, 88.54, 89.33, 87.87, 85.97, 86.79, 84.25,
               85.35, 84.20, 83.07, 83.43, 82.66, 82.12, 83.04, 81.27]
    test_acc = [48.75, 57.50, 62.75, 69.00, 62.50, 66.25, 59.75, 68.50,
                62.50, 61.00, 66.75, 61.00, 61.00, 65.75, 63.25, 58.00,
                59.25, 60.25, 59.25, 64.50, 65.75, 59.75, 62.75, 64.75]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 训练损失（左Y轴）
    color1 = '#3498db'
    ax1.plot(epochs, train_loss, 'o-', color=color1, linewidth=2, markersize=5, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale('log')
    
    # 验证准确率和测试准确率（右Y轴）
    ax2 = ax1.twinx()
    color2 = '#2ecc71'
    color3 = '#e74c3c'
    ax2.plot(epochs, val_acc, 's-', color=color2, linewidth=2, markersize=5, label='Val Acc')
    ax2.plot(epochs, test_acc, 'D-', color=color3, linewidth=2, markersize=5, label='Test Acc (20-way 1-shot)')
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    
    # 标注最佳epoch
    ax2.axvline(x=4, color='gray', linestyle='--', alpha=0.7, label='Best Epoch (4)')
    ax2.annotate('Best: 69.00%', xy=(4, 69.00), xytext=(6, 72),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
    
    ax1.set_title('Figure 3: Training Dynamics (Margin=0.5)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_training_dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 4: Margin消融实验
# ============================================================
def fig4_margin_ablation(save_dir):
    margins = [0.5, 1.0, 1.5, 2.0]
    test_acc = [66.75, 64.75, 64.00, 59.50]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(margins, test_acc, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    
    for bar, acc in zip(bars, test_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 标注最佳值
    ax.axhline(y=66.75, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Best (66.75%)')
    
    ax.set_xlabel('Margin Value', fontsize=12)
    ax.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax.set_title('Figure 4: Ablation Study on Margin', fontsize=14, fontweight='bold')
    ax.set_xticks(margins)
    ax.set_ylim([58, 68])  # 调整范围，聚焦数据区域
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_margin_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 5: 收敛速度对比
# ============================================================
def fig5_convergence_speed(save_dir):
    margins = [0.5, 1.0, 1.5, 2.0]
    best_acc = [66.75, 64.75, 64.00, 59.50]
    best_epoch = [5, 6, 5, 5]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：Best Test Accuracy
    color1 = '#3498db'
    ax1.plot(margins, best_acc, 'o-', color=color1, linewidth=2, markersize=8)
    ax1.set_xlabel('Margin Value', fontsize=12)
    ax1.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Best Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(margins)
    ax1.set_ylim([58, 68])
    ax1.grid(True, alpha=0.3)
    
    # 标注数值
    for m, acc in zip(margins, best_acc):
        ax1.annotate(f'{acc:.2f}%', (m, acc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    # 右图：Best Epoch
    color2 = '#e74c3c'
    ax2.plot(margins, best_epoch, 's-', color=color2, linewidth=2, markersize=8)
    ax2.set_xlabel('Margin Value', fontsize=12)
    ax2.set_ylabel('Best Epoch', fontsize=12)
    ax2.set_title('(b) Convergence Speed', fontsize=13, fontweight='bold')
    ax2.set_xticks(margins)
    ax2.set_ylim([3, 8])
    ax2.grid(True, alpha=0.3)
    
    # 标注数值
    for m, ep in zip(margins, best_epoch):
        ax2.annotate(f'Epoch {ep}', (m, ep), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    fig.suptitle('Figure 5: Convergence Speed vs Margin', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_convergence_speed.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 6: 训练时间对比
# ============================================================
def fig6_training_time(save_dir):
    margins = [0.5, 1.0, 1.5, 2.0]
    times = [631, 602, 605, 603]  # 秒
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars = ax.bar(margins, times, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{t}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Margin Value', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Figure 6: Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(margins)
    ax.set_ylim([590, 640])  # 调整范围，聚焦数据区域
    ax.grid(axis='y', alpha=0.3)
    
    # 添加平均值线
    avg_time = np.mean(times)
    ax.axhline(y=avg_time, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Average: {avg_time:.0f}s')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_training_time.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 7: 测试准确率轨迹
# ============================================================
def fig7_test_accuracy_trajectory(save_dir):
    epochs = np.arange(1, 25)
    test_acc = [48.75, 57.50, 62.75, 69.00, 62.50, 66.25, 59.75, 68.50,
                62.50, 61.00, 66.75, 61.00, 61.00, 65.75, 63.25, 58.00,
                59.25, 60.25, 59.25, 64.50, 65.75, 59.75, 62.75, 64.75]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, test_acc, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    
    # 标注最佳点
    ax.annotate('Best: 69.00%\n(Epoch 4)', xy=(4, 69.00), xytext=(8, 70),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # 标注早停点
    ax.annotate('Early Stop\n(Epoch 24)', xy=(24, 64.75), xytext=(20, 60),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')
    
    ax.axhline(y=69.00, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Best Accuracy')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Figure 7: Test Accuracy Trajectory (Margin=0.5)', fontsize=14, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig7_test_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 8: 系统架构图
# ============================================================
def fig8_system_architecture(save_dir):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(6, 9.5, 'Figure 8: System Architecture', ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    # 入口层
    bbox_entry = dict(boxstyle='round,pad=0.4', facecolor='#3498db', alpha=0.3, edgecolor='#3498db', linewidth=2)
    ax.text(3, 8.5, 'train.py\n(Local Training)', ha='center', va='center', fontsize=10, bbox=bbox_entry)
    ax.text(6, 8.5, 'cloud_train.py\n(Cloud Training)', ha='center', va='center', fontsize=10, bbox=bbox_entry)
    ax.text(9, 8.5, 'margin_ablation.py\n(Ablation)', ha='center', va='center', fontsize=10, bbox=bbox_entry)
    
    ax.text(6, 8.0, 'Entry Layer', ha='center', va='center', fontsize=11, fontweight='bold', color='#3498db')
    
    # 数据层
    bbox_data = dict(boxstyle='round,pad=0.4', facecolor='#2ecc71', alpha=0.3, edgecolor='#2ecc71', linewidth=2)
    ax.text(4, 6.5, 'dataset.py\n(Real-time Loading)', ha='center', va='center', fontsize=10, bbox=bbox_data)
    ax.text(8, 6.5, 'cache_dataset.py\n(Pickle Cache)', ha='center', va='center', fontsize=10, bbox=bbox_data)
    
    ax.text(6, 6.0, 'Data Layer', ha='center', va='center', fontsize=11, fontweight='bold', color='#2ecc71')
    
    # 模型层
    bbox_model = dict(boxstyle='round,pad=0.4', facecolor='#e74c3c', alpha=0.3, edgecolor='#e74c3c', linewidth=2)
    ax.text(6, 4.5, 'model.py\nSiameseContrastive + ContrastiveLoss', ha='center', va='center', fontsize=10, bbox=bbox_model)
    
    ax.text(6, 4.0, 'Model Layer', ha='center', va='center', fontsize=11, fontweight='bold', color='#e74c3c')
    
    # 评估层
    bbox_eval = dict(boxstyle='round,pad=0.4', facecolor='#f39c12', alpha=0.3, edgecolor='#f39c12', linewidth=2)
    ax.text(6, 2.5, 'evaluate.py\nFewShotEvaluator (20-way 1-shot)', ha='center', va='center', fontsize=10, bbox=bbox_eval)
    
    ax.text(6, 2.0, 'Evaluation Layer', ha='center', va='center', fontsize=11, fontweight='bold', color='#f39c12')
    
    # 工具层
    bbox_util = dict(boxstyle='round,pad=0.4', facecolor='#9b59b6', alpha=0.3, edgecolor='#9b59b6', linewidth=2)
    ax.text(6, 0.8, 'utils.py\nset_seed / EarlyStopping / AverageMeter', ha='center', va='center', fontsize=10, bbox=bbox_util)
    
    ax.text(6, 0.3, 'Utility Layer', ha='center', va='center', fontsize=11, fontweight='bold', color='#9b59b6')
    
    # 箭头连接
    for y_start, y_end in [(8.2, 7.0), (6.2, 5.0), (4.2, 3.0), (2.2, 1.2)]:
        ax.annotate('', xy=(6, y_end), xytext=(6, y_start),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig8_system_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 9: 数据流图
# ============================================================
def fig9_data_flow(save_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(7, 7.5, 'Figure 9: Data Flow Diagram', ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    # 训练数据流
    steps_train = [
        (1, 6, 'Omniglot\nDataset', '#3498db'),
        (3.5, 6, 'Writer Split\n(1-12)', '#3498db'),
        (6, 6, 'Pair\nGeneration', '#3498db'),
        (8.5, 6, 'Cache\nWarmup', '#3498db'),
        (11, 6, 'DataLoader\n(Batch)', '#3498db')
    ]
    
    for x, y, text, color in steps_train:
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, bbox=bbox)
    
    ax.text(7, 6.7, 'Training Data Flow', ha='center', va='center', fontsize=11, fontweight='bold', color='#3498db')
    
    # 评估数据流
    steps_eval = [
        (1, 3.5, 'Omniglot\nDataset', '#e74c3c'),
        (3.5, 3.5, 'Test Split\n(Writers 17-20)', '#e74c3c'),
        (6, 3.5, 'Episode\nCreation', '#e74c3c'),
        (8.5, 3.5, 'Embedding\nExtraction', '#e74c3c'),
        (11, 3.5, 'Nearest\nNeighbor', '#e74c3c')
    ]
    
    for x, y, text, color in steps_eval:
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, bbox=bbox)
    
    ax.text(7, 4.2, 'Evaluation Data Flow', ha='center', va='center', fontsize=11, fontweight='bold', color='#e74c3c')
    
    # 箭头
    for x_start, x_end in [(1.8, 2.7), (4.3, 5.2), (6.8, 7.7), (9.3, 10.2)]:
        ax.annotate('', xy=(x_end, 6), xytext=(x_start, 6),
                    arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5))
    
    for x_start, x_end in [(1.8, 2.7), (4.3, 5.2), (6.8, 7.7), (9.3, 10.2)]:
        ax.annotate('', xy=(x_end, 3.5), xytext=(x_start, 3.5),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig9_data_flow.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 10: 过拟合分析
# ============================================================
def fig10_overfitting_analysis(save_dir):
    epochs = np.arange(1, 25)
    train_loss = [0.0281, 0.0190, 0.0151, 0.0124, 0.0104, 0.0087, 0.0075, 0.0064,
                  0.0057, 0.0050, 0.0044, 0.0040, 0.0027, 0.0023, 0.0021, 0.0019,
                  0.0018, 0.0017, 0.0012, 0.0010, 0.0010, 0.0009, 0.0009, 0.0008]
    val_acc = [86.56, 91.08, 90.94, 91.54, 91.38, 91.20, 90.33, 89.87,
               89.99, 88.82, 88.54, 89.33, 87.87, 85.97, 86.79, 84.25,
               85.35, 84.20, 83.07, 83.43, 82.66, 82.12, 83.04, 81.27]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：Train Loss（对数尺度）
    color1 = '#3498db'
    ax1.plot(epochs, train_loss, 'o-', color=color1, linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss (log scale)', fontsize=12)
    ax1.set_title('(a) Training Loss', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim([0.0005, 0.05])
    ax1.grid(True, alpha=0.3)
    
    # 标注最佳点
    ax1.axvline(x=4, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Epoch')
    ax1.legend(fontsize=10)
    
    # 右图：Validation Accuracy
    color2 = '#e74c3c'
    ax2.plot(epochs, val_acc, 's-', color=color2, linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_ylim([78, 94])
    ax2.grid(True, alpha=0.3)
    
    # 标注最佳点
    ax2.axvline(x=4, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Epoch')
    ax2.annotate('Best: 91.54%', xy=(4, 91.54), xytext=(8, 92.5),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, fontweight='bold')
    ax2.legend(fontsize=10)
    
    # 标注过拟合区域
    ax2.axvspan(4, 24, alpha=0.1, color='red', label='Overfitting')
    ax2.annotate('Overfitting\n(Loss decreases, Acc decreases)', 
                xy=(14, 85), xytext=(16, 88),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red', fontweight='bold')
    
    fig.suptitle('Figure 10: Overfitting Analysis (Margin=0.5)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig10_overfitting.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    create_all_figures()
