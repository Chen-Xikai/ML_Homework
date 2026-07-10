"""
generate_paper_figures.py - 生成论文图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.join(BASE_DIR, "..")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MAIN_LOG_DIR = os.path.join(MAIN_DIR, "logs")
FIG_DIR = os.path.join(BASE_DIR, "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLOR_M1 = "#E74C3C"
COLOR_M2 = "#2ECC71"
COLOR_SIAM = "#3498DB"

def load_method1_log():
    return pd.read_csv(os.path.join(LOG_DIR, "method1_log.csv"))

def load_method2_log():
    return pd.read_csv(os.path.join(LOG_DIR, "method2_log.csv"))

def load_siamese_summary():
    return pd.read_csv(os.path.join(MAIN_LOG_DIR, "summary.csv"), encoding="gbk")

def fig1_loss_curve():
    m1 = load_method1_log()
    m2 = load_method2_log()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(m1["epoch"], m1["train_loss"], color=COLOR_M1, linewidth=2, label="Method 1 (CNN+KNN)", alpha=0.8)
    ax.plot(m2["epoch"], m2["train_loss"], color=COLOR_M2, linewidth=2, label="Method 2 (End-to-End)", alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Training Loss", fontsize=14)
    ax.set_title("Training Loss Comparison", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] fig1_loss_curve.png")

def fig2_train_acc_curve():
    m1 = load_method1_log()
    m2 = load_method2_log()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(m1["epoch"], m1["train_acc"] * 100, color=COLOR_M1, linewidth=2, label="Method 1 (CNN+KNN)", alpha=0.8)
    ax.plot(m2["epoch"], m2["train_acc"] * 100, color=COLOR_M2, linewidth=2, label="Method 2 (End-to-End)", alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Training Accuracy (%)", fontsize=14)
    ax.set_title("Training Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_train_acc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] fig2_train_acc_curve.png")

def fig3_val_acc_curve():
    m1 = load_method1_log()
    m2 = load_method2_log()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(m1["epoch"], m1["val_acc"] * 100, color=COLOR_M1, linewidth=2, label="Method 1 (CNN+KNN)", alpha=0.8)
    ax.plot(m2["epoch"], m2["val_acc"] * 100, color=COLOR_M2, linewidth=2, label="Method 2 (End-to-End)", alpha=0.8)
    m1_best_idx = m1["val_acc"].idxmax()
    m2_best_idx = m2["val_acc"].idxmax()
    ax.annotate(f"Best: {m1['val_acc'].iloc[m1_best_idx]*100:.1f}%\n(Epoch {m1['epoch'].iloc[m1_best_idx]})",
                xy=(m1["epoch"].iloc[m1_best_idx], m1["val_acc"].iloc[m1_best_idx]*100),
                xytext=(m1["epoch"].iloc[m1_best_idx]+10, m1["val_acc"].iloc[m1_best_idx]*100-5),
                arrowprops=dict(arrowstyle="->", color=COLOR_M1), fontsize=10, color=COLOR_M1)
    ax.annotate(f"Best: {m2['val_acc'].iloc[m2_best_idx]*100:.1f}%\n(Epoch {m2['epoch'].iloc[m2_best_idx]})",
                xy=(m2["epoch"].iloc[m2_best_idx], m2["val_acc"].iloc[m2_best_idx]*100),
                xytext=(m2["epoch"].iloc[m2_best_idx]+10, m2["val_acc"].iloc[m2_best_idx]*100+3),
                arrowprops=dict(arrowstyle="->", color=COLOR_M2), fontsize=10, color=COLOR_M2)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=14)
    ax.set_title("Validation Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_val_acc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] fig3_val_acc_curve.png")

def fig4_test_acc_bar():
    methods = ["Siamese\n(Trial 1)", "Method 1\n(CNN+KNN)", "Method 2\n(End-to-End)"]
    test_accs = [86.50, 56.58, 79.48]
    colors = [COLOR_SIAM, COLOR_M1, COLOR_M2]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, test_accs, color=colors, width=0.6, edgecolor="black", linewidth=1.2)
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_title("Test Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_test_acc_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] fig4_test_acc_bar.png")

def fig5_val_acc_bar():
    methods = ["Siamese\n(Trial 1)", "Method 1\n(CNN+KNN)", "Method 2\n(End-to-End)"]
    val_accs = [84.06, 77.91, 80.14]
    colors = [COLOR_SIAM, COLOR_M1, COLOR_M2]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, val_accs, color=colors, width=0.6, edgecolor="black", linewidth=1.2)
    for bar, acc in zip(bars, val_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=14)
    ax.set_title("Validation Accuracy Comparison", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_val_acc_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] fig5_val_acc_bar.png")

def fig6_resume_effect():
    m2 = load_method2_log()
    m2_phase1 = m2[m2["epoch"] <= 100]
    m2_phase2 = m2[m2["epoch"] > 100]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(m2_phase1["epoch"], m2_phase1["val_acc"] * 100, color=COLOR_M2, linewidth=2, label="Phase 1 (Epoch 1-100)", alpha=0.8)
    if len(m2_phase2) > 0:
        ax1.plot(m2_phase2["epoch"], m2_phase2["val_acc"] * 100, color="#27AE60", linewidth=2, label="Phase 2 (Epoch 101-161)", alpha=0.8)
    ax1.axvline(x=100, color="gray", linestyle="--", alpha=0.5, label="Resume Point")
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=14)
    ax1.set_title("Method 2: Complete Training Curve", fontsize=16, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    phases = ["Phase 1\n(Epoch 1-100)", "Phase 2\n(Epoch 101-161)", "Overall\nBest"]
    best_vals = [m2_phase1["val_acc"].max() * 100, m2_phase2["val_acc"].max() * 100 if len(m2_phase2) > 0 else 0, m2["val_acc"].max() * 100]
    colors = [COLOR_M2, "#27AE60", "#1ABC9C"]
    bars = ax2.bar(phases, best_vals, color=colors, width=0.6, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars, best_vals):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Best Validation Accuracy (%)", fontsize=14)
    ax2.set_title("Phase Comparison", fontsize=16, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.tick_params(labelsize=12)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_resume_effect.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] fig6_resume_effect.png")

def fig7_trials_bar():
    summary = load_siamese_summary()
    trials = [f"Trial {i}" for i in summary["trial_id"]]
    test_accs = summary["test_accuracy"] * 100
    val_accs = summary["best_val_one_shot_acc"] * 100
    x = np.arange(len(trials))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, val_accs, width, label="Validation Accuracy", color=COLOR_SIAM, alpha=0.7, edgecolor="black")
    bars2 = ax.bar(x + width/2, test_accs, width, label="Test Accuracy", color="#2980B9", alpha=0.9, edgecolor="black")
    best_idx = test_accs.idxmax()
    ax.annotate(f"Best: {test_accs.iloc[best_idx]:.1f}%",
                xy=(x[best_idx] + width/2, test_accs.iloc[best_idx]),
                xytext=(x[best_idx] + 0.8, test_accs.iloc[best_idx] + 2),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=12, color="red", fontweight="bold")
    ax.set_xlabel("Trial", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title("Siamese Network: 10 Trials Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(trials, fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_trials_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] fig7_trials_bar.png")

def fig8_time_bar():
    methods = ["Siamese\n(Trial 1)", "Method 1\n(CNN+KNN)", "Method 2\n(End-to-End)"]
    times_min = [170.5, 57.3, 74.4]
    colors = [COLOR_SIAM, COLOR_M1, COLOR_M2]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, times_min, color=colors, width=0.6, edgecolor="black", linewidth=1.2)
    for bar, t in zip(bars, times_min):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f"{t:.1f} min", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Training Time (minutes)", fontsize=14)
    ax.set_title("Training Time Comparison", fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_time_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  [OK] fig8_time_bar.png")

def main():
    print("=" * 60)
    print("生成论文图表")
    print("=" * 60)
    fig1_loss_curve()
    fig2_train_acc_curve()
    fig3_val_acc_curve()
    fig4_test_acc_bar()
    fig5_val_acc_bar()
    fig6_resume_effect()
    fig7_trials_bar()
    fig8_time_bar()
    print("=" * 60)
    print(f"所有图表已保存至: {FIG_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
