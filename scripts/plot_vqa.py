"""
Ve bieu do danh gia hieu nang mo hinh VQA (BLEU-1 va BLEU-2).
Bieu do gom 2 phan:
1. So sanh BLEU trung binh (Bar chart)
2. Phan phoi diem BLEU tren tap Validation (Box + Swarm plot)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Cau hinh duong dan
BASE_DIR = r"d:\DoAn_DaLieu"
REPORT_PATH = os.path.join(BASE_DIR, "5_Results", "vqa_evaluation_report.json")
OUT_IMG_PATH = os.path.join(BASE_DIR, "5_Results", "vqa_performance.png")

def main():
    print("Reading VQA evaluation report...")
    if not os.path.exists(REPORT_PATH):
        print(f"Error: Report file not found at {REPORT_PATH}")
        return 1

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    details = data.get("details", [])
    
    val_samples = summary.get("val_samples", len(details))
    avg_bleu1 = summary.get("average_bleu1", 0.0)
    avg_bleu2 = summary.get("average_bleu2", 0.0)
    
    print(f"Loaded {val_samples} validation samples.")
    print(f"Avg BLEU-1: {avg_bleu1:.4f}, Avg BLEU-2: {avg_bleu2:.4f}")

    bleu1_scores = [item.get("bleu1", 0.0) for item in details]
    bleu2_scores = [item.get("bleu2", 0.0) for item in details]

    # Thiet lap style cho matplotlib/seaborn
    sns.set_theme(style="whitegrid")
    
    # Tao figure voi 2 panels side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # --------------------------------------------------------------------------
    # Panel 1: Bar Plot (BLEU-1 vs BLEU-2 Average)
    # --------------------------------------------------------------------------
    metrics = ["BLEU-1 (1-gram)", "BLEU-2 (2-gram)"]
    averages = [avg_bleu1 * 100, avg_bleu2 * 100]  # Chuyen sang % de truc quan hon
    colors = ["#4F46E5", "#10B981"]  # Indigo and Emerald (Premium colors)
    
    bars = axes[0].bar(metrics, averages, color=colors, width=0.5, edgecolor="none", alpha=0.9)
    
    # Them gia tri tren dau cot
    for bar in bars:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1F2937"
        )
        
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Diem danh gia (%)", fontsize=12, fontweight="semibold", color="#374151")
    axes[0].set_title(f"Diem BLEU Trung Binh (N={val_samples} mau Validation)", fontsize=13, fontweight="bold", pad=15, color="#111827")
    axes[0].tick_params(axis="both", labelsize=11)
    
    # --------------------------------------------------------------------------
    # Panel 2: Box + Swarm Plot (BLEU Distribution)
    # --------------------------------------------------------------------------
    # Chuan bi du lieu dang tidy data cho seaborn
    plot_data = {
        "Metric": ["BLEU-1"] * len(bleu1_scores) + ["BLEU-2"] * len(bleu2_scores),
        "Score": [s * 100 for s in bleu1_scores] + [s * 100 for s in bleu2_scores]
    }
    
    # Box plot
    sns.boxplot(
        x="Metric", 
        y="Score", 
        data=plot_data, 
        ax=axes[1], 
        palette=colors,
        width=0.4, 
        fliersize=0, # An outliers vi da co swarmplot
        boxprops=dict(alpha=0.6)
    )
    
    # Swarm plot (Hien thi tung diem du lieu le)
    sns.stripplot(
        x="Metric", 
        y="Score", 
        data=plot_data, 
        ax=axes[1], 
        color="#374151", 
        size=6, 
        jitter=0.15,
        alpha=0.85
    )
    
    axes[1].set_ylim(-5, 105)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Phan bo diem (%)", fontsize=12, fontweight="semibold", color="#374151")
    axes[1].set_title("Phan Phoi Diem BLEU tren tap Validation", fontsize=13, fontweight="bold", pad=15, color="#111827")
    axes[1].tick_params(axis="both", labelsize=11)
    
    # Hieu chinh layout tong the
    plt.suptitle("DANH GIA HIEU NANG MO HINH VQA Y TE (CPUMedicalVQAModel)", fontsize=15, fontweight="bold", color="#1F2937", y=0.98)
    plt.tight_layout()
    
    # Luu file anh chat luong cao
    plt.savefig(OUT_IMG_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Successfully generated and saved VQA performance chart to {OUT_IMG_PATH}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
