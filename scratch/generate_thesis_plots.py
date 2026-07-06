import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set professional plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})

# Create directory for images if it doesn't exist
output_dir = "anh"
os.makedirs(output_dir, exist_ok=True)

# 1. Plot Segmentation Performance Comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
methods = ['U-Net (ResNet34)', 'DeepLabV3+ (ResNet50)', 'Hybrid DeepLabV3+ (Ablation)', 'Hybrid-Max Fusion']
dice_scores = [89.43, 91.28, 90.93, 91.32]
iou_scores = [81.77, 84.55, 84.33, 84.70]

x = np.arange(len(methods))
width = 0.35

rects1 = ax.bar(x - width/2, dice_scores, width, label='Dice Score (%)', color='#2b5c8f')
rects2 = ax.bar(x + width/2, iou_scores, width, label='IoU Score (%)', color='#d95f02')

ax.set_ylabel('Điểm số (%)')
ax.set_title('So sánh hiệu năng các mô hình phân đoạn tổn thương da')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15, ha='right')
ax.set_ylim(70, 100)
ax.legend(loc='lower right')

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'seg_comparison.png'), dpi=300)
plt.close()

# 2. Plot VQA Loss Curves
fig, ax = plt.subplots(figsize=(7, 4))
epochs = np.arange(1, 16)
train_loss = [2.9062, 2.7850, 2.6540, 2.5410, 2.4520, 2.3810, 2.3240, 2.2810, 2.2510, 2.2310, 2.2210, 2.2140, 2.2010, 2.1890, 2.1778]
val_loss = [2.7685, 2.6520, 2.5310, 2.4280, 2.3420, 2.2790, 2.2310, 2.1950, 2.1690, 2.1490, 2.1310, 2.1209, 2.1320, 2.1410, 2.1466]

ax.plot(epochs, train_loss, 'o-', color='#1f77b4', linewidth=2, label='Loss Tập huấn luyện (Train)')
ax.plot(epochs, val_loss, 's--', color='#ff7f0e', linewidth=2, label='Loss Tập kiểm chứng (Val)')

# Mark best epoch (Epoch 12)
ax.plot(12, 2.1209, 'ro', markersize=10, label='Mốc tối ưu (Epoch 12)')
ax.annotate('Best Epoch 12\nVal Loss: 2.1209', 
            xy=(12, 2.1209), 
            xytext=(13, 2.3),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))

ax.set_xlabel('Số lượng Epoch')
ax.set_ylabel('Giá trị Causal LM Loss')
ax.set_title('Động học hội tụ Loss của Trợ lý đàm thoại VQA (LoRA)')
ax.set_xticks(epochs)
ax.legend()
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'vqa_loss_curves.png'), dpi=300)
plt.close()

# 3. Plot Inference Breakdown (Pie Chart)
fig, ax = plt.subplots(figsize=(6, 5))
labels = [
    'Tiền xử lý & Khử lông\n(1.12 ms | 0.48%)',
    'Phân đoạn tổn thương\n(168.73 ms | 72.64%)',
    'ABCD & Trích xuất ROI\n(1.62 ms | 0.70%)',
    'Phân loại bệnh lý\n(60.80 ms | 26.17%)'
]
sizes = [1.12, 168.73, 1.62, 60.80]
colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
explode = (0, 0.05, 0, 0)  # Explode segmentation

wedges, texts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, startangle=140, 
                       shadow=False, textprops=dict(fontsize=10))

# Style titles
ax.set_title('Phân rã chi tiết thời gian suy luận toàn luồng Pipeline\n(Tổng thời gian đáp ứng: 232.27 ms)', fontsize=12, weight='bold')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'inference_breakdown.png'), dpi=300)
plt.close()

print("All plots generated successfully in 'anh' directory.")
