import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ==========================================
# DATA EXTRACTION AND CALCULATION
# ==========================================

m1_tn, m1_fp = 78, 18
m1_fn, m1_tp = 26, 81
m1_total = m1_tn + m1_fp + m1_fn + m1_tp

m2_tn, m2_fp = 139, 48
m2_fn, m2_tp = 22, 195
m2_total = m2_tn + m2_fp + m2_fn + m2_tp

m1_radar = [0.75, 0.81, 0.78, 0.82, 0.76, 0.79]

m2_neg_prec = m2_tn / (m2_tn + m2_fn)
m2_neg_rec = m2_tn / (m2_tn + m2_fp)
m2_neg_f1 = 2 * (m2_neg_prec * m2_neg_rec) / (m2_neg_prec + m2_neg_rec)
m2_pos_prec = m2_tp / (m2_tp + m2_fp)
m2_pos_rec = m2_tp / (m2_tp + m2_fn)
m2_pos_f1 = 2 * (m2_pos_prec * m2_pos_rec) / (m2_pos_prec + m2_pos_rec)
m2_radar = [m2_neg_prec, m2_neg_rec, m2_neg_f1, m2_pos_prec, m2_pos_rec, m2_pos_f1]

m1_norm = [m1_tn/m1_total, m1_fp/m1_total, m1_fn/m1_total, m1_tp/m1_total]
m2_norm = [m2_tn/m2_total, m2_fp/m2_total, m2_fn/m2_total, m2_tp/m2_total]

m1_tpr = m1_tp / (m1_tp + m1_fn)
m1_fpr = m1_fp / (m1_fp + m1_tn)

m2_tpr = m2_tp / (m2_tp + m2_fn)
m2_fpr = m2_fp / (m2_fp + m2_tn)


# ==========================================
# GRAPH 2: RADAR CHART (SPIDER PLOT)
# ==========================================
categories = ['Neg Precision', 'Neg Recall', 'Neg F1-Score', 
              'Pos Precision', 'Pos Recall', 'Pos F1-Score']
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] # Close the circle

fig1, ax1 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Add Model 1
values1 = m1_radar + m1_radar[:1]
ax1.plot(angles, values1, linewidth=2, linestyle='solid', label='Perch Embeddings', color='#4C72B0')
ax1.fill(angles, values1, '#4C72B0', alpha=0.25)

# Add Model 2
values2 = m2_radar + m2_radar[:1]
ax1.plot(angles, values2, linewidth=2, linestyle='solid', label='CNN (Convolutional Neural Net)', color='#55A868')
ax1.fill(angles, values2, '#55A868', alpha=0.25)

plt.xticks(angles[:-1], categories, color='grey', size=11)
ax1.set_rlabel_position(0)
plt.yticks([0.70, 0.80, 0.90, 1.00], ["0.7", "0.8", "0.9", "1.0"], color="grey", size=10)
plt.ylim(0.65, 0.95)
plt.title('Radar Chart: Class-Specific Metrics Comparison', size=14, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))


# ==========================================
# GRAPH 3: NORMALIZED STACKED BAR CHART
# ==========================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

labels = ['Perch Embeddings', 'CNN (Convolutional Neural Net)']
categories_stack = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'] # Green, Blue, Orange, Red

# Format data for stacking
data = np.array([
    [m1_norm[3], m2_norm[3]], # TP
    [m1_norm[0], m2_norm[0]], # TN
    [m1_norm[1], m2_norm[1]], # FP
    [m1_norm[2], m2_norm[2]]  # FN
]) * 100 # Convert to percentages

bottom = np.zeros(2)

for i in range(4):
    p = ax2.bar(labels, data[i], 0.5, label=categories_stack[i], bottom=bottom, color=colors[i])
    bottom += data[i]
    # Add text labels inside bars
    ax2.bar_label(p, label_type='center', fmt='%.1f%%', color='white', fontweight='bold')

ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Normalized Confusion Matrix Error Breakdown', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
plt.tight_layout()


# ==========================================
# GRAPH 4: SINGLE-POINT ROC PLOT
# ==========================================
fig3, ax3 = plt.subplots(figsize=(7, 6))

ax3.scatter(m1_fpr, m1_tpr, s=150, color='#4C72B0', label='Model 1 (07-10)', zorder=5)
ax3.scatter(m2_fpr, m2_tpr, s=150, color='#55A868', label='Model 2 (07-11)', zorder=5)

# Add reference diagonal line (random guessing)
ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

# Annotations
ax3.annotate(f'M1\n(FPR: {m1_fpr:.2f}, TPR: {m1_tpr:.2f})', 
             (m1_fpr, m1_tpr), xytext=(10, -20), textcoords='offset points')
ax3.annotate(f'M2\n(FPR: {m2_fpr:.2f}, TPR: {m2_tpr:.2f})', 
             (m2_fpr, m2_tpr), xytext=(-80, 10), textcoords='offset points')

ax3.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax3.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
ax3.set_title('Single-Point ROC Plot (TPR vs. FPR)', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 0.4) # Zooms in on the relevant area
ax3.set_ylim(0.5, 1.0)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='lower right')

# Render all plots
plt.show()
