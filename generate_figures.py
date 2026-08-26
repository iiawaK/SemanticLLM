"""
Generate figures for SemanticLLM paper.
Run: python generate_figures.py
"""
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figs', exist_ok=True)

# ============================================================
# Figure 1: BERTScore vs SNR
# ============================================================
snr = np.array([-10, -5, 0, 5, 10, 15, 20])
bertscore_semanticllm = [0.62, 0.82, 0.89, 0.93, 0.95, 0.96, 0.97]
bertscore_transformer = [0.35, 0.52, 0.74, 0.82, 0.87, 0.89, 0.91]
bertscore_deepjscc    = [0.30, 0.48, 0.71, 0.79, 0.84, 0.87, 0.89]
bertscore_bpgldpc     = [0.20, 0.38, 0.56, 0.68, 0.75, 0.80, 0.83]

plt.figure(figsize=(8, 5))
plt.plot(snr, bertscore_semanticllm, 'r-o', linewidth=2, markersize=8, label='SemanticLLM (Ours)')
plt.plot(snr, bertscore_transformer, 'b-s', linewidth=1.5, markersize=6, label='Transformer SC')
plt.plot(snr, bertscore_deepjscc, 'g-^', linewidth=1.5, markersize=6, label='DeepJSCC')
plt.plot(snr, bertscore_bpgldpc, 'k--d', linewidth=1.5, markersize=6, label='BPG + LDPC')
plt.xlabel('SNR (dB)', fontsize=14)
plt.ylabel('BERTScore', fontsize=14)
plt.title('Text Transmission: BERTScore vs SNR', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([-12, 22])
plt.ylim([0.1, 1.0])
plt.tight_layout()
plt.savefig('figs/fig_bertscore.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figs/fig_bertscore.png', dpi=300, bbox_inches='tight')
print('Saved figs/fig_bertscore.pdf')

# ============================================================
# Figure 2: Framework Architecture (schematic)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

# Transmitter
boxes = [
    (0.5, 1.5, 2.5, 1.2, 'LLM\nEncoder', '#FFD700'),
    (3.5, 1.5, 2.5, 1.2, 'Channel\nProjection', '#87CEEB'),
    (6.5, 1.5, 2.0, 1.2, 'OFDM\nModulator', '#90EE90'),
    (9.0, 1.5, 2.0, 1.2, 'Wireless\nChannel', '#FFB6C1'),
    (11.5, 1.5, 2.0, 1.2, 'LLM\nDecoder', '#FFD700'),
]

for x, y, w, h, label, color in boxes:
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor=color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=11, fontweight='bold')

# Arrows
arrow_style = dict(arrowstyle='->', color='black', lw=2)
for i in range(len(boxes)-1):
    x_start = boxes[i][0] + boxes[i][2]
    x_end = boxes[i+1][0]
    y_mid = boxes[i][1] + boxes[i][3]/2
    ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid), arrowprops=arrow_style)

# Labels
ax.text(1.75, 3.2, 'Transmitter', ha='center', fontsize=13, fontweight='bold', color='blue')
ax.text(10.0, 3.2, 'Receiver', ha='center', fontsize=13, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('figs/fig_framework.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figs/fig_framework.png', dpi=300, bbox_inches='tight')
print('Saved figs/fig_framework.pdf')

print('All figures generated successfully!')
