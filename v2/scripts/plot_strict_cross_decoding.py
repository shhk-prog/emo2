import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 5)

df = pd.read_csv("v2/results/derived/phase3_strict_cross_decoding/strict_cross_decoding.csv")
out_dir = "v2/results/derived/phase3_strict_cross_decoding/plots"
os.makedirs(out_dir, exist_ok=True)

# 1. R2 Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for i, direction in enumerate(['Base2Inst', 'Inst2Base']):
    d = df[df['direction'] == direction]
    axes[i].plot(d['layer'], d['r2_direct'], label='Direct', marker='o')
    axes[i].plot(d['layer'], d['r2_ortho'], label='Ortho Procrustes', marker='s')
    axes[i].plot(d['layer'], d['r2_ridge'], label='RidgeCV', marker='^')
    axes[i].set_title(f'Representational Recovery (R²) : {direction}')
    axes[i].set_xlabel('Layer')
    axes[i].set_ylabel('Variance Weighted R²')
    axes[i].legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'strict_r2.png'), dpi=300)
plt.close()

# 2. IBC (Valence)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for i, direction in enumerate(['Base2Inst', 'Inst2Base']):
    d = df[df['direction'] == direction]
    axes[i].plot(d['layer'], d['UB_V'], label='Target Model (Upper Bound)', color='black', linestyle='--')
    axes[i].plot(d['layer'], d['ibc_V_direct'], label='Direct', marker='o')
    axes[i].plot(d['layer'], d['ibc_V_ortho'], label='Ortho Procrustes', marker='s')
    axes[i].plot(d['layer'], d['ibc_V_ridge'], label='RidgeCV', marker='^')
    axes[i].set_title(f'Valence IBC Transfer : {direction}')
    axes[i].set_xlabel('Layer')
    axes[i].set_ylabel('Pearson r (IBC_V)')
    axes[i].legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'strict_ibc_v.png'), dpi=300)
plt.close()

# 3. Control Transfer (Word Count)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for i, direction in enumerate(['Base2Inst', 'Inst2Base']):
    d = df[df['direction'] == direction]
    axes[i].plot(d['layer'], d['UB_WC'], label='Target Model (Upper Bound)', color='black', linestyle='--')
    axes[i].plot(d['layer'], d['ibc_WC_direct'], label='Direct', marker='o')
    axes[i].plot(d['layer'], d['ibc_WC_ortho'], label='Ortho Procrustes', marker='s')
    axes[i].plot(d['layer'], d['ibc_WC_ridge'], label='RidgeCV', marker='^')
    axes[i].set_title(f'Word Count IBC Transfer : {direction}')
    axes[i].set_xlabel('Layer')
    axes[i].set_ylabel('Pearson r (Word Count)')
    axes[i].legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'strict_ibc_wc.png'), dpi=300)
plt.close()

