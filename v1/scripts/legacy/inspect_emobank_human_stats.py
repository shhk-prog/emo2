#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("v1/results/recognition_baseline/qwen2.5_1.5b_base_emobank.csv")
print("=== EmoBank Stimuli Human Annotation Stats (N = 321) ===")
print("human_v (Valence):")
print(f"  Mean:   {df['human_v'].mean():.4f}")
print(f"  Std:    {df['human_v'].std():.4f}")
print(f"  Median: {df['human_v'].median():.4f}")
print(f"  Min:    {df['human_v'].min():.4f}")
print(f"  Max:    {df['human_v'].max():.4f}")

print("\nhuman_a (Arousal):")
print(f"  Mean:   {df['human_a'].mean():.4f}")
print(f"  Std:    {df['human_a'].std():.4f}")
print(f"  Median: {df['human_a'].median():.4f}")
print(f"  Min:    {df['human_a'].min():.4f}")
print(f"  Max:    {df['human_a'].max():.4f}")

# Check original EmoBank scaling
# EmoBank original is 1-5 scale:
# Valence: 1 (sad) to 5 (happy), 3 is neutral
# Arousal: 1 (calm) to 5 (excited), 3 is neutral
# But our model outputs in 1-9 scale:
# 1 (sad) to 9 (happy), 5 is neutral.
# Let's check human_v_scaled as well!
if 'human_v_scaled' in df.columns:
    print("\nhuman_v_scaled (typically [-1, 1]):")
    print(f"  Mean: {df['human_v_scaled'].mean():.4f}")
    print(f"  Min:  {df['human_v_scaled'].min():.4f}")
    print(f"  Max:  {df['human_v_scaled'].max():.4f}")
