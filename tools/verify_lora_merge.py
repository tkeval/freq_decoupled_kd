"""Verify Stage 1 LoRA checkpoint structure and merge."""
import sys
sys.path.insert(0, '.')
import torch

# 1. Load Stage 1 LoRA checkpoint
ckpt_path = './work_dirs/stage1_v2_lora_strong_fixed/epoch_12.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
sd = ckpt.get('state_dict', ckpt)

# Show top-level key prefixes
prefixes = set()
for k in sd.keys():
    parts = k.split('.')
    if len(parts) >= 2:
        prefixes.add(f'{parts[0]}.{parts[1]}')
print(f'Top-level prefixes in checkpoint: {sorted(prefixes)[:15]}')

# Extract backbone keys
backbone_keys = {k.replace('student.backbone.', ''): v
                 for k, v in sd.items() if k.startswith('student.backbone.')}
print(f'\nStage 1 backbone keys: {len(backbone_keys)}')

# Check for LoRA keys
lora_a_keys = [k for k in backbone_keys if '.lora_A.' in k]
lora_b_keys = [k for k in backbone_keys if '.lora_B.' in k]
original_keys = [k for k in backbone_keys if '.original.' in k]
regular_keys = [k for k in backbone_keys if 'lora_' not in k and '.original.' not in k]
print(f'  LoRA A keys: {len(lora_a_keys)}')
print(f'  LoRA B keys: {len(lora_b_keys)}')
print(f'  .original. keys: {len(original_keys)}')
print(f'  Regular keys: {len(regular_keys)}')

# Show sample LoRA keys
if lora_a_keys:
    print(f'\n  Sample LoRA A keys:')
    for k in sorted(lora_a_keys)[:3]:
        print(f'    {k}: shape={backbone_keys[k].shape}')
if original_keys:
    print(f'  Sample .original. keys:')
    for k in sorted(original_keys)[:3]:
        print(f'    {k}: shape={backbone_keys[k].shape}')

# Check LoRA modification magnitude on ONE layer
if lora_a_keys:
    # Pick first LoRA module
    prefix = lora_a_keys[0].replace('.lora_A.weight', '')
    A = backbone_keys[f'{prefix}.lora_A.weight']
    B = backbone_keys[f'{prefix}.lora_B.weight']
    W = backbone_keys[f'{prefix}.original.weight']
    delta = B @ A  # LoRA modification
    print(f'\n  LoRA analysis for {prefix}:')
    print(f'    W shape: {W.shape}, norm: {W.norm():.4f}')
    print(f'    A shape: {A.shape}, B shape: {B.shape}')
    print(f'    delta (B@A) norm: {delta.norm():.4f}')
    print(f'    Modification ratio: {delta.norm()/W.norm()*100:.2f}%')

# 2. Merge LoRA
from mmdet.models.utils.lora import merge_lora_state_dict
merged = merge_lora_state_dict(backbone_keys, scaling=1.0)
print(f'\nAfter merge: {len(merged)} keys')

# Check merged keys look like vanilla DINOv2 keys
sample_merged = sorted(merged.keys())[:10]
print(f'Sample merged keys:')
for k in sample_merged:
    print(f'  {k}: {merged[k].shape}')

# Verify no LoRA keys remain
remaining_lora = [k for k in merged if 'lora_' in k or '.original.' in k]
print(f'\nRemaining LoRA/original keys after merge: {len(remaining_lora)}')
if remaining_lora:
    print(f'  WARNING: {remaining_lora[:5]}')
else:
    print(f'  GOOD: All LoRA keys properly merged')

# 3. Check total modification across all layers
total_delta_norm = 0
total_orig_norm = 0
for prefix_key in lora_a_keys:
    prefix = prefix_key.replace('.lora_A.weight', '')
    A = backbone_keys[f'{prefix}.lora_A.weight']
    B = backbone_keys[f'{prefix}.lora_B.weight']
    W = backbone_keys[f'{prefix}.original.weight']
    delta = B @ A
    total_delta_norm += delta.norm().item()
    total_orig_norm += W.norm().item()

print(f'\n--- Overall LoRA Modification Summary ---')
print(f'LoRA modules: {len(lora_a_keys)}')
print(f'Total original weight norm: {total_orig_norm:.2f}')
print(f'Total LoRA delta norm: {total_delta_norm:.2f}')
print(f'Overall modification ratio: {total_delta_norm/total_orig_norm*100:.2f}%')
print(f'Average per-module ratio: {total_delta_norm/total_orig_norm*100/len(lora_a_keys):.4f}%')
