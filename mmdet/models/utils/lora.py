import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper for nn.Linear.

    Adds a low-rank decomposition (A @ B) to an existing linear layer,
    keeping the original weights frozen. Only the LoRA matrices are trained.

    Output = original(x) + (B @ A)(x) * (alpha / rank)

    Args:
        original: The original nn.Linear layer to wrap.
        rank: Rank of the LoRA decomposition.
        alpha: Scaling factor. The LoRA output is scaled by alpha/rank.
            Setting alpha=rank gives a scaling of 1.0.
        dropout: Dropout probability applied to input before LoRA path.
    """

    def __init__(self, original: nn.Linear, rank: int = 16,
                 alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return original_out + lora_out * self.scaling


def merge_lora_state_dict(state_dict: dict, scaling: float = 1.0) -> dict:
    """Merge LoRA weights into original weights in a state dict.

    Transforms LoRA checkpoint keys back to vanilla format:
        xxx.original.weight  →  xxx.weight  (with LoRA merged in)
        xxx.original.bias    →  xxx.bias
        xxx.lora_A.weight    →  (removed, folded into weight)
        xxx.lora_B.weight    →  (removed, folded into weight)

    Merge formula: W_merged = W_original + scaling * (lora_B @ lora_A)

    Args:
        state_dict: State dict potentially containing LoRA keys.
        scaling: LoRA scaling factor (alpha / rank). Default 1.0
            assumes alpha == rank (which is the default LoRA config).

    Returns:
        New state dict with LoRA weights merged and keys normalized.
        If no LoRA keys are found, returns the original dict unchanged.
    """
    # Find all LoRA module prefixes
    lora_prefixes = set()
    for key in state_dict:
        if '.lora_A.weight' in key:
            prefix = key.replace('.lora_A.weight', '')
            lora_prefixes.add(prefix)

    if not lora_prefixes:
        return state_dict  # No LoRA weights, return as-is

    merged = {}
    consumed_keys = set()

    for prefix in lora_prefixes:
        orig_w_key = f'{prefix}.original.weight'
        orig_b_key = f'{prefix}.original.bias'
        lora_a_key = f'{prefix}.lora_A.weight'
        lora_b_key = f'{prefix}.lora_B.weight'

        W = state_dict[orig_w_key]
        A = state_dict[lora_a_key]  # shape: [rank, in_features]
        B = state_dict[lora_b_key]  # shape: [out_features, rank]

        # Merge: W_merged = W + scaling * B @ A
        W_merged = W + scaling * (B @ A)

        # Remap key: xxx.original.weight → xxx.weight
        merged[f'{prefix}.weight'] = W_merged
        consumed_keys.update([orig_w_key, lora_a_key, lora_b_key])

        if orig_b_key in state_dict:
            merged[f'{prefix}.bias'] = state_dict[orig_b_key]
            consumed_keys.add(orig_b_key)

    # Copy all non-LoRA keys as-is
    for key, value in state_dict.items():
        if key not in consumed_keys:
            merged[key] = value

    print(f"[LoRA Merge] Merged {len(lora_prefixes)} LoRA modules "
          f"(scaling={scaling})")

    return merged


def inject_lora(model: nn.Module, target_modules: list,
                rank: int = 16, alpha: float = 16.0,
                dropout: float = 0.0) -> int:
    """Inject LoRA adapters into target linear layers of a model.

    Replaces matching nn.Linear modules with LoRALinear wrappers.
    The original weights are frozen; only LoRA parameters are trainable.

    Args:
        model: The model to inject LoRA into.
        target_modules: List of module name suffixes to target.
            E.g., ['attn.qkv', 'attn.proj'] to target attention layers.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout probability.

    Returns:
        Number of modules replaced.
    """
    replaced = 0
    # Collect modules to replace (can't modify during iteration)
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(target) for target in target_modules):
                replacements.append((name, module))

    for name, module in replacements:
        # Navigate to parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace with LoRA wrapper
        lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_module)
        replaced += 1

    return replaced
