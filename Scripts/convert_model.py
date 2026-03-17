#!/usr/bin/env python3
"""
PyTorch -> TBF (TinyBrain Binary Format) Model Converter

Converts PyTorch checkpoints to TinyBrain's optimized binary format with INT8 or INT4 quantization.

Usage:
    python convert_model.py --input model.pt --output model.tbf --quantize int8
    python convert_model.py --input model.pt --output model.tbf --quantize int4

References:
    - docs/tbf-format-spec.md for format specification
    - TB-007 Phase 2 implementation plan
"""

import argparse
import struct
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    num_layers: int
    hidden_dim: int
    num_heads: int
    vocab_size: int
    intermediate_dim: int = None  # FFN intermediate size (default: 4 * hidden_dim)
    max_seq_len: int = 2048
    num_kv_heads: int = None  # For GQA/MQA (defaults to num_heads for MHA)

    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.hidden_dim
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads  # Default to MHA


def write_tbf_header(f, config: ModelConfig):
    """Write TBF header matching Swift ModelWeights.save() format"""
    # Magic bytes "TBFM" (4 bytes)
    f.write(b'TBFM')

    # Version UInt32 (4 bytes, little-endian)
    f.write(struct.pack('<I', 1))

    # Config as JSON (matching Swift JSONEncoder output)
    config_dict = {
        'numLayers': config.num_layers,
        'hiddenDim': config.hidden_dim,
        'numHeads': config.num_heads,
        'numKVHeads': config.num_kv_heads,  # Add GQA support
        'vocabSize': config.vocab_size,
        'maxSeqLen': config.max_seq_len,
        'intermediateDim': config.intermediate_dim
    }
    config_json = json.dumps(config_dict).encode('utf-8')

    # Config length UInt32 (4 bytes)
    f.write(struct.pack('<I', len(config_json)))

    # Config JSON
    f.write(config_json)

    # Pad to 4KB
    header_size = 4 + 4 + 4 + len(config_json)
    padded_size = ((header_size + 4095) // 4096) * 4096
    padding = padded_size - header_size
    f.write(b'\x00' * padding)

    return padded_size


def load_pytorch_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load PyTorch checkpoint and extract state_dict

    Args:
        checkpoint_path: Path to .pt or .safetensors file

    Returns:
        state_dict: Dictionary of weight tensors
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Run: pip install torch", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    if checkpoint_path.suffix == '.safetensors':
        try:
            from safetensors import safe_open

            state_dict = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

            return state_dict
        except ImportError:
            print("Error: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
            sys.exit(1)
    else:
        # Standard PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model' in checkpoint:
                return checkpoint['model']
            else:
                # Assume the checkpoint itself is the state_dict
                return checkpoint
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")


def extract_weights(state_dict: Dict, config: ModelConfig) -> Dict:
    """
    Extract and organize weights from PyTorch state_dict

    Args:
        state_dict: Raw state dict from checkpoint
        config: Model configuration

    Returns:
        Organized weights dictionary
    """
    print("Extracting weights...")

    def to_numpy(tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy"""
        if hasattr(tensor, 'detach'):
            # Convert BFloat16 to Float32 first (numpy doesn't support bf16)
            if hasattr(tensor, 'dtype') and 'bfloat16' in str(tensor.dtype):
                tensor = tensor.float()
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    # Helper to find key in state_dict (handles different naming conventions)
    def find_key(patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            for key in state_dict.keys():
                if pattern in key:
                    return key
        return None

    weights = {}

    # 1. Extract embeddings
    embed_key = find_key(['embed_tokens.weight', 'tok_embeddings.weight', 'embeddings.weight'])
    if embed_key:
        weights['embeddings'] = to_numpy(state_dict[embed_key])
        print(f"  Embeddings: {weights['embeddings'].shape}")
    else:
        raise ValueError("Could not find embedding weights")

    # 2. Extract transformer layers
    weights['layers'] = []
    for layer_idx in range(config.num_layers):
        layer_weights = {}

        # Attention projections
        # NOTE: PyTorch stores weights as [out_features, in_features]
        # but TinyBrain expects [in_features, out_features] for matmul
        # So we need to TRANSPOSE!
        head_dim = config.hidden_dim // config.num_heads
        kv_dim = config.num_kv_heads * head_dim

        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            key = find_key([
                f'layers.{layer_idx}.self_attn.{proj}.weight',
                f'layers.{layer_idx}.attention.{proj}.weight',
                f'h.{layer_idx}.attn.{proj}.weight',
            ])
            if key:
                # Keep as [out, in] — quantize per output channel, then transpose
                w = to_numpy(state_dict[key])

                # Validate GQA shapes: K/V should be [kv_dim, hidden_dim]
                if proj in ('k_proj', 'v_proj') and w.ndim == 2:
                    expected_out = kv_dim
                    if w.shape[0] != expected_out:
                        print(f"  WARNING: Layer {layer_idx} {proj} has shape {w.shape}, "
                              f"expected ({expected_out}, {config.hidden_dim}) for GQA "
                              f"({config.num_kv_heads} KV heads, head_dim={head_dim})")
                elif proj == 'q_proj' and w.ndim == 2:
                    if w.shape[0] != config.hidden_dim:
                        print(f"  WARNING: Layer {layer_idx} {proj} has shape {w.shape}, "
                              f"expected ({config.hidden_dim}, {config.hidden_dim})")

                layer_weights[proj] = w

        # MLP projections
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            key = find_key([
                f'layers.{layer_idx}.mlp.{proj}.weight',
                f'layers.{layer_idx}.feed_forward.{proj}.weight',
                f'h.{layer_idx}.mlp.{proj}.weight',
            ])
            if key:
                # Keep as [out, in] — quantize per output channel, then transpose
                layer_weights[proj] = to_numpy(state_dict[key])

        # Handle merged gate+up projection (some models combine these)
        if 'gate_proj' not in layer_weights and 'up_proj' not in layer_weights:
            key = find_key([f'layers.{layer_idx}.mlp.fc1.weight'])
            if key:
                full_weight = to_numpy(state_dict[key])  # [out, in]
                mid = full_weight.shape[0] // 2  # Split on dim 0 (output)
                layer_weights['gate_proj'] = full_weight[:mid, :]
                layer_weights['up_proj'] = full_weight[mid:, :]

        # Layer norms
        for norm in ['input_layernorm', 'post_attention_layernorm']:
            key = find_key([
                f'layers.{layer_idx}.{norm}.weight',
                f'h.{layer_idx}.ln_{1 if "input" in norm else 2}.weight',
            ])
            if key:
                layer_weights[norm] = to_numpy(state_dict[key])

        weights['layers'].append(layer_weights)
        print(f"  Layer {layer_idx}: {len(layer_weights)} weight tensors")

    # 3. Extract final norm (RMSNorm before output projection)
    final_norm_key = find_key(['model.norm.weight', 'norm.weight'])
    if final_norm_key:
        weights['final_norm'] = to_numpy(state_dict[final_norm_key])
        print(f"  Final norm: {weights['final_norm'].shape}")
    else:
        print("  Final norm not found (model may not use pre-output normalization)")

    # 4. Extract LM head
    lm_head_key = find_key(['lm_head.weight', 'output.weight', 'head.weight'])
    if lm_head_key:
        # Keep as [vocab_size, hidden_dim] — quantize per output channel, then transpose
        weights['lm_head'] = to_numpy(state_dict[lm_head_key])
        print(f"  LM Head: {weights['lm_head'].shape}")
    else:
        # Some models tie embeddings and LM head
        print("  LM head not found, using tied embeddings")
        # Embeddings are [vocab_size, hidden_dim] — same layout as lm_head
        weights['lm_head'] = weights['embeddings'].copy()

    return weights


def quantize_int8_per_channel(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize tensor to INT8 with per-channel symmetric quantization
    """
    # Handle 1D tensors (layer norms, biases)
    if tensor.ndim == 1:
        max_val = np.abs(tensor).max()
        scale = max_val / 127.0  # INT8 range: -127 to 127

        quantized = np.round(tensor / scale).astype(np.int8)
        return quantized, np.array([scale], dtype=np.float32), np.array([0], dtype=np.int8)

    # 2D tensors: per-channel (per output channel)
    num_channels = tensor.shape[0]
    scales = np.zeros(num_channels, dtype=np.float32)
    quantized = np.zeros_like(tensor, dtype=np.int8)

    for ch in range(num_channels):
        max_val = np.abs(tensor[ch]).max()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        scales[ch] = scale
        quantized[ch] = np.round(tensor[ch] / scale).astype(np.int8)

    zero_points = np.zeros(num_channels, dtype=np.int8)  # Symmetric quantization

    return quantized, scales, zero_points


def quantize_int4_per_group(tensor: np.ndarray, group_size: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize tensor to INT4 with per-group symmetric quantization.

    Per-group quantization divides the flattened tensor into groups of
    `group_size` consecutive elements, each sharing one FP32 scale factor.
    Two INT4 values are packed into each byte (high nibble + low nibble).

    Memory savings vs FP32: 87.5% (4 bytes -> 0.5 bytes per element)
    Memory savings vs INT8: 50%   (1 byte  -> 0.5 bytes per element)

    Args:
        tensor: Float32 numpy array (1D or 2D)
        group_size: Number of elements per quantization group (default: 128)

    Returns:
        (packed_data, scales, zero_points)
        - packed_data: int8 array with 2 INT4 values per byte
        - scales: float32 array with one scale per group
        - zero_points: int8 array with one zero point per group (always 0 for symmetric)
    """
    flat = tensor.flatten().astype(np.float32)
    total_elements = len(flat)
    num_groups = (total_elements + group_size - 1) // group_size

    scales = np.zeros(num_groups, dtype=np.float32)
    zero_points = np.zeros(num_groups, dtype=np.int8)
    packed_count = (total_elements + 1) // 2
    packed_data = np.zeros(packed_count, dtype=np.uint8)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, total_elements)
        group_data = flat[start:end]

        # Symmetric quantization: map abs_max to 7 (max positive INT4)
        abs_max = np.abs(group_data).max()
        scale = abs_max / 7.0 if abs_max > 0 else 1.0
        scales[g] = scale

        # Quantize each element to [-7, 7] range
        quantized = np.clip(np.round(group_data / scale), -7, 7).astype(np.int8)

        # Pack pairs of INT4 values into bytes
        for i_in_group, q_val in enumerate(quantized):
            i = start + i_in_group
            byte_idx = i // 2
            nibble = int(q_val) & 0x0F  # 4-bit two's complement

            if i % 2 == 0:
                # Even index -> high nibble (bits 7-4)
                packed_data[byte_idx] = (nibble << 4) & 0xFF
            else:
                # Odd index -> low nibble (bits 3-0)
                packed_data[byte_idx] |= nibble & 0x0F

    # Convert to int8 view for consistency with Swift side
    packed_data = packed_data.view(np.int8)

    return packed_data, scales, zero_points


def write_tbf_format(weights: Dict, config: ModelConfig, output_path: str,
                     quantize_mode: str = 'int8', group_size: int = 128):
    """
    Write weights to TBF binary format (matches Swift ModelWeights.save() format EXACTLY)

    Args:
        weights: Organized weights dictionary
        config: Model configuration
        output_path: Output file path
        quantize_mode: 'int8', 'int4', or 'none'
        group_size: Group size for INT4 quantization (default: 128)
    """
    output_path = Path(output_path)
    print(f"\nWriting TBF format to: {output_path}")
    print(f"  Quantization: {quantize_mode}" + (f" (group_size={group_size})" if quantize_mode == 'int4' else ""))

    # Prepare quantized tensors
    quantized_weights = {}

    def prepare_tensor(name: str, tensor: np.ndarray):
        """Quantize tensor per output channel (dim 0), then transpose to [in, out] layout.

        Weights arrive as [out, in] from PyTorch. We:
        1. Quantize per dim 0 (output channels) — correct per-channel quantization
        2. Transpose quantized data to [in, out] for TinyBrain's matmul layout
        3. Store scales per output channel (applied per column in Swift dequantizer)
        """
        if quantize_mode == 'int4' and len(tensor.shape) >= 1:
            q_tensor, scales, zero_points = quantize_int4_per_group(tensor, group_size=group_size)
            # INT4 is flattened, transpose doesn't apply the same way
            # Store original shape transposed for matmul layout
            transposed_shape = tuple(reversed(tensor.shape)) if tensor.ndim == 2 else tensor.shape
            quantized_weights[name] = {
                'data': q_tensor,
                'scales': scales,
                'zero_points': zero_points,
                'shape': transposed_shape,
                'quantized': True,
                'precision': 'int4',
                'group_size': group_size,
            }
        elif quantize_mode == 'int8' and tensor.ndim == 2:
            # Quantize per output channel (dim 0 of [out, in])
            q_tensor, scales, zero_points = quantize_int8_per_channel(tensor)
            # Transpose quantized data: [out, in] -> [in, out]
            q_transposed = np.ascontiguousarray(q_tensor.T)
            quantized_weights[name] = {
                'data': q_transposed,
                'scales': scales,
                'zero_points': zero_points,
                'shape': q_transposed.shape,  # [in, out]
                'quantized': True,
                'precision': 'int8',
            }
        elif quantize_mode == 'int8' and len(tensor.shape) >= 1:
            # 1D tensors (norms): no transpose needed
            q_tensor, scales, zero_points = quantize_int8_per_channel(tensor)
            quantized_weights[name] = {
                'data': q_tensor,
                'scales': scales,
                'zero_points': zero_points,
                'shape': q_tensor.shape,
                'quantized': True,
                'precision': 'int8',
            }
        else:
            quantized_weights[name] = {
                'data': tensor.astype(np.float32),
                'scales': [],
                'zero_points': None,
                'shape': tensor.shape,
                'quantized': False,
            }

    print("Quantizing weights...")
    # Embeddings (Float32, not quantized)
    quantized_weights['embeddings'] = {
        'data': weights['embeddings'].astype(np.float32),
        'scales': [],
        'zero_points': None,
        'shape': weights['embeddings'].shape,
        'quantized': False
    }

    # Layers (quantized)
    pbar = tqdm(total=len(weights['layers']) * 9, desc="Quantizing")
    for layer_idx, layer in enumerate(weights['layers']):
        # Attention projections
        prepare_tensor(f'layer_{layer_idx}_attn_q', layer['q_proj'])
        pbar.update(1)
        prepare_tensor(f'layer_{layer_idx}_attn_k', layer['k_proj'])
        pbar.update(1)
        prepare_tensor(f'layer_{layer_idx}_attn_v', layer['v_proj'])
        pbar.update(1)
        prepare_tensor(f'layer_{layer_idx}_attn_o', layer['o_proj'])
        pbar.update(1)

        # FFN projections (gated FFN: gate_proj, up_proj, down_proj)
        if 'gate_proj' in layer:
            prepare_tensor(f'layer_{layer_idx}_ffn_gate', layer['gate_proj'])
        pbar.update(1)
        prepare_tensor(f'layer_{layer_idx}_ffn_up', layer['up_proj'])
        pbar.update(1)
        prepare_tensor(f'layer_{layer_idx}_ffn_down', layer['down_proj'])
        pbar.update(1)

        # Layer norms (Float32, not quantized for numerical stability)
        quantized_weights[f'layer_{layer_idx}_ln_input'] = {
            'data': layer['input_layernorm'].astype(np.float32),
            'scales': [],
            'zero_points': None,
            'shape': layer['input_layernorm'].shape,
            'quantized': False
        }
        pbar.update(1)
        quantized_weights[f'layer_{layer_idx}_ln_post'] = {
            'data': layer['post_attention_layernorm'].astype(np.float32),
            'scales': [],
            'zero_points': None,
            'shape': layer['post_attention_layernorm'].shape,
            'quantized': False
        }
        pbar.update(1)
    pbar.close()

    # Final norm (Float32, not quantized for numerical stability)
    if 'final_norm' in weights:
        quantized_weights['final_norm'] = {
            'data': weights['final_norm'].astype(np.float32),
            'scales': [],
            'zero_points': None,
            'shape': weights['final_norm'].shape,
            'quantized': False
        }

    # Output projection
    prepare_tensor('output', weights['lm_head'])

    print(f"Writing to: {output_path}")

    with open(output_path, 'wb') as f:
        # 1. Write header
        header_end = write_tbf_header(f, config)

        # 2. Write quantization metadata section (starts at 4KB boundary)
        # Count quantized tensors only
        quant_tensor_names = [name for name, info in quantized_weights.items() if info['quantized']]
        f.write(struct.pack('<I', len(quant_tensor_names)))

        for name in quant_tensor_names:
            info = quantized_weights[name]

            # Tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            # Precision (1 = INT8, 2 = INT4)
            precision_code = 2 if info.get('precision') == 'int4' else 1
            f.write(struct.pack('<B', precision_code))

            # Mode (2 = perChannel for INT8, 3 = int4 per-group for INT4)
            mode_code = 3 if info.get('precision') == 'int4' else 2
            f.write(struct.pack('<B', mode_code))

            # Group size (only meaningful for INT4, 0 for INT8)
            gs = info.get('group_size', 0) if info.get('precision') == 'int4' else 0
            f.write(struct.pack('<I', gs))

            # Scales
            f.write(struct.pack('<I', len(info['scales'])))
            for scale in info['scales']:
                f.write(struct.pack('<f', scale))

            # Zero points
            if info['zero_points'] is not None:
                f.write(struct.pack('<I', len(info['zero_points'])))
                for zp in info['zero_points']:
                    f.write(struct.pack('<b', zp))
            else:
                f.write(struct.pack('<I', 0))

        # Pad to 4KB
        current_pos = f.tell()
        padded_pos = ((current_pos + 4095) // 4096) * 4096
        f.write(b'\x00' * (padded_pos - current_pos))

        # 3. Write tensor index section (two-pass: placeholders then patch)
        f.write(struct.pack('<I', len(quantized_weights)))

        tensor_list = list(quantized_weights.items())

        # Pass 1: write index entries with placeholder offsets, record positions to patch
        offset_positions = []  # (file_pos, data_size) for each tensor
        for name, info in tensor_list:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            f.write(struct.pack('<I', len(info['shape'])))
            for dim in info['shape']:
                f.write(struct.pack('<i', dim))

            data_size = info['data'].nbytes
            offset_positions.append((f.tell(), data_size))
            f.write(struct.pack('<Q', 0))  # placeholder offset
            f.write(struct.pack('<Q', data_size))

        # Pad index to 4KB
        current_pos = f.tell()
        padded_pos = ((current_pos + 4095) // 4096) * 4096
        f.write(b'\x00' * (padded_pos - current_pos))

        # Now we know the actual data_start
        data_start = padded_pos

        # Pass 2: patch the offset placeholders with correct absolute offsets
        current_data_offset = 0
        for file_pos, data_size in offset_positions:
            f.seek(file_pos)
            f.write(struct.pack('<Q', data_start + current_data_offset))
            padded_size = ((data_size + 4095) // 4096) * 4096
            current_data_offset += padded_size

        # Seek back to data section start
        f.seek(data_start)

        # 4. Write weight data blobs (4KB aligned per tensor)
        pbar = tqdm(total=len(tensor_list), desc="Writing weights")
        for name, info in tensor_list:
            f.write(info['data'].tobytes())

            # Pad to 4KB
            current_pos = f.tell()
            padded_pos = ((current_pos + 4095) // 4096) * 4096
            f.write(b'\x00' * (padded_pos - current_pos))

            pbar.update(1)
        pbar.close()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nConversion complete! Output: {output_path} ({file_size_mb:.2f} MB)")


def infer_config_from_weights(state_dict: Dict, checkpoint_path: str = None) -> ModelConfig:
    """
    Attempt to infer model configuration from weight shapes or config.json
    """
    # First, try to load config.json if checkpoint path is provided
    if checkpoint_path:
        checkpoint_dir = Path(checkpoint_path).parent
        config_json_path = checkpoint_dir / 'config.json'

        if config_json_path.exists():
            print(f"Loading config from: {config_json_path}")
            with open(config_json_path) as f:
                hf_config = json.load(f)

            vocab_size = hf_config.get('vocab_size')
            hidden_dim = hf_config.get('hidden_size')
            num_layers = hf_config.get('num_hidden_layers')
            num_heads = hf_config.get('num_attention_heads')
            num_kv_heads = hf_config.get('num_key_value_heads', num_heads)  # GQA support!
            intermediate_dim = hf_config.get('intermediate_size', 4 * hidden_dim)
            max_seq_len = hf_config.get('max_position_embeddings', 2048)

            print("\nInferred configuration:")
            print(f"  Vocab size: {vocab_size}")
            print(f"  Hidden dim: {hidden_dim}")
            print(f"  Num layers: {num_layers}")
            print(f"  Num heads: {num_heads}")
            print(f"  Num KV heads: {num_kv_heads}")
            print(f"  Intermediate dim: {intermediate_dim}")

            return ModelConfig(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                vocab_size=vocab_size,
                intermediate_dim=intermediate_dim,
                max_seq_len=max_seq_len
            )

    # Fallback: infer from weight shapes
    embed_key = None
    for key in state_dict.keys():
        if 'embed' in key and 'weight' in key:
            embed_key = key
            break

    if embed_key is None:
        raise ValueError("Could not find embedding weights to infer config")

    embed_shape = state_dict[embed_key].shape
    vocab_size, hidden_dim = embed_shape

    # Count number of layers
    num_layers = 0
    while True:
        if any(f'layers.{num_layers}.' in key or f'h.{num_layers}.' in key for key in state_dict.keys()):
            num_layers += 1
        else:
            break

    # Get num_heads from attention projection shapes
    num_heads = 8  # Default guess
    if hidden_dim % 128 == 0:
        num_heads = hidden_dim // 128
    elif hidden_dim % 64 == 0:
        num_heads = hidden_dim // 64

    # Try to infer num_kv_heads from K/V projection shapes
    num_kv_heads = num_heads  # Default to MHA
    for key in state_dict.keys():
        if 'layers.0' in key and ('k_proj' in key or 'key' in key):
            if len(state_dict[key].shape) == 2:
                kv_out_dim = state_dict[key].shape[0]  # Before transpose
                head_dim = hidden_dim // num_heads
                num_kv_heads = kv_out_dim // head_dim
                break

    # Get intermediate_dim from MLP
    intermediate_dim = 4 * hidden_dim  # Default
    for key in state_dict.keys():
        if 'mlp' in key and ('gate_proj' in key or 'up_proj' in key or 'fc1' in key):
            if len(state_dict[key].shape) == 2:
                intermediate_dim = state_dict[key].shape[0]
                break

    print("\nInferred configuration:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num KV heads: {num_kv_heads}")
    print(f"  Intermediate dim: {intermediate_dim}")

    return ModelConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        intermediate_dim=intermediate_dim,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoints to TinyBrain Binary Format (TBF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with INT8 quantization (75% memory savings)
  python convert_model.py --input tinyllama.pt --output tinyllama-int8.tbf --quantize int8

  # Convert with INT4 quantization (87.5% memory savings)
  python convert_model.py --input tinyllama.pt --output tinyllama-int4.tbf --quantize int4

  # INT4 with custom group size
  python convert_model.py --input model.pt --output model-int4.tbf --quantize int4 --group-size 64

  # Convert without quantization (FP32)
  python convert_model.py --input model.pt --output model.tbf --quantize none

  # Auto-infer configuration
  python convert_model.py --input model.pt --output model.tbf --auto-config
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input PyTorch checkpoint (.pt or .safetensors)')
    parser.add_argument('--output', '-o', required=True, help='Output TBF file path')
    parser.add_argument('--quantize', choices=['int8', 'int4', 'none'], default='int8',
                        help='Quantization mode (int8: 75%% savings, int4: 87.5%% savings)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Group size for INT4 quantization (default: 128)')
    parser.add_argument('--auto-config', action='store_true', help='Auto-infer model config from weights')

    # Manual config overrides
    parser.add_argument('--num-layers', type=int, help='Number of transformer layers')
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension size')
    parser.add_argument('--num-heads', type=int, help='Number of attention heads')
    parser.add_argument('--num-kv-heads', type=int, help='Number of KV heads for GQA (defaults to num-heads)')
    parser.add_argument('--vocab-size', type=int, help='Vocabulary size')

    args = parser.parse_args()

    # Load checkpoint
    state_dict = load_pytorch_checkpoint(args.input)

    # Determine config
    if args.auto_config or not all([args.num_layers, args.hidden_dim, args.num_heads, args.vocab_size]):
        config = infer_config_from_weights(state_dict, checkpoint_path=args.input)
    else:
        config = ModelConfig(
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_kv_heads=getattr(args, 'num_kv_heads', None),
            vocab_size=args.vocab_size,
        )

    # Extract weights
    weights = extract_weights(state_dict, config)

    # Write TBF format
    write_tbf_format(weights, config, args.output,
                     quantize_mode=args.quantize,
                     group_size=args.group_size)


if __name__ == '__main__':
    main()
