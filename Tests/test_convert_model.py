"""
Test suite for PyTorch → TBF model converter

TDD RED phase: These tests define the expected behavior before implementation.
"""

import pytest
import numpy as np
import os
import sys
import tempfile
import json
import struct
from pathlib import Path

# Add Scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "Scripts"))

try:
    from convert_model import (
        load_pytorch_checkpoint,
        extract_weights,
        quantize_int8_per_channel,
        quantize_int4_per_group,
        write_tbf_format,
        ModelConfig,
        infer_config_from_weights,
    )
except ImportError:
    # Tests will fail until we implement convert_model.py
    pytest.skip("convert_model.py not implemented yet", allow_module_level=True)


def _align_4kb(offset):
    return ((offset + 4095) // 4096) * 4096


def _read_tbf_config(path):
    """Read the JSON config from a converter-written TBF header."""
    with open(path, 'rb') as f:
        assert f.read(4) == b'TBFM'
        version = struct.unpack('<I', f.read(4))[0]
        assert version == 1
        config_len = struct.unpack('<I', f.read(4))[0]
        return json.loads(f.read(config_len))


def _read_tbf_tensor_names(path):
    """Return tensor names from the TBF tensor index."""
    with open(path, 'rb') as f:
        blob = f.read()

    offset = 0
    assert blob[offset:offset + 4] == b'TBFM'
    offset += 4
    _version = struct.unpack_from('<I', blob, offset)[0]
    offset += 4
    config_len = struct.unpack_from('<I', blob, offset)[0]
    offset += 4 + config_len
    offset = _align_4kb(offset)

    metadata_count = struct.unpack_from('<I', blob, offset)[0]
    offset += 4
    for _ in range(metadata_count):
        name_len = struct.unpack_from('<I', blob, offset)[0]
        offset += 4 + name_len
        offset += 1  # precision
        offset += 1  # mode
        offset += 4  # group size
        scales_count = struct.unpack_from('<I', blob, offset)[0]
        offset += 4 + scales_count * 4
        zp_count = struct.unpack_from('<I', blob, offset)[0]
        offset += 4 + zp_count

    offset = _align_4kb(offset)
    tensor_count = struct.unpack_from('<I', blob, offset)[0]
    offset += 4

    names = []
    for _ in range(tensor_count):
        name_len = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        name = blob[offset:offset + name_len].decode('utf-8')
        offset += name_len
        names.append(name)
        dim_count = struct.unpack_from('<I', blob, offset)[0]
        offset += 4 + dim_count * 4
        offset += 8  # data offset
        offset += 8  # data size

    return names


def _read_tbf_quantized_tensor(path, tensor_name):
    """Read one quantized tensor from a converter-written TBF file."""
    with open(path, 'rb') as f:
        blob = f.read()

    offset = 0
    assert blob[offset:offset + 4] == b'TBFM'
    offset += 4
    _version = struct.unpack_from('<I', blob, offset)[0]
    offset += 4
    config_len = struct.unpack_from('<I', blob, offset)[0]
    offset += 4 + config_len
    offset = _align_4kb(offset)

    metadata = {}
    metadata_count = struct.unpack_from('<I', blob, offset)[0]
    offset += 4
    for _ in range(metadata_count):
        name_len = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        name = blob[offset:offset + name_len].decode('utf-8')
        offset += name_len
        precision = struct.unpack_from('<B', blob, offset)[0]
        offset += 1
        mode = struct.unpack_from('<B', blob, offset)[0]
        offset += 1
        group_size = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        scales_count = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        scales = np.array(
            struct.unpack_from(f'<{scales_count}f', blob, offset),
            dtype=np.float32,
        )
        offset += scales_count * 4
        zp_count = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        zero_points = np.array(
            struct.unpack_from(f'<{zp_count}b', blob, offset),
            dtype=np.int8,
        ) if zp_count else None
        offset += zp_count
        metadata[name] = {
            'precision': precision,
            'mode': mode,
            'group_size': group_size,
            'scales': scales,
            'zero_points': zero_points,
        }

    offset = _align_4kb(offset)
    tensor_index = {}
    tensor_count = struct.unpack_from('<I', blob, offset)[0]
    offset += 4
    for _ in range(tensor_count):
        name_len = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        name = blob[offset:offset + name_len].decode('utf-8')
        offset += name_len
        dim_count = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        shape = struct.unpack_from(f'<{dim_count}i', blob, offset)
        offset += dim_count * 4
        data_offset = struct.unpack_from('<Q', blob, offset)[0]
        offset += 8
        data_size = struct.unpack_from('<Q', blob, offset)[0]
        offset += 8
        tensor_index[name] = {
            'shape': tuple(shape),
            'data_offset': data_offset,
            'data_size': data_size,
        }

    index = tensor_index[tensor_name]
    raw = blob[index['data_offset']:index['data_offset'] + index['data_size']]
    return {
        **metadata[tensor_name],
        'shape': index['shape'],
        'data': np.frombuffer(raw, dtype=np.int8).copy(),
    }


def _dequantize_int4_swift_order(packed, shape, scales, zero_points, group_size):
    """Mirror QuantizedTensor.dequantize() for mode .int4."""
    flat_count = int(np.prod(shape))
    output = np.zeros(flat_count, dtype=np.float32)
    packed_u8 = packed.view(np.uint8)
    gs = group_size if group_size > 0 else flat_count
    zps = zero_points if zero_points is not None else np.zeros_like(scales, dtype=np.int8)

    for i in range(flat_count):
        byte = int(packed_u8[i // 2])
        raw = (byte >> 4) & 0x0F if i % 2 == 0 else byte & 0x0F
        int4_val = raw - 16 if raw > 7 else raw
        group_idx = i // gs
        output[i] = (int4_val - int(zps[group_idx])) * scales[group_idx]

    return output.reshape(shape)


def _dequantize_int8_swift_order(q_data, shape, scales, zero_points):
    """Mirror QuantizedTensor.dequantize() for 2D INT8 per-channel tensors."""
    rows, cols = shape
    q = q_data.reshape(shape).astype(np.float32)
    zps = zero_points if zero_points is not None else np.zeros_like(scales, dtype=np.int8)
    output = np.zeros(shape, dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            output[row, col] = (q[row, col] - int(zps[col])) * scales[col]
    return output


class TestModelLoading:
    """Test PyTorch checkpoint loading"""
    
    def test_load_pytorch_checkpoint_dict(self):
        """Should load a .pt file and extract state_dict"""
        # Create a minimal fake checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
            
        try:
            import torch
            fake_checkpoint = {
                'model_state_dict': {
                    'embed_tokens.weight': torch.randn(1000, 256),
                    'layers.0.self_attn.q_proj.weight': torch.randn(256, 256),
                }
            }
            torch.save(fake_checkpoint, checkpoint_path)
            
            # Test loading
            state_dict = load_pytorch_checkpoint(checkpoint_path)
            
            assert 'embed_tokens.weight' in state_dict
            assert 'layers.0.self_attn.q_proj.weight' in state_dict
            assert state_dict['embed_tokens.weight'].shape == (1000, 256)
            
        finally:
            os.unlink(checkpoint_path)
    
    def test_load_safetensors_checkpoint(self):
        """Should load a .safetensors file"""
        pytest.skip("SafeTensors support can be added later")


class TestWeightExtraction:
    """Test weight extraction and shape validation"""
    
    def test_extract_weights_shapes(self):
        """Should extract weights and validate tensor dimensions"""
        import torch
        
        # Fake state dict for a tiny transformer
        state_dict = {
            'model.embed_tokens.weight': torch.randn(1000, 128),  # vocab=1000, dim=128
            'model.layers.0.self_attn.q_proj.weight': torch.randn(128, 128),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(128, 128),
            'model.layers.0.self_attn.v_proj.weight': torch.randn(128, 128),
            'model.layers.0.self_attn.o_proj.weight': torch.randn(128, 128),
            'model.layers.0.mlp.gate_proj.weight': torch.randn(512, 128),
            'model.layers.0.mlp.down_proj.weight': torch.randn(128, 512),
            'model.layers.0.input_layernorm.weight': torch.randn(128),
            'model.layers.0.post_attention_layernorm.weight': torch.randn(128),
            'lm_head.weight': torch.randn(1000, 128),
        }
        
        config = ModelConfig(
            num_layers=1,
            hidden_dim=128,
            num_heads=4,
            vocab_size=1000,
            intermediate_dim=512,
        )
        
        weights = extract_weights(state_dict, config)
        
        # Validate structure
        assert 'embeddings' in weights
        assert 'layers' in weights
        assert len(weights['layers']) == 1
        assert 'lm_head' in weights
        
        # Validate shapes
        assert weights['embeddings'].shape == (1000, 128)
        assert weights['layers'][0]['q_proj'].shape == (128, 128)
        assert weights['lm_head'].shape == (1000, 128)


class TestGQAWeightExtraction:
    """Test GQA (Grouped Query Attention) weight extraction and shape validation"""

    def test_gqa_kv_projection_shapes(self):
        """K/V projections must have shape [num_kv_heads * head_dim, hidden_dim] for GQA"""
        import torch

        hidden_dim = 2048
        num_heads = 32
        num_kv_heads = 4
        head_dim = hidden_dim // num_heads  # 64
        kv_dim = num_kv_heads * head_dim    # 256

        # Simulate TinyLlama GQA weight shapes
        state_dict = {
            'model.embed_tokens.weight': torch.randn(32000, hidden_dim),
            'model.layers.0.self_attn.q_proj.weight': torch.randn(hidden_dim, hidden_dim),  # [2048, 2048]
            'model.layers.0.self_attn.k_proj.weight': torch.randn(kv_dim, hidden_dim),      # [256, 2048]
            'model.layers.0.self_attn.v_proj.weight': torch.randn(kv_dim, hidden_dim),      # [256, 2048]
            'model.layers.0.self_attn.o_proj.weight': torch.randn(hidden_dim, hidden_dim),  # [2048, 2048]
            'model.layers.0.mlp.gate_proj.weight': torch.randn(5632, hidden_dim),
            'model.layers.0.mlp.up_proj.weight': torch.randn(5632, hidden_dim),
            'model.layers.0.mlp.down_proj.weight': torch.randn(hidden_dim, 5632),
            'model.layers.0.input_layernorm.weight': torch.randn(hidden_dim),
            'model.layers.0.post_attention_layernorm.weight': torch.randn(hidden_dim),
            'lm_head.weight': torch.randn(32000, hidden_dim),
        }

        config = ModelConfig(
            num_layers=1,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=32000,
            intermediate_dim=5632,
        )

        weights = extract_weights(state_dict, config)

        # Q and O should be [hidden_dim, hidden_dim] = [2048, 2048]
        assert weights['layers'][0]['q_proj'].shape == (hidden_dim, hidden_dim), \
            f"Q proj shape wrong: {weights['layers'][0]['q_proj'].shape}"
        assert weights['layers'][0]['o_proj'].shape == (hidden_dim, hidden_dim), \
            f"O proj shape wrong: {weights['layers'][0]['o_proj'].shape}"

        # K and V should be [kv_dim, hidden_dim] = [256, 2048] for GQA
        assert weights['layers'][0]['k_proj'].shape == (kv_dim, hidden_dim), \
            f"K proj shape wrong: {weights['layers'][0]['k_proj'].shape}, expected ({kv_dim}, {hidden_dim})"
        assert weights['layers'][0]['v_proj'].shape == (kv_dim, hidden_dim), \
            f"V proj shape wrong: {weights['layers'][0]['v_proj'].shape}, expected ({kv_dim}, {hidden_dim})"

    def test_gqa_conversion_roundtrip(self):
        """Full GQA conversion: extract → quantize → write TBF → verify shapes in file"""
        import torch
        import struct

        hidden_dim = 128
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads  # 16
        kv_dim = num_kv_heads * head_dim    # 32
        intermediate_dim = 256
        vocab_size = 100

        state_dict = {
            'model.embed_tokens.weight': torch.randn(vocab_size, hidden_dim),
            'model.layers.0.self_attn.q_proj.weight': torch.randn(hidden_dim, hidden_dim),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(kv_dim, hidden_dim),
            'model.layers.0.self_attn.v_proj.weight': torch.randn(kv_dim, hidden_dim),
            'model.layers.0.self_attn.o_proj.weight': torch.randn(hidden_dim, hidden_dim),
            'model.layers.0.mlp.gate_proj.weight': torch.randn(intermediate_dim, hidden_dim),
            'model.layers.0.mlp.up_proj.weight': torch.randn(intermediate_dim, hidden_dim),
            'model.layers.0.mlp.down_proj.weight': torch.randn(hidden_dim, intermediate_dim),
            'model.layers.0.input_layernorm.weight': torch.randn(hidden_dim),
            'model.layers.0.post_attention_layernorm.weight': torch.randn(hidden_dim),
            'model.norm.weight': torch.randn(hidden_dim),
            'lm_head.weight': torch.randn(vocab_size, hidden_dim),
        }

        config = ModelConfig(
            num_layers=1,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            intermediate_dim=intermediate_dim,
        )

        weights = extract_weights(state_dict, config)

        with tempfile.NamedTemporaryFile(suffix='.tbf', delete=False) as f:
            output_path = f.name

        try:
            write_tbf_format(weights, config, output_path, quantize_mode='int8')

            # Read and verify the TBF file
            with open(output_path, 'rb') as f:
                magic = f.read(4)
                assert magic == b'TBFM'

                version = struct.unpack('<I', f.read(4))[0]
                assert version == 1

                config_len = struct.unpack('<I', f.read(4))[0]
                config_json = json.loads(f.read(config_len))

                # Verify GQA config is preserved
                assert config_json['numKVHeads'] == num_kv_heads, \
                    f"numKVHeads wrong: {config_json['numKVHeads']}, expected {num_kv_heads}"
                assert config_json['numHeads'] == num_heads

            # Verify file is non-empty and reasonable size
            file_size = os.path.getsize(output_path)
            assert file_size > 4096, f"File too small: {file_size}"

        finally:
            os.unlink(output_path)

    def test_gqa_quantized_kv_shapes(self):
        """Quantized K/V tensors must have transposed GQA shapes [hidden_dim, kv_dim]"""
        import torch

        hidden_dim = 128
        num_heads = 8
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads  # 16
        kv_dim = num_kv_heads * head_dim    # 32

        # Simulate K projection: PyTorch [kv_dim, hidden_dim] = [32, 128]
        k_weight = np.random.randn(kv_dim, hidden_dim).astype(np.float32)

        # Quantize per output channel (dim 0)
        quantized, scales, zero_points = quantize_int8_per_channel(k_weight)

        assert quantized.shape == (kv_dim, hidden_dim), \
            f"Quantized shape wrong: {quantized.shape}"
        assert len(scales) == kv_dim, \
            f"Should have {kv_dim} scales (one per KV output channel), got {len(scales)}"

        # After transpose (as converter does): [hidden_dim, kv_dim]
        transposed = np.ascontiguousarray(quantized.T)
        assert transposed.shape == (hidden_dim, kv_dim), \
            f"Transposed shape wrong: {transposed.shape}"

    def test_infer_config_gqa_from_config_json(self):
        """Config inference from config.json should correctly detect GQA"""
        import torch

        hidden_dim = 2048
        num_heads = 32
        num_kv_heads = 4
        head_dim = hidden_dim // num_heads  # 64
        kv_dim = num_kv_heads * head_dim    # 256

        state_dict = {
            'model.embed_tokens.weight': torch.randn(32000, hidden_dim),
            'model.layers.0.self_attn.q_proj.weight': torch.randn(hidden_dim, hidden_dim),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(kv_dim, hidden_dim),
            'model.layers.0.self_attn.v_proj.weight': torch.randn(kv_dim, hidden_dim),
            'model.layers.0.self_attn.o_proj.weight': torch.randn(hidden_dim, hidden_dim),
            'model.layers.0.mlp.gate_proj.weight': torch.randn(5632, hidden_dim),
            'model.layers.0.mlp.up_proj.weight': torch.randn(5632, hidden_dim),
            'model.layers.0.mlp.down_proj.weight': torch.randn(hidden_dim, 5632),
            'model.layers.0.input_layernorm.weight': torch.randn(hidden_dim),
            'model.layers.0.post_attention_layernorm.weight': torch.randn(hidden_dim),
            'lm_head.weight': torch.randn(32000, hidden_dim),
        }

        # Create a temporary config.json (simulating HuggingFace model dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            config_json = {
                'vocab_size': 32000,
                'hidden_size': hidden_dim,
                'num_hidden_layers': 1,
                'num_attention_heads': num_heads,
                'num_key_value_heads': num_kv_heads,
                'intermediate_size': 5632,
                'max_position_embeddings': 2048,
            }
            config_path = os.path.join(tmpdir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_json, f)

            # Point checkpoint_path to the dir so config.json is found
            fake_checkpoint = os.path.join(tmpdir, 'model.safetensors')
            with open(fake_checkpoint, 'w') as f:
                f.write('')  # dummy file

            config = infer_config_from_weights(state_dict, checkpoint_path=fake_checkpoint)

        assert config.num_kv_heads == num_kv_heads, \
            f"Inferred num_kv_heads={config.num_kv_heads}, expected {num_kv_heads}"
        assert config.num_heads == num_heads
        assert config.hidden_dim == hidden_dim

    def test_qwen2_auto_config_reads_rope_norm_and_qkv_biases(self):
        """Qwen2 auto-config must preserve theta/epsilon and q/k/v biases."""
        import torch

        hidden_dim = 24
        num_heads = 12
        num_kv_heads = 2
        head_dim = hidden_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        intermediate_dim = 48
        vocab_size = 64

        state_dict = {
            'model.embed_tokens.weight': torch.randn(vocab_size, hidden_dim),
            'model.layers.0.self_attn.q_proj.weight': torch.randn(hidden_dim, hidden_dim),
            'model.layers.0.self_attn.q_proj.bias': torch.randn(hidden_dim),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(kv_dim, hidden_dim),
            'model.layers.0.self_attn.k_proj.bias': torch.randn(kv_dim),
            'model.layers.0.self_attn.v_proj.weight': torch.randn(kv_dim, hidden_dim),
            'model.layers.0.self_attn.v_proj.bias': torch.randn(kv_dim),
            'model.layers.0.self_attn.o_proj.weight': torch.randn(hidden_dim, hidden_dim),
            'model.layers.0.mlp.gate_proj.weight': torch.randn(intermediate_dim, hidden_dim),
            'model.layers.0.mlp.up_proj.weight': torch.randn(intermediate_dim, hidden_dim),
            'model.layers.0.mlp.down_proj.weight': torch.randn(hidden_dim, intermediate_dim),
            'model.layers.0.input_layernorm.weight': torch.randn(hidden_dim),
            'model.layers.0.post_attention_layernorm.weight': torch.randn(hidden_dim),
            'model.norm.weight': torch.randn(hidden_dim),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_json = {
                'model_type': 'qwen2',
                'vocab_size': vocab_size,
                'hidden_size': hidden_dim,
                'num_hidden_layers': 1,
                'num_attention_heads': num_heads,
                'num_key_value_heads': num_kv_heads,
                'intermediate_size': intermediate_dim,
                'max_position_embeddings': 32768,
                'rope_theta': 1000000.0,
                'rms_norm_eps': 1e-6,
                'tie_word_embeddings': True,
            }
            config_path = os.path.join(tmpdir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_json, f)

            fake_checkpoint = os.path.join(tmpdir, 'model.safetensors')
            with open(fake_checkpoint, 'w') as f:
                f.write('')

            config = infer_config_from_weights(state_dict, checkpoint_path=fake_checkpoint)
            assert config.architecture == 'llama'
            assert config.num_kv_heads == 2
            assert config.max_seq_len == 32768
            assert config.rope_theta == pytest.approx(1000000.0)
            assert config.rms_norm_epsilon == pytest.approx(1e-6)

            weights = extract_weights(state_dict, config)
            layer = weights['layers'][0]
            assert layer['q_proj_bias'].shape == (hidden_dim,)
            assert layer['k_proj_bias'].shape == (kv_dim,)
            assert layer['v_proj_bias'].shape == (kv_dim,)
            assert 'o_proj_bias' not in layer

            output_path = os.path.join(tmpdir, 'qwen2-like.tbf')
            write_tbf_format(weights, config, output_path, quantize_mode='int8')
            header = _read_tbf_config(output_path)
            assert header['architecture'] == 'llama'
            assert header['numKVHeads'] == 2
            assert header['maxSeqLen'] == 32768
            assert header['ropeTheta'] == pytest.approx(1000000.0)
            assert header['rmsNormEpsilon'] == pytest.approx(1e-6)

            tensor_names = set(_read_tbf_tensor_names(output_path))
            assert 'layer_0_attn_q_bias' in tensor_names
            assert 'layer_0_attn_k_bias' in tensor_names
            assert 'layer_0_attn_v_bias' in tensor_names
            assert 'layer_0_attn_o_bias' not in tensor_names

    def test_infer_config_gqa_fallback_detects_kv_difference(self):
        """Fallback inference (no config.json) should detect KV heads differ from Q heads"""
        import torch

        hidden_dim = 2048
        kv_dim = 256  # 4 KV heads * 64 head_dim

        state_dict = {
            'model.embed_tokens.weight': torch.randn(32000, hidden_dim),
            'model.layers.0.self_attn.q_proj.weight': torch.randn(hidden_dim, hidden_dim),
            'model.layers.0.self_attn.k_proj.weight': torch.randn(kv_dim, hidden_dim),
            'model.layers.0.self_attn.v_proj.weight': torch.randn(kv_dim, hidden_dim),
            'model.layers.0.self_attn.o_proj.weight': torch.randn(hidden_dim, hidden_dim),
            'model.layers.0.mlp.gate_proj.weight': torch.randn(5632, hidden_dim),
            'model.layers.0.mlp.up_proj.weight': torch.randn(5632, hidden_dim),
            'model.layers.0.mlp.down_proj.weight': torch.randn(hidden_dim, 5632),
            'model.layers.0.input_layernorm.weight': torch.randn(hidden_dim),
            'model.layers.0.post_attention_layernorm.weight': torch.randn(hidden_dim),
            'lm_head.weight': torch.randn(32000, hidden_dim),
        }

        config = infer_config_from_weights(state_dict)

        # Fallback can't know exact num_heads without config.json,
        # but it MUST detect that num_kv_heads != num_heads (GQA is present)
        assert config.num_kv_heads < config.num_heads, \
            f"Fallback should detect GQA: num_kv_heads={config.num_kv_heads} should be < num_heads={config.num_heads}"
        assert config.hidden_dim == hidden_dim


class TestQuantization:
    """Test INT8 quantization accuracy"""
    
    def test_quantize_int8_accuracy(self):
        """INT8 quantization should have < 1% error"""
        import torch
        
        # Create a test tensor with known distribution
        np.random.seed(42)
        float_tensor = np.random.randn(256, 256).astype(np.float32)
        
        # Quantize
        quantized, scales, zero_points = quantize_int8_per_channel(float_tensor)
        
        # Validate output types
        assert quantized.dtype == np.int8
        assert scales.dtype == np.float32
        assert len(scales) == 256  # Per-channel quantization
        
        # Dequantize and check error
        dequantized = quantized.astype(np.float32) * scales[:, np.newaxis]
        
        # Calculate relative error
        max_val = np.abs(float_tensor).max()
        relative_error = np.abs(float_tensor - dequantized).max() / max_val
        
        # Should be < 1% error
        assert relative_error < 0.01, f"Quantization error too high: {relative_error:.4f}"
    
    def test_quantize_preserves_shape(self):
        """Quantization should preserve tensor shape"""
        import torch
        
        float_tensor = np.random.randn(128, 256).astype(np.float32)
        quantized, scales, _ = quantize_int8_per_channel(float_tensor)
        
        assert quantized.shape == float_tensor.shape
        assert len(scales) == 128  # Num output channels

    def test_int4_2d_weights_are_transposed_before_packing(self):
        """INT4 2D weights must store packed bytes in TinyBrain [in, out] order."""
        group_size = 2
        hidden_dim = 6
        kv_dim = 2
        intermediate_dim = 8

        # PyTorch linear weights are [out, in]. Distinct values make any missed
        # transpose visible after Swift-style flat dequantization into [in, out].
        k_weight = np.array(
            [[out * 1000 + i for i in range(hidden_dim)] for out in range(kv_dim)],
            dtype=np.float32,
        )

        config = ModelConfig(
            num_layers=1,
            hidden_dim=hidden_dim,
            num_heads=3,
            num_kv_heads=1,
            vocab_size=4,
            intermediate_dim=intermediate_dim,
        )
        weights = {
            'embeddings': np.zeros((config.vocab_size, hidden_dim), dtype=np.float32),
            'layers': [{
                'q_proj': np.ones((hidden_dim, hidden_dim), dtype=np.float32),
                'k_proj': k_weight,
                'v_proj': np.ones((kv_dim, hidden_dim), dtype=np.float32),
                'o_proj': np.ones((hidden_dim, hidden_dim), dtype=np.float32),
                'gate_proj': np.ones((intermediate_dim, hidden_dim), dtype=np.float32),
                'up_proj': np.ones((intermediate_dim, hidden_dim), dtype=np.float32),
                'down_proj': np.ones((hidden_dim, intermediate_dim), dtype=np.float32),
                'input_layernorm': np.ones((hidden_dim,), dtype=np.float32),
                'post_attention_layernorm': np.ones((hidden_dim,), dtype=np.float32),
            }],
            'lm_head': np.ones((config.vocab_size, hidden_dim), dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix='.tbf', delete=False) as f:
            int4_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.tbf', delete=False) as f:
            int8_path = f.name

        try:
            write_tbf_format(weights, config, int4_path,
                             quantize_mode='int4', group_size=group_size)
            int4_tensor = _read_tbf_quantized_tensor(int4_path, 'layer_0_attn_k')

            assert int4_tensor['precision'] == 2
            assert int4_tensor['mode'] == 3
            assert int4_tensor['group_size'] == group_size
            assert int4_tensor['shape'] == k_weight.T.shape

            actual_int4 = _dequantize_int4_swift_order(
                int4_tensor['data'],
                int4_tensor['shape'],
                int4_tensor['scales'],
                int4_tensor['zero_points'],
                int4_tensor['group_size'],
            )

            expected_packed, expected_scales, expected_zps = quantize_int4_per_group(
                np.ascontiguousarray(k_weight.T),
                group_size=group_size,
            )
            expected_int4 = _dequantize_int4_swift_order(
                expected_packed,
                k_weight.T.shape,
                expected_scales,
                expected_zps,
                group_size,
            )
            np.testing.assert_allclose(actual_int4, expected_int4, atol=1e-6)

            expected_group = (
                np.arange(k_weight.size, dtype=np.int64) // group_size
            ).reshape(k_weight.T.shape)
            expected_error_bound = expected_scales[expected_group] / 2.0 + 1e-6
            assert np.all(np.abs(actual_int4 - k_weight.T) <= expected_error_bound), (
                f"INT4 dequantized matrix should approximate W.T.\n"
                f"actual:\n{actual_int4}\nexpected W.T:\n{k_weight.T}"
            )

            # INT8 already follows this convention; keep it as a regression guard
            # and as a contrast with the INT4 packing path.
            write_tbf_format(weights, config, int8_path, quantize_mode='int8')
            int8_tensor = _read_tbf_quantized_tensor(int8_path, 'layer_0_attn_k')
            assert int8_tensor['precision'] == 1
            assert int8_tensor['shape'] == k_weight.T.shape

            actual_int8 = _dequantize_int8_swift_order(
                int8_tensor['data'],
                int8_tensor['shape'],
                int8_tensor['scales'],
                int8_tensor['zero_points'],
            )
            expected_q, expected_int8_scales, expected_int8_zps = quantize_int8_per_channel(k_weight)
            expected_int8 = _dequantize_int8_swift_order(
                np.ascontiguousarray(expected_q.T),
                k_weight.T.shape,
                expected_int8_scales,
                expected_int8_zps,
            )
            np.testing.assert_allclose(actual_int8, expected_int8, atol=1e-6)

            int8_error_bound = expected_int8_scales[np.arange(kv_dim)] / 2.0 + 1e-6
            assert np.all(np.abs(actual_int8 - k_weight.T) <= int8_error_bound[np.newaxis, :])
        finally:
            os.unlink(int4_path)
            os.unlink(int8_path)

    def test_int4_conversion_keeps_output_head_int8(self):
        """INT4 model conversion must keep logits-sensitive output projection at INT8."""
        hidden_dim = 6
        kv_dim = 2
        intermediate_dim = 8
        config = ModelConfig(
            num_layers=1,
            hidden_dim=hidden_dim,
            num_heads=3,
            num_kv_heads=1,
            vocab_size=5,
            intermediate_dim=intermediate_dim,
        )
        weights = {
            'embeddings': np.zeros((config.vocab_size, hidden_dim), dtype=np.float32),
            'layers': [{
                'q_proj': np.ones((hidden_dim, hidden_dim), dtype=np.float32),
                'k_proj': np.ones((kv_dim, hidden_dim), dtype=np.float32),
                'v_proj': np.ones((kv_dim, hidden_dim), dtype=np.float32),
                'o_proj': np.ones((hidden_dim, hidden_dim), dtype=np.float32),
                'gate_proj': np.ones((intermediate_dim, hidden_dim), dtype=np.float32),
                'up_proj': np.ones((intermediate_dim, hidden_dim), dtype=np.float32),
                'down_proj': np.ones((hidden_dim, intermediate_dim), dtype=np.float32),
                'input_layernorm': np.ones((hidden_dim,), dtype=np.float32),
                'post_attention_layernorm': np.ones((hidden_dim,), dtype=np.float32),
            }],
            'lm_head': np.arange(config.vocab_size * hidden_dim, dtype=np.float32)
                .reshape(config.vocab_size, hidden_dim),
        }

        with tempfile.NamedTemporaryFile(suffix='.tbf', delete=False) as f:
            output_path = f.name

        try:
            write_tbf_format(weights, config, output_path,
                             quantize_mode='int4', group_size=2)

            output_tensor = _read_tbf_quantized_tensor(output_path, 'output')
            hidden_tensor = _read_tbf_quantized_tensor(output_path, 'layer_0_attn_q')

            assert output_tensor['precision'] == 1
            assert output_tensor['mode'] == 2
            assert output_tensor['group_size'] == 0
            assert output_tensor['shape'] == weights['lm_head'].T.shape

            assert hidden_tensor['precision'] == 2
            assert hidden_tensor['mode'] == 3
            assert hidden_tensor['group_size'] == 2
        finally:
            os.unlink(output_path)


class TestTBFFormat:
    """Test TBF format compliance with docs/tbf-format-spec.md"""
    
    def test_tbf_format_compliance(self):
        """Written TBF file should match specification"""
        import torch
        
        # Create minimal model weights
        config = ModelConfig(
            num_layers=1,
            hidden_dim=64,
            num_heads=2,
            vocab_size=100,
            intermediate_dim=256,
        )
        
        weights = {
            'embeddings': np.random.randn(100, 64).astype(np.float32),
            'layers': [{
                'q_proj': np.random.randn(64, 64).astype(np.float32),
                'k_proj': np.random.randn(64, 64).astype(np.float32),
                'v_proj': np.random.randn(64, 64).astype(np.float32),
                'o_proj': np.random.randn(64, 64).astype(np.float32),
                'gate_proj': np.random.randn(256, 64).astype(np.float32),
                'up_proj': np.random.randn(256, 64).astype(np.float32),
                'down_proj': np.random.randn(64, 256).astype(np.float32),
                'input_layernorm': np.random.randn(64).astype(np.float32),
                'post_attention_layernorm': np.random.randn(64).astype(np.float32),
            }],
            'lm_head': np.random.randn(100, 64).astype(np.float32),
        }
        
        # Write to TBF format
        with tempfile.NamedTemporaryFile(suffix='.tbf', delete=False) as f:
            output_path = f.name
        
        try:
            write_tbf_format(weights, config, output_path, quantize_mode='int8')
            
            # Validate file exists and has content
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Validate TBF header (magic bytes "TBFM")
            with open(output_path, 'rb') as f:
                magic = f.read(4)
                assert magic == b'TBFM', f"Invalid magic bytes: {magic}"
                
                # Version (UInt32)
                version = int.from_bytes(f.read(4), byteorder='little')
                assert version > 0, "Invalid version"
            
        finally:
            os.unlink(output_path)
    
    def test_tbf_4kb_alignment(self):
        """TBF format should use 4KB page alignment per spec"""
        # This test validates that weight sections are 4KB aligned
        # for efficient mmap loading
        pytest.skip("4KB alignment validation - implement after basic format works")


class TestRoundTrip:
    """Test complete conversion pipeline"""
    
    def test_roundtrip_swift(self):
        """Convert → Load in Swift → verify shapes"""
        # This requires:
        # 1. Python: convert model to TBF
        # 2. Swift: load TBF via ModelWeights.load(from:)
        # 3. Validate shapes match
        
        # For now, skip - this needs Swift integration
        pytest.skip("Round-trip test requires Swift integration - manual validation needed")


class TestCLI:
    """Test command-line interface"""
    
    def test_cli_help(self):
        """CLI should show help message"""
        import subprocess
        
        result = subprocess.run(
            [sys.executable, 'Scripts/convert_model.py', '--help'],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert 'input' in result.stdout.lower()
        assert 'output' in result.stdout.lower()
    
    def test_cli_missing_args(self):
        """CLI should fail gracefully with missing arguments"""
        import subprocess
        
        result = subprocess.run(
            [sys.executable, 'Scripts/convert_model.py'],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode != 0  # Should fail
        assert 'required' in result.stderr.lower() or 'usage' in result.stderr.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
