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
from pathlib import Path

# Add Scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "Scripts"))

try:
    from convert_model import (
        load_pytorch_checkpoint,
        extract_weights,
        quantize_int8_per_channel,
        write_tbf_format,
        ModelConfig,
        infer_config_from_weights,
    )
except ImportError:
    # Tests will fail until we implement convert_model.py
    pytest.skip("convert_model.py not implemented yet", allow_module_level=True)


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

