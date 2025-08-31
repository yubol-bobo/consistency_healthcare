#!/usr/bin/env python3
"""
GPU Readiness Test for GPT OSS 20B
Run this script to verify your GPU setup before loading the full model.
"""

import torch
import os
import sys

def test_gpu_setup():
    """Test GPU configuration for GPT OSS 20B deployment"""
    print("🔍 Testing GPU setup for GPT OSS 20B...")
    print("=" * 50)
    
    # Basic CUDA check
    print(f"🚀 PyTorch version: {torch.__version__}")
    print(f"🚀 CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please install CUDA-enabled PyTorch.")
        print("💡 Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # GPU details
    device_count = torch.cuda.device_count()
    print(f"🎯 GPU count: {device_count}")
    
    total_memory = 0
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        total_memory += memory_gb
        print(f"   GPU {i}: {props.name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        print(f"   Compute capability: {props.major}.{props.minor}")
    
    print(f"🔥 Total GPU memory: {total_memory:.1f} GB")
    
    # Memory requirements check
    if total_memory >= 40:
        print("✅ Excellent! You have sufficient memory for GPT OSS 20B")
    elif total_memory >= 24:
        print("⚠️  Moderate memory. GPT OSS 20B may run with optimizations:")
        print("   • Use gradient checkpointing")
        print("   • Reduce batch size to 1")
        print("   • Consider quantization")
    else:
        print("❌ Insufficient GPU memory for GPT OSS 20B")
        print("💡 Consider upgrading to:")
        print("   • RTX 4090 (24GB)")
        print("   • A100 (40GB/80GB)")
        print("   • RTX 6000 Ada (48GB)")
        return False
    
    # Test bfloat16 support
    try:
        test_tensor = torch.zeros(1, dtype=torch.bfloat16, device='cuda')
        print("✅ bfloat16 support: Available")
        del test_tensor
    except:
        print("❌ bfloat16 support: Not available")
        return False
    
    # Test Flash Attention
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention
        print("✅ GPT OSS model support: Available")
    except ImportError:
        print("⚠️  GPT OSS model support: Install latest transformers")
        print("   pip install git+https://github.com/huggingface/transformers.git")
    
    # Test kernels package
    try:
        import kernels
        print("✅ Kernels package: Available")
    except ImportError:
        print("⚠️  Kernels package: Not found")
        print("   pip install kernels")
    
    print("=" * 50)
    print("🎯 GPU setup test completed!")
    return True

def estimate_inference_speed():
    """Estimate inference speed based on GPU setup"""
    if not torch.cuda.is_available():
        return
    
    print("\n🚀 Performance Estimates:")
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        
        if "4090" in props.name:
            tokens_per_sec = "~15-25 tokens/sec"
        elif "A100" in props.name:
            tokens_per_sec = "~25-40 tokens/sec"
        elif "A10G" in props.name:
            tokens_per_sec = "~8-15 tokens/sec"
        elif "V100" in props.name:
            tokens_per_sec = "~10-20 tokens/sec"
        else:
            tokens_per_sec = "~5-15 tokens/sec (estimated)"
        
        print(f"   GPU {i} ({props.name}): {tokens_per_sec}")

if __name__ == "__main__":
    success = test_gpu_setup()
    estimate_inference_speed()
    
    if success:
        print("\n✅ Your system is ready for GPT OSS 20B deployment!")
        print("🚀 Run: python ./src/qwen3_chat.py --gpt")
    else:
        print("\n❌ Please resolve the issues above before deploying GPT OSS 20B")
        sys.exit(1)
