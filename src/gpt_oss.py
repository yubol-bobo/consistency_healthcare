def chat_with_gpt_oss(messages, model=None, tokenizer=None, max_new_tokens=512):
    """
    Memory-efficient version: Use GPT OSS 20B to generatedef load_gpt_oss_model():
    print(f"🚀 Loading GPT OSS 20B model from: '{MODEL_LOCAL_PATH}'")
    
    # Check GPU readiness first
    gpu_ready, gpu_status = check_gpu_readiness()
    print(f"🔍 GPU Status: {gpu_status}")
    
    # Check if local model exists
    if not os.path.exists(MODEL_LOCAL_PATH) or not os.listdir(MODEL_LOCAL_PATH):
        print(f"❌ Local model not found at {MODEL_LOCAL_PATH}")
        print("💡 Please run download_models.py to download the GPT OSS 20B model")
        raise FileNotFoundError(f"GPT OSS 20B model not found at {MODEL_LOCAL_PATH}")
    
    model_path = MODEL_LOCAL_PATH
    print(f"✅ Found GPT OSS 20B model at {model_path}"), returning both thinking content 
    and the final answer, along with token-probability pairs for each step.
    Note: GPT OSS 20B doesn't have built-in thinking mode like Qwen3, so we simulate it.
    messages: list of dicts, e.g. [{"role": "user", "content": "Hello!"}]
    Returns:
        thinking_content, response_content, thinking_token_probs, response_token_probs
        where thinking_token_probs and response_token_probs are lists of (token, probability) tuples
    """
    import torch
    import torch.nn.functional as F
    
    if model is None or tokenizer is None:
        model, tokenizer = load_gpt_oss_model()
    
    # Format messages for GPT OSS with harmony format
    # GPT OSS requires the harmony response format for proper thinking
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            # Use the harmony format as recommended in the documentation
            content = msg['content']
            formatted_messages.append({"role": "user", "content": content})
        else:
            formatted_messages.append(msg)
    
    # Apply the chat template (should automatically use harmony format)
    try:
        text = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("✅ Using harmony chat template")
    except Exception as e:
        print(f"⚠️  Chat template failed ({e}), using fallback format")
        # Fallback to manual formatting
        text = ""
        for msg in formatted_messages:
            if msg["role"] == "user":
                text += f"Human: {msg['content']}\n\nAssistant: "
            elif msg["role"] == "assistant":
                text += f"{msg['content']}\n\nHuman: "
        
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = len(model_inputs.input_ids[0])
    
    # Generate with GPT OSS optimized settings for GPU
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        gen_out = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=0.2,     # Lower temperature for more focused reasoning
            top_p=0.9,          # Nucleus sampling 
            top_k=50,           # Top-k sampling
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,  # Slight repetition penalty
            use_cache=True,     # Enable KV caching for speed
            num_beams=1,        # Use sampling instead of beam search for efficiency
        )
    
    output_ids = gen_out.sequences[0][input_length:].tolist()
    scores = gen_out.scores  # list of tensors, each [1, vocab_size]
    
    # For GPT OSS, we'll try to detect thinking vs response based on content patterns
    # Look for common transition phrases
    transition_phrases = ["Final answer:", "Answer:", "In conclusion:", "Therefore:", "So the answer is"]
    think_end_index = None
    
    # Decode the full output to look for transition phrases
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    for phrase in transition_phrases:
        if phrase.lower() in full_output.lower():
            # Find approximate token position
            phrase_pos = full_output.lower().find(phrase.lower())
            # Estimate token position (rough approximation)
            think_end_index = int(len(output_ids) * phrase_pos / len(full_output))
            break
    
    if think_end_index is None:
        # If no clear transition found, split roughly in the middle or use first 60%
        think_end_index = int(len(output_ids) * 0.6)
    
    # Process tokens and probabilities memory-efficiently
    thinking_token_probs = []
    response_token_probs = []
    
    # Batch decode tokens for efficiency
    all_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in output_ids]
    
    for i, (token_id, token_text, score) in enumerate(zip(output_ids, all_tokens, scores)):
        # Convert logits to probabilities and extract only the probability of the generated token
        with torch.no_grad():
            prob = F.softmax(score[0], dim=-1)[token_id].cpu().item()
        
        if i < think_end_index:
            thinking_token_probs.append((token_text, prob))
        else:
            response_token_probs.append((token_text, prob))
        
        # Clear score tensor to free memory immediately
        del score
    
    # Decode content sections
    thinking_tokens = output_ids[:think_end_index]
    response_tokens = output_ids[think_end_index:]
    
    thinking_content = tokenizer.decode(thinking_tokens, skip_special_tokens=True).strip()
    response_content = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    
    # Clean up GPU memory
    del gen_out, scores, output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all GPU operations complete
    
    return thinking_content, response_content, thinking_token_probs, response_token_probs


from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import os


# GPT OSS 20B model path - optimized for GPU deployment
MODEL_LOCAL_PATH = os.path.abspath(os.path.normpath("./models/gpt_oss_20b"))

def check_gpu_readiness():
    """Check if the system is ready for GPT OSS 20B deployment"""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    device_count = torch.cuda.device_count()
    total_memory = sum(torch.cuda.get_device_properties(i).total_memory / 1024**3 
                      for i in range(device_count))
    
    if total_memory < 30:
        return False, f"Insufficient GPU memory: {total_memory:.1f}GB (need 40GB+)"
    
    return True, f"Ready: {device_count} GPU(s), {total_memory:.1f}GB total"

def load_gpt_oss_model():
    print(f"Loading GPT OSS 20B model from local path: '{MODEL_LOCAL_PATH}'...")
    
    # Check if local model exists
    if not os.path.exists(MODEL_LOCAL_PATH) or not os.listdir(MODEL_LOCAL_PATH):
        print(f"❌ Local model not found at {MODEL_LOCAL_PATH}")
        print("� Please run download_models.py to download the GPT OSS 20B model")
        raise FileNotFoundError(f"GPT OSS 20B model not found at {MODEL_LOCAL_PATH}")
    
    model_path = MODEL_LOCAL_PATH
    print(f"✅ Found GPT OSS 20B model at {model_path}")
    
    # Check available devices and provide GPU recommendations
    import torch
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.device_count()} GPU(s)")
        total_memory = 0
        for i in range(torch.cuda.device_count()):
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            total_memory += mem_gb
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.1f} GB)")
        print(f"  Total GPU Memory: {total_memory:.1f} GB")
        
        # Memory recommendations
        if total_memory >= 40:
            print("✅ Sufficient memory for GPT OSS 20B")
        else:
            print("⚠️  GPT OSS 20B requires ~40GB+ VRAM for optimal performance")
            print("💡 Consider using gradient checkpointing or model parallelism")
    else:
        print("🔥 No CUDA detected - GPT OSS 20B is optimized for GPU deployment")
        print("💡 Recommended setups:")
        print("   • 2x RTX 4090 (48GB total VRAM) - Excellent for inference")
        print("   • A100 80GB - Single GPU solution")
        print("   • RTX 6000 Ada (48GB) - Professional option")
        print("🎯 AWS GPU Instances:")
        print("   • p4d.xlarge (A100 40GB) - Good for testing")
        print("   • p4d.2xlarge (A100 80GB) - Recommended")
        print("   • g5.xlarge (A10G 24GB) - Budget option with model parallelism")
        print("   • g5.2xlarge (A10G 24GB) - For multi-GPU setup")
    
    # Import kernels to enable GPT OSS support
    try:
        import kernels
        print("✅ Kernels package imported successfully")
    except ImportError:
        print("❌ Kernels package not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "-U", "kernels"])
        import kernels
        print("✅ Kernels package installed and imported")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load with optimizations - GPT OSS 20B specific settings for GPU
    print("🔄 Loading GPT OSS 20B model (optimized for GPU deployment)...")
    
    # Use bfloat16 for optimal GPU performance (GPT OSS is designed for this)
    dtype = torch.bfloat16
    print("� Using bfloat16 dtype for optimal GPU performance")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=dtype,
        device_map="auto",  # Automatically distribute across available GPUs
        low_cpu_mem_usage=True,
        trust_remote_code=True,   # Required for custom GPT OSS architecture
        attn_implementation="flash_attention_2",  # Use Flash Attention 2 for speed
        use_cache=True,  # Enable KV caching for faster generation
        max_memory={0: "40GB", 1: "40GB"} if torch.cuda.device_count() >= 2 else None  # For dual GPU setup
    )
    
    print(f"🎯 GPT OSS 20B model loaded and ready for GPU inference!")
    print(f"   Device: {model.device}")
    print(f"   Dtype: {model.dtype}")
    print(f"   Parameters: ~20B (3.6B active via MoE)")
    print(f"   Model config: {type(model.config).__name__}")
    print(f"   Flash Attention: {'✅ Enabled' if hasattr(model.config, 'attn_implementation') else '❌ Disabled'}")
    return model, tokenizer
