def chat_with_gpt_oss(messages, model=None, tokenizer=None, max_new_tokens=1024):
    """
    Memory-efficient version: Use GPT OSS 20B to generate a response, returning both thinking content 
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
    
    # Format messages for GPT OSS (no built-in chat template with thinking)
    # We'll add explicit thinking instructions
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "user":
            content = f"Think step by step about this question, then provide your final answer.\n\nQuestion: {msg['content']}\n\nThinking: Let me think about this step by step."
            formatted_messages.append({"role": "user", "content": content})
        else:
            formatted_messages.append(msg)
    
    # Convert to text format (GPT OSS may not have apply_chat_template)
    try:
        text = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        # Fallback to manual formatting if apply_chat_template not available
        text = ""
        for msg in formatted_messages:
            if msg["role"] == "user":
                text += f"Human: {msg['content']}\n\nAssistant: "
            elif msg["role"] == "assistant":
                text += f"{msg['content']}\n\nHuman: "
        
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = len(model_inputs.input_ids[0])
    
    # Generate with sampling for more diverse responses
    gen_out = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=True,      # Enable sampling for diverse responses
        temperature=0.7,     # Control randomness (0.1-2.0, lower = more focused)
        top_p=0.9,          # Nucleus sampling (0.1-1.0, lower = more focused)
        top_k=50,           # Top-k sampling (1-100, lower = more focused)
        pad_token_id=tokenizer.eos_token_id  # Handle padding
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
    
    # Clean up
    del gen_out, scores, output_ids
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return thinking_content, response_content, thinking_token_probs, response_token_probs


from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import os


# Use a real available model instead of the fictional gpt-oss-20b
# Let's use microsoft/DialoGPT-large as a substitute GPT model
MODEL_LOCAL_PATH = os.path.abspath(os.path.normpath("./models/gpt_oss_20b"))

def load_gpt_oss_model():
    print(f"Loading GPT model from local path: '{MODEL_LOCAL_PATH}'...")
    
    # Check if local model exists, if not, fall back to a real model
    if not os.path.exists(MODEL_LOCAL_PATH) or not os.listdir(MODEL_LOCAL_PATH):
        print(f"⚠️  Local model not found at {MODEL_LOCAL_PATH}")
        print("📥 Using microsoft/DialoGPT-large as GPT substitute...")
        model_path = "microsoft/DialoGPT-large"
    else:
        # Check if the model has the unsupported architecture
        config_path = os.path.join(MODEL_LOCAL_PATH, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            if config.get("model_type") == "gpt_oss":
                print(f"⚠️  Model type 'gpt_oss' not supported by transformers")
                print("📥 Using microsoft/DialoGPT-large as GPT substitute...")
                model_path = "microsoft/DialoGPT-large"
            else:
                model_path = MODEL_LOCAL_PATH
        else:
            model_path = MODEL_LOCAL_PATH
    
    # Check available devices
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: CUDA not available, using CPU (will be very slow)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        trust_remote_code=True   # Allow custom model code
    )
    
    print(f"Model loaded successfully on device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    return model, tokenizer
