def chat_with_qwen3(messages, model=None, tokenizer=None, max_new_tokens=1024):
    """
    Memory-efficient version: Use Qwen3's chat template to generate a response, returning both thinking content 
    and the final answer, along with token probabilities for each step.
    messages: list of dicts, e.g. [{"role": "user", "content": "Hello!"}]
    Returns:
        thinking_content, response_content, thinking_probs, response_probs
    """
    import torch
    import torch.nn.functional as F
    
    if model is None or tokenizer is None:
        model, tokenizer = load_qwen3_model()
        
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = len(model_inputs.input_ids[0])
    
    # Generate with sampling for more diverse responses
    gen_out = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=True,      # Enable sampling for diverse responses
        temperature=0.5,     # Control randomness (0.1-2.0, lower = more focused)
        top_p=0.9,          # Nucleus sampling (0.1-1.0, lower = more focused)
        top_k=10            # Top-k sampling (1-100, lower = more focused)
    )
    
    output_ids = gen_out.sequences[0][input_length:].tolist()
    scores = gen_out.scores  # list of tensors, each [1, vocab_size]
    
    # Find </think> token (id 151668) to separate thinking from response
    think_end_token = 151668
    think_end_index = None
    
    for i, token_id in enumerate(output_ids):
        if token_id == think_end_token:
            think_end_index = i + 1  # Include the </think> token in thinking
            break
    
    if think_end_index is None:
        think_end_index = 0  # No thinking section found
    
    # Process tokens and probabilities memory-efficiently
    thinking_probs = []
    response_probs = []
    
    for i, (token_id, score) in enumerate(zip(output_ids, scores)):
        # Convert logits to probabilities and extract only the probability of the generated token
        with torch.no_grad():
            prob = F.softmax(score[0], dim=-1)[token_id].cpu().item()
        
        if i < think_end_index:
            thinking_probs.append(prob)
        else:
            response_probs.append(prob)
        
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
    
    return thinking_content, response_content, thinking_probs, response_probs


from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import os


# Use the clean local directory for the model and tokenizer
MODEL_LOCAL_PATH = os.path.abspath(os.path.normpath("./models/qwen3"))

def load_qwen3_model():
    print(f"Loading model and tokenizer from local path: '{MODEL_LOCAL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL_PATH, torch_dtype=torch.float16, device_map="auto")
    print("Model and tokenizer loaded successfully from local files.")
    return model, tokenizer

