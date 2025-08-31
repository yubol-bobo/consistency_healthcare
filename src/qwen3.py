def chat_with_qwen3(messages, model=None, tokenizer=None, max_new_tokens=1024):
    """
    Use Qwen3's chat template to generate a response, returning both thinking content and the final answer.
    messages: list of dicts, e.g. [{"role": "user", "content": "Hello!"}]
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_qwen3_model()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    # Try to find </think> token (id 151668)
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return thinking_content, content


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

