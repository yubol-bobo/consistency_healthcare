import os
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_qwen3_model(local_dir='models/qwen3'):
    model_name = 'Qwen/Qwen3-8B'
    
    # Check if directory already exists and has files
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Model already exists in {local_dir}, skipping download.")
        return
    
    print(f"Downloading {model_name} to {local_dir} ...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Important: no symlinks, copy actual files
        repo_type="model"
    )
    print(f"Download complete. Model saved to {local_dir}")

def download_gpt_oss_20b(local_dir='models/gpt_oss_20b'):
    model_name = 'openai/gpt-oss-20b'
    
    # Check if directory already exists and has files
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Model already exists in {local_dir}, skipping download.")
        return
    
    print(f"Downloading {model_name} to {local_dir} ...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Important: no symlinks, copy actual files
        repo_type="model"
    )
    print(f"Download complete. Model saved to {local_dir}")

if __name__ == "__main__":
    download_qwen3_model()
    download_gpt_oss_20b()
