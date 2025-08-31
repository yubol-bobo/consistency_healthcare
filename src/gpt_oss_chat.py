from gpt_oss import load_gpt_oss_model, chat_with_gpt_oss
import json
import os
import time
import subprocess
import csv
import pandas as pd
import re
from datetime import datetime

def cleanup_existing_processes():
    """Kill any existing Python processes to free up memory before starting"""
    try:
        print("🧹 Cleaning up existing Python processes...")
        
        # Get current process ID to avoid killing ourselves
        current_pid = os.getpid()
        
        # Find and kill other gpt_oss_chat.py processes (not this one)
        result = subprocess.run(['pgrep', '-f', 'gpt_oss_chat.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid and int(pid) != current_pid:
                    try:
                        subprocess.run(['kill', '-9', pid], check=False)
                        print(f"  Killed process {pid}")
                    except:
                        pass
        
        # Kill any hanging model processes
        subprocess.run(['pkill', '-f', 'transformers'], capture_output=True, check=False)
        subprocess.run(['pkill', '-f', 'torch'], capture_output=True, check=False)
        
        # Clear system cache for good measure
        subprocess.run(['sync'], check=False)
        try:
            subprocess.run(['sudo', 'bash', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], 
                          capture_output=True, check=False)
        except:
            pass  # Ignore if sudo fails
            
        print("✅ Memory cleanup completed")
        
        # Show current memory status
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Current memory status:")
            print(result.stdout)
            
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
        print("Continuing anyway...")

def load_qa_from_csv(csv_path):
    """Load multiple choice Q&A pairs from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        qa_pairs = []
        for _, row in df.iterrows():
            # Format the multiple choice question
            question_text = f"{row['question']}\n\nA) {row['option_a']}\nB) {row['option_b']}\nC) {row['option_c']}\nD) {row['option_d']}\n\nPlease select the correct answer (A, B, C, or D) and explain your reasoning."
            
            qa_pairs.append({
                'question': row['question'],
                'formatted_question': question_text,
                'option_a': row['option_a'],
                'option_b': row['option_b'], 
                'option_c': row['option_c'],
                'option_d': row['option_d'],
                'correct_answer': row['correct_answer'],
                'explanation': row['explanation']
            })
        return qa_pairs
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def batch_chat():
    # Clean up any existing processes first
    cleanup_existing_processes()
    
    # Load Q&A pairs from CSV
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    csv_path = os.path.join(data_dir, "qa.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    qa_pairs = load_qa_from_csv(csv_path)
    if not qa_pairs:
        print("❌ No Q&A pairs loaded from CSV")
        return
        
    print(f"📋 Loaded {len(qa_pairs)} Q&A pairs from {csv_path}")
    
    model, tokenizer = load_gpt_oss_model()
    print("🚀 Starting GPT OSS 20B batch processing...\n")
    
    # Create generated directory and filename with timestamp
    generated_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated")
    os.makedirs(generated_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_file = os.path.join(generated_dir, f"gpt_oss_batch_qa_results_{timestamp}.jsonl")
    
    conversation_id = timestamp
    
    for i, qa_pair in enumerate(qa_pairs, 1):
        question = qa_pair['question']
        formatted_question = qa_pair['formatted_question']
        correct_answer = qa_pair['correct_answer']
        explanation = qa_pair['explanation']
        
        print(f"🔍 Question {i}/{len(qa_pairs)}: {question}")
        print(f"A) {qa_pair['option_a']}")
        print(f"B) {qa_pair['option_b']}")
        print(f"C) {qa_pair['option_c']}")
        print(f"D) {qa_pair['option_d']}")
        
        # Reset messages for each question (no conversation history)
        messages = [{"role": "user", "content": formatted_question}]
        
        # Time the generation
        start_time = time.time()
        thinking, response, thinking_token_probs, response_token_probs = chat_with_gpt_oss(
            messages, model, tokenizer, max_new_tokens=512
        )
        generation_time = time.time() - start_time
        
        print(f"🤖 Model Response: {response}")
        print(f"✅ Correct Answer: {correct_answer}")
        print(f"📚 Explanation: {explanation}")
        
        # Simple answer extraction (look for A, B, C, or D in response)
        model_answer = "Unknown"
        answer_match = re.search(r'\b([ABCD])\b', response.upper())
        if answer_match:
            model_answer = answer_match.group(1)
        
        is_correct = model_answer == correct_answer.upper()
        print(f"🎯 Model Selected: {model_answer} {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
        # Performance info
        total_tokens = len(thinking_token_probs) + len(response_token_probs)
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        print(f"⚡ Performance: {generation_time:.2f}s, {total_tokens} tokens, {tokens_per_second:.1f} tok/s")
        
        # Save to JSONL file
        qa_entry = {
            "conversation_id": conversation_id,
            "question_number": i,
            "timestamp": datetime.now().isoformat(),
            "model": "gpt_oss_20b",
            "question": question,
            "formatted_question": formatted_question,
            "option_a": qa_pair['option_a'],
            "option_b": qa_pair['option_b'],
            "option_c": qa_pair['option_c'],
            "option_d": qa_pair['option_d'],
            "correct_answer": correct_answer,
            "explanation": explanation,
            "model_thinking": thinking,
            "model_response": response,
            "model_selected_answer": model_answer,
            "is_correct": is_correct,
            "generation_time_seconds": generation_time,
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "thinking_token_probs": thinking_token_probs,
            "response_token_probs": response_token_probs
        }
        
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_entry, ensure_ascii=False) + '\n')
        
        print(f"💾 Saved result {i} to {jsonl_file}\n")
        print("-" * 80)
    
    print(f"✅ GPT OSS 20B batch processing completed! Results saved to: {jsonl_file}")

def chat():
    # Clean up any existing processes first
    cleanup_existing_processes()
    
    model, tokenizer = load_gpt_oss_model()
    print("GPT OSS 20B Chat. Type 'exit' to quit.")
    
    # Create generated directory and filename with timestamp
    generated_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated")
    os.makedirs(generated_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_file = os.path.join(generated_dir, f"gpt_oss_chat_history_{timestamp}.jsonl")
    
    messages = []
    conversation_id = timestamp
    turn_number = 0
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            break
            
        turn_number += 1
        messages.append({"role": "user", "content": user_input})
        
        # Time the generation
        start_time = time.time()
        thinking, response, thinking_token_probs, response_token_probs = chat_with_gpt_oss(messages, model, tokenizer, max_new_tokens=512)
        generation_time = time.time() - start_time
        
        print(f"GPT OSS (thinking): {thinking}")
        print(f"GPT OSS: {response}")
        
        # Performance info
        total_tokens = len(thinking_token_probs) + len(response_token_probs)
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        print(f"\n=== Performance ===")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Total tokens: {total_tokens}")
        print(f"Speed: {tokens_per_second:.1f} tokens/second")
        
        # Debug: Check token-probability mapping
        print(f"\n=== Debug Info ===")
        print(f"Thinking tokens count: {len(thinking.split()) if thinking else 0}")
        print(f"Thinking token-prob pairs count: {len(thinking_token_probs)}")
        print(f"Response tokens count: {len(response.split()) if response else 0}")
        print(f"Response token-prob pairs count: {len(response_token_probs)}")
        
        # Show token-probability pairs (first 10 and last 10 for brevity)
        print(f"\nThinking tokens (first 10): {thinking_token_probs[:10]}")
        print(f"Thinking tokens (last 10): {thinking_token_probs[-10:]}")
        print(f"\nResponse tokens (all): {response_token_probs}")
        
        messages.append({"role": "assistant", "content": response})
        
        # Save to JSONL file
        chat_entry = {
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "timestamp": datetime.now().isoformat(),
            "model": "gpt_oss_20b",
            "user_input": user_input,
            "thinking_content": thinking,
            "response_content": response,
            "thinking_token_probs": thinking_token_probs,
            "response_token_probs": response_token_probs,
            "full_messages_history": messages.copy()
        }
        
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(chat_entry, ensure_ascii=False) + '\n')
        
        print(f"\n[Saved to: {jsonl_file}]")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Interactive mode (original behavior)
        chat()
    else:
        # Batch mode (default - process CSV)
        batch_chat()
