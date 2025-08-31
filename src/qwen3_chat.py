

from qwen3 import load_qwen3_model, chat_with_qwen3

def chat():
    model, tokenizer = load_qwen3_model()
    print("Qwen3 Chat. Type 'exit' to quit.")
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            break
        messages.append({"role": "user", "content": user_input})
        thinking, response, thinking_probs, response_probs = chat_with_qwen3(messages, model, tokenizer, max_new_tokens=256)
        print(f"Qwen3 (thinking): {thinking}")
        print(f"Qwen3: {response}")
        # Show the probability of each generated token
        print(f"Thinking token probabilities: {[f'{p:.3f}' for p in thinking_probs]}")
        print(f"Response token probabilities: {[f'{p:.3f}' for p in response_probs]}")
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chat()
