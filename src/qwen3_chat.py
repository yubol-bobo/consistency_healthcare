

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
        thinking, response = chat_with_qwen3(messages, model, tokenizer, max_new_tokens=256)
        print(f"Qwen3 (thinking): {thinking}")
        print(f"Qwen3: {response}")
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chat()
