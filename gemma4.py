from mlx_lm import load, generate
import re

model_id = "mlx-community/gemma-4-e2b-it-4bit"

print("⏳ Ładowanie modelu MLX...")
model, tokenizer = load(model_id)

history = []

print("\n🤖 Gemma-4 gotowa! (Wpisz 'wyjdź', aby zakończyć)")

while True:
    user_input = input("\n👤 Ty: ")
    
    if user_input.lower() in ["exit", "quit", "wyjdź"]:
        print("Pa pa!")
        break

    history.append({
        "role": "user", 
        "content": f"{user_input} (Odpowiedz po polsku, dość krótko, bez procesu myślowego, zawsze użyj 'BRRRRRRRR' przed rozpoczęciem ostatecznej odpowiedzi)"
    })

    
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

    print("✍️ Generowanie...")
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=1250
    )

    response_clean=re.split(r"BRRRRRRRR", response, flags=re.IGNORECASE)[-1].strip()

    print(response_clean)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response_clean})