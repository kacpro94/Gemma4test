from mlx_lm import load, generate

model_id = "mlx-community/gemma-4-e2b-it-4bit"

print("⏳ Ładowanie modelu MLX...")
model, tokenizer = load(model_id)

# Gemma 4 oczekuje konkretnego formatu konwersacji
messages = [
    {"role": "user", "content": "Cześć! Jeśli mnie słyszysz, napisz: 'Gemma 4 melduje się na pokładzie Maca M1!'"}
]

# Tworzymy prompt używając szablonu modelu
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("✍️ Generowanie...")
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=200, 
    verbose=True
)

# Jeśli generate samo nie wypisało tekstu (zależy od wersji), robimy to ręcznie:
if response:
    print("\n--- ODPOWIEDŹ ---")
    print(response)
else:
    print("\nBłąd: Model nadal nie wygenerował tekstu.")