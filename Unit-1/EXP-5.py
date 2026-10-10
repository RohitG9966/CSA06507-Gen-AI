
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial Intelligence is"

result = generator(
    prompt,
    max_new_tokens=30,
    do_sample=False,
    pad_token_id=generator.tokenizer.eos_token_id
)

print("Input Prompt:", prompt)
print("Generated Text:", result[0]["generated_text"])
