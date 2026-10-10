
from transformers import AutoTokenizer

text = "Generative AI is very useful."

bert = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt = AutoTokenizer.from_pretrained("gpt2")

print("Original Text:", text)
print("BERT Tokens:", bert.tokenize(text))
print("GPT Tokens:", gpt.tokenize(text))
