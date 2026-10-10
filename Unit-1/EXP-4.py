
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "AI makes learning easier."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs)

print("Input Text:", text)
print("Embedding Shape:", output.last_hidden_state.shape)
