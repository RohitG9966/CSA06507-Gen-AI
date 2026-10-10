
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")

print("Model loaded successfully!")
print("Model type:", model.config.model_type)
