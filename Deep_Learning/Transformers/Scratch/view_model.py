from transformers import AutoModel, AutoTokenizer

# Choose a pre-trained transformer model from Hugging Face
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Print layer information
for name, param in model.named_parameters():
    print(f"Layer: {name}, Type: {param.data.type()}, Size: {param.size()}")