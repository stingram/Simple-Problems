from transformers import AutoModel, AutoTokenizer

# Choose a pre-trained transformer model from Hugging Face
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example text for tokenization
text = "Hugging Face is creating a transformer library for natural language processing."

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Print the output layers
for layer_idx, layer_output in enumerate(outputs["last_hidden_state"].squeeze().detach().numpy()):
    print(f"Layer {layer_idx + 1} output:")
    print(layer_output)
    print("=" * 50)