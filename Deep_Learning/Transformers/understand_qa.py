from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.nn.functional import softmax

# Example data
question = "Who is the president of the United States?"
context = "The president of the United States is Joe Biden."
answer = "Joe Biden"

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Tokenize input
inputs = tokenizer(question, context, return_tensors="pt")

# Forward pass to get start and end logits
outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Create one-hot encoded labels for start and end positions
start_label = torch.zeros_like(start_logits)
end_label = torch.zeros_like(end_logits)

# Find the positions of the answer in the tokenized sequence
start_position = tokenizer(context, return_offsets_mapping=True)["offset_mapping"].index((answer.lower(), answer.lower()))
end_position = start_position + len(answer.split()) - 1

# Set the corresponding positions in labels to 1
start_label[0, start_position] = 1
end_label[0, end_position] = 1

# Calculate the cross-entropy losses
loss_start = torch.nn.functional.cross_entropy(start_logits, torch.argmax(start_label, dim=1))
loss_end = torch.nn.functional.cross_entropy(end_logits, torch.argmax(end_label, dim=1))

# Sum the losses to get the total loss
total_loss = loss_start + loss_end

# Print the losses
print("Loss for start position prediction:", loss_start.item())
print("Loss for end position prediction:", loss_end.item())
print("Total loss:", total_loss.item())

# Convert logits to probabilities using softmax
start_probs = softmax(start_logits, dim=1)
end_probs = softmax(end_logits, dim=1)

# Get the predicted start and end positions
predicted_start = torch.argmax(start_probs, dim=1).item()
predicted_end = torch.argmax(end_probs, dim=1).item()

# Convert the token IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].numpy())

# Print the answer span
predicted_answer = tokenizer.convert_tokens_to_string(tokens[predicted_start:predicted_end+1])
print("Predicted answer:", predicted_answer)