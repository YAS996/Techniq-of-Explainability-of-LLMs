import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import IntegratedGradients

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

text = "This is a good movie."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

def construct_input_embeddings(input_ids):
    return model.bert.embeddings(input_ids)

def forward_func(embeddings, attention_mask):
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    probs = F.softmax(outputs.logits, dim=1)
    return probs[:, 1]  

+input_embeddings = construct_input_embeddings(input_ids)
input_embeddings.requires_grad_()

baseline = torch.zeros_like(input_embeddings)

ig = IntegratedGradients(forward_func)
attributions, delta = ig.attribute(
    inputs=input_embeddings,
    baselines=baseline,
    additional_forward_args=(attention_mask,),
    return_convergence_delta=True
)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
attribution_scores = attributions.sum(dim=-1).squeeze(0)

print("Token Attributions:")
for token, score in zip(tokens, attribution_scores):
    print(f"{token}: {score.item():.4f}")
