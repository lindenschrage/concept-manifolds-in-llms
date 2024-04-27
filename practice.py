from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import json
import torch
import numpy as np
import pandas as pd


access_token = "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"
model = "meta-llama/Llama-2-7b-hf"

llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)

llama_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=access_token,
    output_hidden_states=True,
    output_attentions=True)

target_word = 'apple'
tokens_target_word = llama_tokenizer.tokenize(target_word)
target_word_id =  llama_tokenizer.convert_tokens_to_ids(tokens_target_word)
print('Target word: ', target_word)
print('Target word id: ', target_word_id)


input = "I love eating an"
tokens = llama_tokenizer(input, return_tensors="pt").to(device)
output=llama_model.forward(**tokens, return_dict = True)
print('User input: ', input)
print('Tokens: ', tokens['input_ids'][0])

logits = output['logits'][0, -1, :]
probabilities = F.softmax(logits, dim=-1)
final_layer_hidden_states = output['hidden_states'][-1]
last_hidden_state_for_last_token = final_layer_hidden_states[0, -1, :]
prob_target_word = probabilities[target_word_id].item()*100
print('Prob target word:', prob_target_word)

sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
print(sorted_probabilities.sum())

top_preds = llama_tokenizer.convert_ids_to_tokens(sorted_indices)
print('Sorted probabilities: ', sorted_probabilities)
print('Sorted indices: ', sorted_indices)
print('Top predictions: ', top_preds)