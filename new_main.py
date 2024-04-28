from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import json
import torch
import numpy as np
import pandas as pd


access_token = "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"
llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)

llama_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=access_token,
    output_hidden_states=True)

input_filepath = '/n/home09/lschrage/projects/llama/sompolinsky-research/long_inputs.json'

with open(input_filepath, 'r') as file:
    inputs_dict = json.load(file)

toplayerdict = {}
thresholds = {10: 'top_10_words', 50: 'top_50_words', 100: 'top_100_words'}

for key in inputs_dict:
  toplayer = {thresholds[thresh]: [] for thresh in thresholds}
  for input in inputs_dict[key]:
    tokens = llama_tokenizer.encode(input, return_tensors="pt")
    with torch.no_grad():
      outputs = llama_model(tokens)
      predictions = outputs[0]
    next_token_candidates_tensor = predictions[0, -1, :]
    for thresh in thresholds:
      topk_candidates_indexes = torch.topk(
          next_token_candidates_tensor, thresh).indices.tolist()
      all_candidates_probabilities = torch.nn.functional.softmax(
          next_token_candidates_tensor, dim=-1)
      topk_candidates_probabilities = \
          all_candidates_probabilities[topk_candidates_indexes].tolist()
      topk_candidates_tokens = \
          [llama_tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
      print(topk_candidates_tokens)
      if str(key) in topk_candidates_tokens:
        curr_toplayer = outputs[2][-1][0, -1, :]
        toplayer[thresholds[thresh]].append(curr_toplayer)  
toplayerdict[key] = toplayer

print(len(toplayerdict['dog']['top_10_words']))
print(len(toplayerdict['dog']['top_50_words']))
print(len(toplayerdict['dog']['top_100_words']))
print("")
print(len(toplayerdict['apple']['top_10_words']))
print(len(toplayerdict['apple']['top_50_words']))
print(len(toplayerdict['apple']['top_100_words']))
print("")
print(len(toplayerdict['pen']['top_10_words']))
print(len(toplayerdict['pen']['top_50_words']))
print(len(toplayerdict['pen']['top_100_words']))