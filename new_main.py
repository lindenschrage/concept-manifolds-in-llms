from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pprint
from helpers import get_embedding_dict, dict_to_json, compute_geometry, process_geometry, convert_to_serializable

access_token = "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"
llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)

llama_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=access_token,
    output_hidden_states=True)

input_filepath = '/n/home09/lschrage/projects/llama/sompolinsky-research/long_inputs.json'

with open(input_filepath, 'r') as file:
    inputs_dict = json.load(file)

## GENERATE EMBEDDINGS
'''
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
'''
toplayerdict = get_embedding_dict(inputs_dict, llama_model, llama_tokenizer)
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

## STORE AS JSON FILE
'''
inner_dict = {
    outer_key: {
        inner_key: [tensor.cpu().detach().numpy().tolist() for tensor in inner_value]
        for inner_key, inner_value in outer_value.items()
    }
    for outer_key, outer_value in toplayerdict.items()
}

data_dict = {}

for key in inner_dict:
    data_dict[f'{key}'] = np.array(inner_dict[key]).T.tolist()
'''

data_dict = dict_to_json(toplayerdict)
with open('/n/home09/lschrage/projects/llama/outputs/manifolds.json', 'w') as json_file:
    json.dump(data_dict, json_file, indent=4)


#CALCULATE MANIFOLD GEOMETRY
with open('/n/home09/lschrage/projects/llama/outputs/manifolds.json', 'r') as file:
    open_data_dict = json.load(file)

geometry = compute_geometry(open_data_dict)

'''
first_outer_key = next(iter(geometry))
threshold_keys = geometry[first_outer_key].keys()

centers = {key: [] for key in threshold_keys}
Rs = {key: [] for key in threshold_keys}
Us = {key: [] for key in threshold_keys}
MSR = {key: [] for key in threshold_keys}

for outer_key, inner_dicts in geometry.items():
    for threshold, attributes in inner_dicts.items():
        centers[threshold].append(attributes['center'])
        Rs[threshold].append(attributes['Rs'])
        Us[threshold].append(attributes['Us'])
        MSR[threshold].append(attributes['mean_squared_radius'])

dists = {}
dists_norm = {}
dsvds = {}
bias = {}
signal = {}
bias = {}
signal = {}

for key in centers.keys():
    matrix = squareform(pdist(centers[key]))
    dists[key] = matrix

    Rs_array = np.array(Rs[key])
    norm_factor = np.sqrt((Rs_array**2).mean())
    dists_norm[key] = matrix / norm_factor

    l = []
    for r in Rs_array:
      total_variance_squared = np.sum(r**2)**2
      sum_fourth_powers = np.sum(r**4)
      l.append(total_variance_squared / sum_fourth_powers)
    dsvds[key] = l

    m = []
    n = []
    for r in np.array(Rs[key]):
      m.append((r**2).sum(-1))
      n.append((r**2).sum(-1))
    m = np.array(m)
    n = np.array(n)
    bias[key] = m/n[:,None] - 1

    signal[key] = dists_norm[key]**2 + bias[key]
'''
dists, dists_norm, dsvds, bias, signal = process_geometry(geometry)

pp = pprint.PrettyPrinter(indent=4)
print("Distances:")
pp.pprint(dists)
print("\nNormalized Distances:")
pp.pprint(dists_norm)
print("\nDsvds (Participation Ratio):")
pp.pprint(dsvds)
print("\nBiases:")
pp.pprint(bias)
print("\nSignals:")
pp.pprint(signal)

data = {
    "Distances": convert_to_serializable(dists),
    "Normalized Distances": convert_to_serializable(dists_norm),
    "Dsvds (Participation Ratio)": convert_to_serializable(dsvds),
    "Biases": convert_to_serializable(bias),
    "Signals": convert_to_serializable(signal)
}

# Writing the data to a JSON file
with open('/n/home09/lschrage/projects/llama/outputs/outputs.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)