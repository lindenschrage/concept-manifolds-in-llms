# -*- coding: utf-8 -*-
"""main

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wmOEtF_U-fZAZrVVpw9tgLTDKQAC8d5R

## Load and save the Llama-2-7b model and tokenizer from Hugging Face
"""

from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pprint

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

access_token = "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"
model = "meta-llama/Llama-2-7b-hf"

llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)

llama_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=access_token,
    output_hidden_states=True,
    output_attentions=True).to(device)

input_filepath = '/n/home09/lschrage/projects/llama/sompolinsky-research/long_inputs.json'

with open(input_filepath, 'r') as file:
    inputs_dict = json.load(file)
print('input', inputs_dict.keys())

outputdict = {}
prob_toplayerdict = {}
probsdict = {}
word_toplayerdict = {}

WORD_THRESHOLD = [40, 50, 100]
WORD_THRESHOLD_NAMES = {40: 'top_40_words', 50: 'top_50_words', 100: 'top_100_words'}

PROBS_THRESHOLD = [0, 10, 20]
PROBS_THRESHOLD_NAMES = {0: 'top_0_perc', 10: 'top_10_perc', 20: 'top_20_perc'}

with torch.no_grad():
    for key in inputs_dict:
        target_word = key
        tokens_target_word = llama_tokenizer.tokenize(target_word)
        target_word_id =  llama_tokenizer.convert_tokens_to_ids(tokens_target_word)
        outputs = []
        prob_toplayers = {PROBS_THRESHOLD_NAMES[p]: [] for p in PROBS_THRESHOLD}  # Initialize as a dictionary
        probs = []
        word_toplayers = {WORD_THRESHOLD_NAMES[w]: [] for w in WORD_THRESHOLD}  # Initialize as a dictionary


        for input in inputs_dict[key]:

            tokens = llama_tokenizer(input, return_tensors="pt").to(device)
            output=llama_model.forward(**tokens, return_dict = True)
            outputs.append(output)

            logits = output['logits'][0][-1]
            probabilities = F.softmax(logits, dim=-1)
            final_layer_hidden_states = output['hidden_states'][-1]
            last_hidden_state_for_last_token = final_layer_hidden_states[0, -1, :]
            prob_target_word = probabilities[target_word_id].item()*100
            probs.append(probabilities)
            
            for p_threshold in PROBS_THRESHOLD:
                if prob_target_word > p_threshold:
                    prob_toplayers[PROBS_THRESHOLD_NAMES[p_threshold]].append(last_hidden_state_for_last_token)

            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            top_preds = llama_tokenizer.convert_ids_to_tokens(sorted_indices)

            for w_threshold in WORD_THRESHOLD:
                top_x_words = top_preds[:w_threshold]
                if ('▁' + str(key) in top_x_words) or (str(key) in top_x_words):
                    word_toplayers[WORD_THRESHOLD_NAMES[w_threshold]].append(last_hidden_state_for_last_token)

        outputdict[key] = outputs
        probsdict[key] = probs
        prob_toplayerdict[key] = prob_toplayers
        word_toplayerdict[key] = word_toplayers

print('word top layer', word_toplayerdict.keys())
print('length of word_toplayerdict[dog][top_40_words]', len(word_toplayerdict['dog']['top_40_words']))
print('length of word_toplayerdict[apple][top_40_words]', len(word_toplayerdict['apple']['top_40_words']))
print('length of word_toplayerdict[dog][top_50_words]', len(word_toplayerdict['dog']['top_50_words']))
print('length of word_toplayerdict[apple][top_50_words]', len(word_toplayerdict['apple']['top_50_words']))
print('length of word_toplayerdict[dog][top_100_words]', len(word_toplayerdict['dog']['top_100_words']))
print('length of word_toplayerdict[apple][top_100_words]', len(word_toplayerdict['apple']['top_100_words']))
print('')
print('')
print('')
print('prob top layer:', prob_toplayerdict.keys())
print('length of prob_toplayerdict[dog][top_0_perc]', len(prob_toplayerdict['dog']['top_0_perc']))
print('length of prob_toplayerdict[apple][top_0_perc]', len(prob_toplayerdict['apple']['top_0_perc']))
print('length of prob_toplayerdict[dog][top_10_perc]', len(prob_toplayerdict['dog']['top_10_perc']))
print('length of prob_toplayerdict[apple][top_10_perc]', len(prob_toplayerdict['apple']['top_10_perc']))
print('length of prob_toplayerdict[dog][top_20_perc]', len(prob_toplayerdict['dog']['top_20_perc']))
print('length of prob_toplayerdict[apple][top_20_perc]', len(prob_toplayerdict['apple']['top_20_perc']))


## force list sizes to be the same accross thresholds
threshold_min_sizes = {threshold: float('inf') for threshold in WORD_THRESHOLD_NAMES.values()}

for key in word_toplayerdict:
    for threshold in WORD_THRESHOLD_NAMES.values():
        if threshold in word_toplayerdict[key]:
            current_length = len(word_toplayerdict[key][threshold])
            if current_length < threshold_min_sizes[threshold]:
                threshold_min_sizes[threshold] = current_length

for key in word_toplayerdict:
    for threshold, min_size in threshold_min_sizes.items():
        if threshold in word_toplayerdict[key] and min_size != float('inf'):
            word_toplayerdict[key][threshold] = word_toplayerdict[key][threshold][:min_size]


## store top layer representations in json dictionary
## WORD TOP LAYERS
word_np_dict = {
    outer_key: {
        inner_key: [tensor.cpu().detach().numpy().tolist() for tensor in inner_value]
        for inner_key, inner_value in outer_value.items()
    }
    for outer_key, outer_value in word_toplayerdict.items()
}
#print('np-dict', np_dict.keys())
## PROB TOP LAYERS

prob_np_dict = {
    outer_key: {
        inner_key: [tensor.cpu().detach().numpy().tolist() for tensor in inner_value]
        for inner_key, inner_value in outer_value.items()
    }
    for outer_key, outer_value in prob_toplayerdict.items()
}

data_dict = {}

for key in word_np_dict:
    data_dict[f'{key}'] = np.array(word_np_dict[key]).T.tolist()

with open('/n/home09/lschrage/projects/llama/outputs/manifolds.json', 'w') as json_file:
    json.dump(data_dict, json_file, indent=4)

    #####################################################
with open('/n/home09/lschrage/projects/llama/outputs/manifolds.json', 'r') as file:
    open_data_dict = json.load(file)

def compute_geometry(manifolds):
    geometry = {}
    for key, inner_dict in manifolds.items():
        geometry[key] = {}
        for subkey, data in inner_dict.items():
            data_array = np.array(data)
            print(data_array.shape)
            centroid = data_array.mean(axis=0)
            norm_manifold = data_array - centroid

            U, S, Vh = np.linalg.svd(norm_manifold, full_matrices=False)
            mean_squared_radius = np.mean(S**2)

            geometry[key][subkey] = {
                'center': centroid,
                'Rs': S,
                'Us': U,
                'mean_squared_radius': mean_squared_radius
            }
    return geometry

geometry = compute_geometry(open_data_dict)

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

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]  
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}  
    else:
        return obj 

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