from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pprint
import random
from utils import get_embedding_dict, dict_to_json, compute_geometry, process_geometry, convert_to_serializable, sample_tensors_from_dict, plot_participation_ratios
from collections import defaultdict
import matplotlib.pyplot as plt


thresholds = {5: 'top_5_words', 100: 'top_100_words', 300: 'top_300_words'}
results = {key: [] for key in thresholds.values()}

sample_size = 76

access_token = "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"
llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)

llama_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=access_token,
    output_hidden_states=True).to('cuda')

## OPEN 
input_filepath = '/n/home09/lschrage/projects/llama/sompolinsky-research/long_inputs.json'

with open(input_filepath, 'r') as file:
    inputs_dict = json.load(file)

## GENERATE EMBEDDINGS
toplayerdict = get_embedding_dict(thresholds, inputs_dict, llama_model, llama_tokenizer)

for r in range(5):
    new_data = sample_tensors_from_dict(toplayerdict, sample_size)

    ## STORE AS JSON FILE
    data_dict = dict_to_json(new_data)

    # CALCULATE MANIFOLD GEOMETRY
    geometry = compute_geometry(data_dict)
    dists, dists_norm, dsvds, bias, signal = process_geometry(geometry)
    print(dsvds)
    for threshold_name, values in dsvds.items():
        results[threshold_name].append(values)

averaged_results = {}
for key, arrays in results.items():
    averaged_results[key] = np.mean(np.array(arrays), axis=0)

print("Averaged Dsvds (Participation Ratio) for each threshold:")
for key, avg in averaged_results.items():
    print(f"{key}: {avg}")

fig, ax = plt.subplots(figsize=(10, 6))

# Data preparation
thresholds = ['top_5_words', 'top_100_words', 'top_300_words']
indices = np.arange(len(averaged_results[thresholds[0]]))  # the number of bars per group
bar_width = 0.25  # width of bars

# Plotting
for i, threshold in enumerate(thresholds):
    # Calculate the correct position for each bar based on its group
    bar_positions = indices + i * bar_width
    # Plotting each group of bars
    ax.bar(bar_positions, averaged_results[threshold], width=bar_width, label=threshold)

# Labeling and aesthetics
ax.set_xlabel('Concepts')
ax.set_ylabel('Dsvds (Participation Ratio)')
ax.set_title('Dsvds Averaged Values for Different Thresholds')
ax.set_xticks(indices + bar_width)  # positioning x-ticks in the center of the groups of bars
ax.set_xticklabels([f'Concept {i+1}' for i in indices])
ax.legend(title='Thresholds')

# Show the plot
plt.show()

'''
plot_participation_ratios(averaged_results["Dsvds (Participation Ratio)"])

data = {
    "Distances": convert_to_serializable(dists),
    "Normalized Distances": convert_to_serializable(dists_norm),
    "Dsvds (Participation Ratio)": convert_to_serializable(dsvds),
    "Biases": convert_to_serializable(bias),
    "Signals": convert_to_serializable(signal)
}

#WRITE TO JSON
with open('/n/home09/lschrage/projects/llama/outputs/outputs.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)


'''