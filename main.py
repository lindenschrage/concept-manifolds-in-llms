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


thresholds = {5: 'top_5_words', 100: 'top_100_words', 300: 'top_300_words'}
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

results = defaultdict(lambda: defaultdict(list))
for _ in range(5):
    new_data = sample_tensors_from_dict(toplayerdict, sample_size)

    ## STORE AS JSON FILE
    data_dict = dict_to_json(new_data)

    # CALCULATE MANIFOLD GEOMETRY
    geometry = compute_geometry(data_dict)
    dists, dists_norm, dsvds, bias, signal = process_geometry(geometry)
    for metric, data in zip(["Distances", "Normalized Distances", "Dsvds (Participation Ratio)", "Biases", "Signals"], 
                            [dists, dists_norm, dsvds, bias, signal]):
        results[metric]["run"].append(data)

averaged_results = {}
for metric, data in results.items():
    averaged_results[metric] = np.mean([np.mean(v) for v in data["run"]], axis=0)

# Print results
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(averaged_results)

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