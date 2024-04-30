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
from utils import get_embedding_dict, dict_to_json, compute_geometry, process_geometry, convert_to_serializable, sample_tensors_from_dict, plot_dsvds
from collections import defaultdict
import matplotlib.pyplot as plt


thresholds = {5: 'top_5_words', 100: 'top_100_words', 300: 'top_300_words'}
results = {key: [] for key in thresholds.values()}
SAMP_SIZE = 76
LOOPS = 100

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


for r in range(LOOPS):
    new_data = sample_tensors_from_dict(toplayerdict, SAMP_SIZE)

    ## STORE AS JSON FILE
    data_dict = dict_to_json(new_data)

    ## CALCULATE MANIFOLD GEOMETRY
    geometry = compute_geometry(data_dict)
    dists, dists_norm, dsvds, bias, signal = process_geometry(geometry)
    for threshold_name, values in dsvds.items():
        results[threshold_name].append(values)

## AVERAGE RESULTS
averaged_results = {}
for key, arrays in results.items():
    averaged_results[key] = np.mean(np.array(arrays), axis=0)
print("Averaged Dsvds (Participation Ratio) for each threshold:")
for key, avg in averaged_results.items():
    print(f"{key}: {avg}")

## GRAPH RESULTS
plot_dsvds(averaged_results, '/n/home09/lschrage/projects/llama/sompolinsky-research/Dsvds_Participation_Ratio_Plot.png')
