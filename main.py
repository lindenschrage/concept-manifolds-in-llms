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
from utils import get_embedding_dict, dict_to_json, compute_geometry, process_geometry, convert_to_serializable, sample_tensors_from_dict, plot_data, average_results, plot_signal
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()
access_token = os.getenv('ACCESS_TOKEN')

thresholds = {5: 'top_5_words', 100: 'top_100_words', 300: 'top_300_words'}
SAMP_SIZE = 76
LOOPS = 10

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
    dists, dists_norm, dsvds, bias, signal, msr = process_geometry(geometry)

    ## CALCULATE DSVD, MSR FOR KEY, THRESHOLD
    dsvds_results = {key: [] for key in thresholds.values()}
    msr_results = {key: [] for key in thresholds.values()}
    dists_norm_results = {key: [] for key in thresholds.values()}

    for threshold_name, values in dsvds.items():
        dsvds_results[threshold_name].append(values)

    for threshold_name, values in msr.items():
        msr_results[threshold_name].append(values)

    for threshold_name, values in dists_norm.items():
        dists_norm_results[threshold_name].append(values)

## AVERAGE RESULTS
averaged_dsvds_results = average_results(dsvds_results)
averaged_msr_results = average_results(msr_results)
averaged_dists_norm_results = average_results(dists_norm_results)

## PRINT DIMENSTIONALITY
print("Averaged Dsvds (Participation Ratio) for each threshold:")
for key, avg in averaged_dsvds_results.items():
    print(f"{key}: {avg}")

## PRINT MSR
print("Averaged MSR for each threshold:")
for key, avg in averaged_msr_results.items():
    print(f"{key}: {avg}")

## PRINT DISTANCES
print("Averaged dists_norm for each threshold:")
for key, avg in dists_norm_results.items():
    print(f"{key}: {avg}")

## GRAPH RESULTS
plot_data(averaged_dsvds_results, '/n/home09/lschrage/projects/llama/sompolinsky-research/Dsvds_Participation_Ratio_Plot.png', 'Dsvds (Participation Ratio)')

plot_data(averaged_msr_results, '/n/home09/lschrage/projects/llama/sompolinsky-research/MSR_Plot.png', 'Mean Squared Radius')

plot_signal(dists_norm_results, '/n/home09/lschrage/projects/llama/sompolinsky-research/Signal_Plot.png')

