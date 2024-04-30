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
from utils import get_embedding_dict, dict_to_json, compute_geometry, process_geometry, convert_to_serializable, sample_tensors_from_dict

thresholds = {3: 'top_3_words', 10: 'top_100_words', 200: 'top_200_words'}
sample_size = 60

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
new_data = sample_tensors_from_dict(toplayerdict, sample_size)

## STORE AS JSON FILE
data_dict = dict_to_json(new_data)

# CALCULATE MANIFOLD GEOMETRY
geometry = compute_geometry(data_dict)
dists, dists_norm, dsvds, bias, signal = process_geometry(geometry)

pp = pprint.PrettyPrinter(indent=4)
print("\nDsvds (Participation Ratio):")
pp.pprint(dsvds)

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