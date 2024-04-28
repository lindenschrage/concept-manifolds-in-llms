from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_embedding_dict(inputs_dict, model, tokenizer):
  for key in inputs_dict:
    print("Starting:", key)
    thresholds = {10: 'top_10_words', 50: 'top_50_words', 100: 'top_100_words'}
    toplayerdict = {}
    toplayer = {thresholds[thresh]: [] for thresh in thresholds}
    for input in inputs_dict[key]:
      print(input)
      tokens = tokenizer.encode(input, return_tensors="pt").to('cuda')
      with torch.no_grad():
        outputs = model(tokens)
        predictions = outputs[0]
      next_token_candidates_tensor = predictions[0, -1, :]
      for thresh in thresholds:
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, thresh).indices.tolist()
        topk_candidates_tokens = \
            [tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
        topk_lower = [x.lower() for x in topk_candidates_tokens]
        if str(key) in topk_lower:
          curr_toplayer = outputs[2][-1][0, -1, :]
          toplayer[thresholds[thresh]].append(curr_toplayer)  
    toplayerdict[key] = toplayer
  return toplayerdict


def dict_to_json(toplayerdict):
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
    return data_dict

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

def process_geometry(geometry):
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

    for key in centers.keys():
        matrix = squareform(pdist(centers[key]))
        dists[key] = matrix

        Rs_array = np.array(Rs[key])
        norm_factor = np.sqrt((Rs_array**2).mean())
        dists_norm[key] = matrix / norm_factor

        dsvds[key] = [np.sum(r**2)**2 / np.sum(r**4) for r in Rs_array]

        m = (Rs_array**2).sum(axis=1)
        bias[key] = m[:, None] / m - 1
        signal[key] = dists_norm[key]**2 + bias[key]

    return dists, dists_norm, dsvds, bias, signal


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]  
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}  
    else:
        return obj 
