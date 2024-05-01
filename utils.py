from transformers import LlamaTokenizer, LlamaForCausalLM
import seaborn as sns
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
import torch

def get_embedding_dict(thresholds, inputs_dict, model, tokenizer):
    toplayerdict = {}
    for key in inputs_dict:
        print(f"Starting: {key}")
        toplayer = {thresholds[thresh]: [] for thresh in thresholds}
        for input in inputs_dict[key]:
            tokens = tokenizer.encode(input, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(tokens)
                predictions = outputs[0]
            next_token_candidates_tensor = predictions[0, -1, :]
            for thresh in thresholds:
                topk_candidates_indexes = torch.topk(next_token_candidates_tensor, thresh).indices.tolist()
                topk_candidates_tokens = [tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
                topk_lower = [x.lower() for x in topk_candidates_tokens]
                if str(key) in topk_lower:
                    curr_toplayer = outputs[2][-1][0, -1, :]
                    toplayer[thresholds[thresh]].append(curr_toplayer)  
        toplayerdict[key] = toplayer
    
    print("\nKeys in the top layer dictionary:", toplayerdict.keys())
    for key in toplayerdict:
        print(f"\nLengths for '{key}':")
        for threshold in thresholds.values():
            print(f"{threshold}: {len(toplayerdict[key][threshold])}")

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
    msr = {}

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

        msr[key] = np.array(MSR[key])

    return dists, dists_norm, dsvds, bias, signal, msr

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]  
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}  
    else:
        return obj 

def sample_tensors_from_dict(data, num_to_sample):
    def sample_tensors(tensors, num):
        if len(tensors) < num:
            raise ValueError("Not enough tensors to sample the requested number.")
        return random.sample(tensors, num)
    new_data = {}
    for key, thresholds in data.items():
        new_data[key] = {}
        for threshold, tensor_list in thresholds.items():
            try:
                sampled_tensors = sample_tensors(tensor_list, num_to_sample)
                new_data[key][threshold] = sampled_tensors
            except ValueError as e:
                print(f"Error for {key}, {threshold}: {e}")
                new_data[key][threshold] = None  
    return new_data

def average_results(results):
    averaged_results = {}
    for key, arrays in results.items():
        averaged_results[key] = np.mean(np.array(arrays), axis=0)
    return averaged_results

def plot_data(data, save_path, value):
    fig, ax = plt.subplots(figsize=(12, 8))
    thresholds = list(data.keys())
    indices = np.arange(len(data[thresholds[0]]))  
    bar_width = 0.25  
    colors = ['#6aabd1', '#b6d957', '#ef8354']
    for i, threshold in enumerate(thresholds):
        bar_positions = indices + i * bar_width
        ax.bar(bar_positions, data[threshold], width=bar_width, label=threshold.replace('_', ' ').title(), color=colors[i])
    ax.set_xlabel('Concepts', fontsize=14, fontweight='bold')
    ax.set_ylabel(value, fontsize=14, fontweight='bold')
    ax.set_title(f'{value} for Different Thresholds', fontsize=16, fontweight='bold')  
    ax.set_xticks(indices + bar_width * (len(thresholds) - 1) / 2) 
    ax.set_xticklabels(['Dog', 'Apple', 'Pen'], fontsize=12)
    ax.legend(title='Thresholds', title_fontsize='13', fontsize='11', loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    plt.savefig(save_path, format='png', bbox_inches='tight')

'''
def plot_signal (data, save_path):
    labels = ['dog', 'apple', 'pencil']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, matrix) in zip(axes, data.items()):
        sns.heatmap(matrix, annot=True, ax=ax, cmap="Reds")
        ax.set_title(title)
        ax.set_xticklabels(labels)  
        ax.set_yticklabels(labels)  

    plt.savefig(save_path, format='png', bbox_inches='tight')
'''
def plot_signal(data, save_path):
    labels = ['dog', 'apple', 'pencil']
    fig, axes = plt.subplots(1, len(data), figsize=(15, 5))  # Adjust the number of subplots dynamically
    
    for ax, (title, matrix) in zip(axes.flatten(), data.items()):
        if matrix.ndim == 3:  # Check if the matrix is 3D
            matrix = matrix.squeeze()  # Reduce it to 2D if it is only a single 3D matrix (e.g., shape (1, 3, 3))
        
        if matrix.shape == (1, 3, 3):  # Additional check if it's still wrapped in an extra dimension
            matrix = matrix[0]  # Select the first matrix in the set
        
        sns.heatmap(pd.DataFrame(matrix), annot=True, ax=ax, cmap="Reds")
        ax.set_title(title)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.close()