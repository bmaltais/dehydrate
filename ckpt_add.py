import os
import argparse
import torch
from tqdm import tqdm
import bz2
import pickle
import _pickle as cPickle
import pathlib

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_dehydrated", type=str, help="Path to dehydrated model")
parser.add_argument("model_base", type=str, help="Path to base model")
parser.add_argument("--str", type=float, help="Strength of the rehydration (-0.05..0.05)", default=0, required=False)
parser.add_argument("--output", type=str, help="Output file name", default="merged", required=False)
args = parser.parse_args()

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data

print(pathlib.Path(args.model_dehydrated).suffix)
if pathlib.Path(args.model_dehydrated).suffix == ".pbz2":
    print("Loading and decompressing dehydrated model...")
    theta_dehydrated = decompress_pickle(args.model_dehydrated)  # torch.load(args.model_dehydrated)
else:
    print("Loading dehydrated model...")
    model_dehydrated = torch.load(args.model_dehydrated)
    theta_dehydrated = model_dehydrated["state_dict"]
print("Loading base model...")
model_base = torch.load(args.model_base)
theta_base = model_base["state_dict"]
str = args.str

output_file = f'{args.output}'

print("Hydrating model...")
for key in tqdm(theta_dehydrated.keys(), desc="Stage 1/2: merge common keys"):
    if "model" in key and key in theta_base:
        theta_dehydrated[key] = theta_dehydrated[key] * (1 + str) + theta_base[key] * (1 - str)
        # if str > 0.:
        #     theta_dehydrated[key] = torch.where(theta_dehydrated[key] > 1., 1., theta_dehydrated[key])
        # if str < 0.:
        #     theta_dehydrated[key] = torch.where(theta_dehydrated[key] < -1., -1., theta_dehydrated[key])

for key in tqdm(theta_base.keys(), desc="Stage 2/2: add missing keys"):
    if "model" in key and key not in theta_dehydrated:
        theta_dehydrated[key] = theta_base[key]

print("Saving hydrated model...")

torch.save({"state_dict": theta_dehydrated}, output_file)

print("Done!")
