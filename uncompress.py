import os
import argparse
import torch
from tqdm import tqdm
import bz2
import pickle
import _pickle as cPickle
import pathlib

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_0", type=str, help="Path to compress dehydrated model")
parser.add_argument("--output", type=str, help="Output file name for the uncompressed dehydrated model", default="uncompressed", required=False)
args = parser.parse_args()

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data

print(pathlib.Path(args.model_0).suffix)
if pathlib.Path(args.model_0).suffix == ".pbz2":
    print("Loading and decompressing dehydrated model...")
    theta_0 = decompress_pickle(args.model_0)  # torch.load(args.model_0)
else:
    print("Not a compressed dehydrated model...")

output_file = f'{args.output}'

print("Saving uncompressed dehydrated model...")

torch.save({"state_dict": theta_0}, output_file)

print("Done!")
