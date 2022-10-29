import os
import argparse
import torch
from tqdm import tqdm
import bz2
import pickle
import _pickle as cPickle

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("model_dreambooth", type=str, help="Path to dreambooth model")
parser.add_argument("model_base", type=str, help="Path to base model")
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument('--nocompress', help="Should the dehydrated model be compressed", action='store_true')
args = parser.parse_args()

# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + ".pbz2", "w") as f:
        cPickle.dump(data, f)

print("Loading Dreambooth model...")
model_dreambooth = torch.load(args.model_dreambooth)
print("Loading base model...")
model_base = torch.load(args.model_base)
theta_dreambooth = model_dreambooth["state_dict"]
theta_base = model_base["state_dict"]
theta_diff = {}

output_file = f'{args.output}'

print("Dehydrating model...")
for key in tqdm(theta_dreambooth.keys(), desc="Subtracting keys"):
    if "model" in key and key in theta_base:
        # theta_dreambooth[key] = theta_dreambooth[key] - theta_base[key]
        if not torch.equal(theta_dreambooth[key], theta_base[key]):
            theta_diff.update({key: (theta_dreambooth[key] - theta_base[key])})

if args.nocompress:
    print("Saving uncompressed dehydrated model...")
    torch.save({"state_dict": theta_diff}, output_file)
else:
    print('Saving compressed dehydrated model...')
    compressed_pickle(output_file, theta_diff)
    # compressed_pickle(output_file, theta_dreambooth)

print("Done!")
