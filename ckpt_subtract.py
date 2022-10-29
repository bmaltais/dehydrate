import os
import argparse
import torch
from tqdm import tqdm
import bz2
import pickle
import _pickle as cPickle

parser = argparse.ArgumentParser(description="Merge two models")
parser.add_argument("dreambooth_model", type=str, help="Path to model 0")
parser.add_argument("base_model", type=str, help="Path to model 1")
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument('--nocompress', help="Should the dehydrated model be compressed", action='store_true')
args = parser.parse_args()

# Pickle a file and then compress it into a file with extension


def compressed_pickle(title, data):
    with bz2.BZ2File(title + ".pbz2", "w") as f:
        cPickle.dump(data, f)


print("Loading Dreambooth model...")
model_0 = torch.load(args.dreambooth_model)
print("Loading base model...")
model_1 = torch.load(args.base_model)
theta_0 = model_0["state_dict"]
theta_1 = model_1["state_dict"]
theta_diff = {}

output_file = f'{args.output}'

print("Dehydrating model...")
for key in tqdm(theta_0.keys(), desc="Subtracting keys"):
    if "model" in key and key in theta_1:
        # theta_0[key] = theta_0[key] - theta_1[key]
        if not torch.equal(theta_0[key], theta_1[key]):
            theta_diff.update({key: (theta_0[key] - theta_1[key])})

if args.nocompress:
    print("Saving uncompressed dehydrated model...")
    torch.save({"state_dict": theta_diff}, output_file)
else:
    print('Saving compressed dehydrated model...')
    compressed_pickle(output_file, theta_diff)
    # compressed_pickle(output_file, theta_0)

print("Done!")
