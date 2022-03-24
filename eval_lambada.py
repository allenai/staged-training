import argparse
import math
import os
import time

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from tqdm import trange


from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    default_data_collator,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader, Dataset, Subset

model_name = 'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='lambada_test.jsonl', help='location of lambada dataset')
parser.add_argument('--batch', type=int, default=4, help='batch size')
parser.add_argument('--max-batches', type=int, default=0, help='batch size')
parser.add_argument('--ignore-fragments',  action='store_true', help="Whether to run training.")
parser.add_argument('--preprocess',  action='store_true', help="strip quotes")
parser.add_argument('--jeff_suggestion',  action='store_true', help="use jeff's suggestion of prepending \n to each example")
parser.add_argument('--dryrun',  action='store_true', help="test preprocessing pipeline")
parser.add_argument('--checkpoint-file', default=None, help='location of lambada dataset')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device


model_name = 'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name)

checkpoint_file = args.checkpoint_file

if checkpoint_file is not None:
    state_dict = torch.load(checkpoint_file)
    state_dict = state_dict['state_dict']
    state_dict = {k.replace("model.", ""): p for k, p in state_dict.items()}

    config = GPT2Config.from_pretrained(model_name)

    # Guess model size.
    print(state_dict.keys)
    config.n_embd = state_dict["transformer.wte.weight"].shape[1]
    config.n_layer = len(
        [key for key in state_dict if key.endswith("mlp.c_proj.bias")]
    )
    print(
        f"Adjusting hidden_size to {config.n_embd}, num_layers to {config.n_layer}"
    )
    if config.n_embd == 1536 or config.n_embd == 3072:
        config.n_head = 16
    else:
        config.n_head = 8
        #config.n_head = 12
    config.n_positions == 1024
    if "alibi" in checkpoint_file:
        config.alibi_embeddings = True
    if "rotary" in checkpoint_file:
        config.rotary_embeddings = True
    # Initialize model.
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict, strict=True)
else:
    model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)


def argmax(t):
    return int(torch.argmax(t).item())

# from https://github.com/openai/gpt-2/issues/131#issuecomment-492786058
def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    return '\n'+text.strip()

def score_batch(batch):
    """Return number of last-word mismatches in a batch."""
    batch_encoded = []
    lengths = []
    fragments = []
    for line in batch:
        line = line.strip()
        if args.jeff_suggestion:
            line = '\n'+line
        line_encoded = enc.encode(line)
        encoded_last_word = enc.decode(line_encoded[-1:]).strip()
        actual_last_word = line.split()[-1].strip()
        if encoded_last_word != actual_last_word:
            fragments.append(True)
        else:
            fragments.append(False)
        batch_encoded.append(line_encoded)

    # array is ragged, so pad to turn into rectangular tensor
    max_len = max(len(encoded) for encoded in batch_encoded)
    batch_padded = []
    for encoded in batch_encoded:
        batch_padded.append(encoded+[0]*(max_len - len(encoded)))
        lengths.append(len(encoded))

    batch_padded = torch.tensor(batch_padded)
    batch_padded = batch_padded.to(device)
    if args.dryrun:
        return 0, 1
    
    # logits, presents = model(batch_padded)
    outputs = model(batch_padded)
    logits = outputs.logits
    errors = 0
    total = 0
    for i in range(args.batch):
        # break on small last batch
        if i >= len(batch_padded):
            break
        last_idx = lengths[i]-1
        observed = batch_encoded[i][last_idx]
        predicted = argmax(logits[i][last_idx-1])
        if args.ignore_fragments and fragments[i]:
            continue
        total+=1
        errors += 0 if (observed == predicted) else 1

    return errors, total


def main():
    ds_raw = open(f'{args.path}').read()
    if args.preprocess:
        ds_raw = preprocess(ds_raw)
        
    ds = ds_raw.strip().split('\n')

    # special handling for jsonl file
    lines = []
    if args.path.endswith('.jsonl'):
        # special handling for file from Jeff
        for line in ds:
            #            candidate1 = eval(line)['text']
            #            lines.append(candidate1)
            candidate2 = line[len('{"text": "'):-len('"}')]
            candidate2 = f'''"""{candidate2}"""'''
            lines.append(eval(candidate2))

            #            lines.append(eval(line))
            #print(line)
            #            break
            #            print(line)
            #            eprint(lines[-1])
        ds = lines
    data_loader = DataLoader(ds, batch_size=args.batch, shuffle=False)
    
    errors = 0
    total = 0
    for batch in tqdm.tqdm(data_loader):
        errors_batch, total_batch = score_batch(batch)
        errors += errors_batch
        total += total_batch
        # if args.max_batches and i>=args.max_batches-1:
        #     break

    print("Accuracy: %.4f"%(1-errors/total,))
        

if __name__=='__main__':
    main()
