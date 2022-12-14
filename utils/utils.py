import torch
import numpy as np
import random
import argparse
import pickle
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AdamW, default_data_collator, EvalPrediction
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import load_metric,load_dataset
from tqdm import tqdm


def set_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', required=True, type=float)
    parser.add_argument('--mode', required=True, type=str)
    args = parser.parse_args()

    return args


def get_aligned_tokens(langauge_ids):
    aligned_tokens = {}
    for _lg in langauge_ids:
        with open("data/aligned-tokens-en-{}".format(_lg), "rb") as fr:
            _aligned_tokens = pickle.load(fr)
            for key, val in tqdm(_aligned_tokens.items()):
                if key not in aligned_tokens:
                    aligned_tokens[key] = []
                aligned_tokens[key] += val[:]
    return aligned_tokens
