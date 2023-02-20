import numpy as np 
import pandas as pd 
from transformers import pipeline, set_seed
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# read in data 
character_bios = pd.read_csv('data/character_bios.csv')

