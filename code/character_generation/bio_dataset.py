from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class BioDataset(Dataset):

    def __init__(self, control_code, truncate = False, gpt2_type = 'gpt2', max_length = 1024):
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.bios = []
