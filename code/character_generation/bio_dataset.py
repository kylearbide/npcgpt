import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class BioDataset(Dataset):
    ''' Dataset class for the sample character bios. 

    Attributes
    ----------
    tokenizer : GPT2Tokenizer
        GPT2 Tokenizer.
    bios : list
        List of sample bios.
    bios_count : int
        Number of sample bios in the dataset.
    
    Methods
    -------
    __init__(bios, gpt2_type, max_length, truncate, **kwargs): 
        Constructor.
    __len__(): 
        Returns the number of sample bios in the dataset.
    __getitem__(idx): 
        Return a specific record from the dataset.
    '''

    def __init__(self, bios : pd.Series, gpt2_type = 'gpt2', max_length = 1022, truncate = 0, **kwargs):
        ''' Constructor. 

        Parameters
        ----------
        bios : pd.Series
            Pandas series of sample bios to populate the dataset.
        gpt2_type : str
            Gpt2 type to load in from the transformers module.
        max_length : int
            Specifies the maximum length of a token sequence after tokenization.
        truncate : int
            Whether to truncate dataset to a specific size. 
        **kwargs : dict, optional
            Arbitrary keyword arguments. 
        '''
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type, **kwargs)
        self.bios = []

        for idx, text in bios.iteritems():
            
            if (truncate > 0) and (idx == truncate):
                break 
        
            bio_tokens = self.tokenizer.tokenize(text)
            if len(bio_tokens) > max_length:
                is_start = np.random.randint((len(bio_tokens) - max_length))
                bio_tokens = bio_tokens[is_start:(is_start + max_length)]
            
            self.bios.append(torch.LongTensor([
                self.tokenizer.bos_token_id,
                *self.tokenizer.convert_tokens_to_ids(bio_tokens),
                self.tokenizer.eos_token_id])
            )
        
        self.bios_count = len(self.bios)
    
    def __len__(self):
        ''' Returns the number of bios in the dataset. 

        Returns 
        -------
        int
            The number of bios in the dataset. 
        '''
        return self.bios_count 
    
    def __getitem__(self, idx):
        ''' Returns the item at the given index. 

        Parameters
        ----------
        idx : int 
            Specified index of the sample to be returned. 

        Returns 
        -------
        (int, torch.Tensor)
            Index of the item to be returned. Bio at the specified index. 
        '''
        return idx, self.bios[idx]