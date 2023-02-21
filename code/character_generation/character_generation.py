import numpy as np 
import pandas as pd 
import re 
from tqdm import tqdm, trange 
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from bio_dataset import BioDataset

delimiter = re.compile(r'\s+')
TEST_SET_SIZE = 0.15
MODEL_TYPE = 'gpt2'
EPOCHS = 5

cuda = torch.cuda.is_available()
if cuda:
    DEVICE = 'cuda'
    torch.cuda.manual_seed(42)

### read in and prepare data 
character_bios = pd.read_csv('data/character_bios.csv')

# calculate length of bios
character_bios['bio_word_count'] = character_bios['bio'].str.split().str.len() 

# tokenize bios into new column in dataframe and then overwrite original bio strings with simplified versions
tmp = character_bios.bio.str.split(delimiter)
def notempty(x): return (len(x) > 0)
tmp = tmp.apply(lambda x : list(filter(notempty, x)))
character_bios.insert(character_bios.shape[1], 'bio_tokens', tmp)
character_bios['bio'] = character_bios.bio_tokens.apply(' '.join)
del tmp 

# filter out bios with too few (<35) or too many words (>80)
character_bios = character_bios[(character_bios['bio_word_count'] >= 35) & (character_bios['bio_word_count'] <= 80)].reset_index(drop = True)

### create small test set 
test_set = character_bios.sample(n = int(character_bios.shape[0] * TEST_SET_SIZE), random_state = 8)
train_set = character_bios.drop(index = test_set.index).copy() 
test_set.reset_index(drop = True, inplace = True)
train_set.reset_index(drop = True, inplace = True)
# check everything worked correctly 
assert character_bios.shape[0] == (train_set.shape[0] + test_set.shape[0])
# for the test set only, keep last 15 words in a new column and remove from original bio column
test_set.insert(test_set.shape[1], 'true_bio_end', test_set.bio_tokens.str[-15:].apply(' '.join))
test_set.loc['bio'] = test_set.bio_tokens.str[:-15].apply(' '.join)

### train set
dataset = BioDataset(train_set.bio, gpt2_type = MODEL_TYPE)
tokenizer = dataset.tokenizer 
model = GPT2LMHeadModel.from_pretrained(MODEL_TYPE)

### model and packing code

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    ''' 
    '''
    if pack_tensor is None:
        return new_tensor, True, None 
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim = 1)
        return packed_tensor, True, None 

def train(dataset, model, tokenizer,
        batch_size = 16, epochs = EPOCHS,
        learning_rate = 2e-5, max_seq_len = 768,
        warmup_steps = 5000, gpt2_type = MODEL_TYPE,
        output_dir = '.', output_prefix = 'character_generation',
        test_mode = False, save_model_on_epoch = False):
    '''
    '''
    
    # send model to the gpu (if available) and then set model to training mode
    model = model.to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr = learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = -1)

    # pin_memory = True 
    train_dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

    for epoch in range(epochs):

        print(f'Epoch: {epoch + 1}/{EPOCHS}')
        
        optimizer.zero_grad()
        loss = 0
        input_tensor = None 
        losses = []
        accumulate = torch.zeros(len(train_dataloader), dtype = torch.bool)

        for batch_idx, (idx, entry) in tqdm(enumerate(train_dataloader)):
            pass
