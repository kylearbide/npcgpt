import numpy as np 
import pandas as pd 
import re 
import os
from tqdm import tqdm, trange 
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
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
    ''' Dynamically batches data into a single tensor of length max_seq_len. Due to the size of 
    GPT2 and the data, this is done to more effeciently use computation resources instead of 
    padding single tensors with 0s to make them of the same length. 

    Parameters
    ----------
    new_tensor : torch.Tensor
        The new data to be batched.
    packed_tensor : torch.Tensor or None
        The existing packed_tensor to add the new_tensor to. 
    max_seq_len : int 
        Maximum sequence length for the packed tensors. 

    Returns
    -------
    (torch.Tensor, bool, torch.Tensor or None)
        The new packed tensor. Boolean value indicating whether the packing was successful.
        If the packing was not successful, the tensor that could not be packed, None otherwise.
    '''
    if packed_tensor is None:
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
    ''' Training loop. 

    Parameters
    ----------

    Returns 
    -------
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

        # debugging
        # count = 0

        for batch_idx, (idx, entry) in tqdm(enumerate(train_dataloader)):

            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, max_seq_len)

            # debugging 
            # print(f'\nBatch {batch_idx}----------------------------------------')
            # print(f'idx: {idx}')
            # print(f'Entry shape: {entry.shape}')
            # print(f'Entry: {entry}')
            # print(f'Input tensor shape: {input_tensor.shape}')
            # print(f'Input tensor: {input_tensor}')
            # print(f'Input tensor[:,1:]: {input_tensor[:, 1:]}')
            # print(f'Input tensor[:,1:] shape: {input_tensor[:, 1:].shape}')
            # print(f'Carry on: {carry_on}')
            # print(f'Remainder: {remainder}')
            # if count == 3:
            #     break 
            # count += 1

            if carry_on and ((batch_idx + 1) != len(train_dataloader)):
                continue
            
            input_tensor = input_tensor.to(DEVICE)
            outputs = model(input_tensor, labels = input_tensor)
            loss = outputs[0]
            loss.backward()

            if (((batch_idx + 1) == 0) or (batch_idx + 1) == len(train_dataloader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulate[batch_idx] = 1
            
            input_tensor = remainder
            losses.append(loss.detach().cpu().item())
        
        print(f'Average loss: {np.mean(losses)} in epoch {epoch}')

        if save_model_on_epoch:
            print(f'Saving epoch {epoch} state')
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f'{output_prefix}-{epoch}.pt')
            )
        
        # debugging 
        # if count >= 1:
        #     break
    
    return model 
            
model = train(dataset, model, tokenizer)