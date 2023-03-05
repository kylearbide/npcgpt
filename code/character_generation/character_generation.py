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
EPOCHS = 15

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
# for the test set only, keep last 35 words in a new column and remove from original bio column
test_set.insert(test_set.shape[1], 'true_bio_end', test_set.bio_tokens.str[-35:].apply(' '.join))
test_set.loc[:,'bio'] = test_set.bio_tokens.str[:-35].apply(' '.join)

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
        # eos and bos tokens are the same, only need one between sequences
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim = 1)
        return packed_tensor, True, None 

def train(dataset, model, tokenizer,
        batch_size = 16, epochs = EPOCHS,
        learning_rate = 2e-5, max_seq_len = 768,
        warmup_steps = 200, output_dir = 'code/models/', 
        output_prefix = 'character_generation',
        save_model_on_epoch = False):
    ''' Training loop. 

    Parameters
    ----------
    dataset : torch.Dataset 
        Dataset to train on. 
    model : transformer.GPT2LMHeadModel
        Model to train. 
    tokenizer : transformers.GPT2Tokenizer
        Tokenizer for the data. 
    batch_size : int 
        Batch size when looping through mini batches.
    epochs : int
        Number of epochs.
    learning_rate : float
        Initial learning rate. 
    max_seq_len : int
        Maximum sequence length for the packed tensors. 
    warmup_steps : int 
        Number of warump steps for the scheduler. 
    output_dir : str
        Output directory to save the model in (if applicable).
    output_prefix : str
        Prefix for the saved model filename.
    save_model_on_epoch : bool 
        Whether to save the model after each epoch. 

    Returns 
    -------
    transformer.GPT2LMHeadModel 
        The fine tuned GPT2 model. 
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

            if (((batch_idx + 1) % batch_size == 0) or (batch_idx + 1) == len(train_dataloader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulate[batch_idx] = 1
            
            input_tensor = remainder
            losses.append(loss.detach().cpu().item())
        
        print(f'Average loss: {np.mean(losses)} in epoch {epoch + 1}')

        if save_model_on_epoch:
            print(f'Saving epoch {epoch + 1} state')
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f'{output_prefix}-{epoch}.pt')
            )
        
        # debugging 
        # if count >= 1:
        #     break
    
    return model 
            
model = train(dataset, model, tokenizer, save_model_on_epoch = False)
torch.save(model, 'code/models/character_bio_generation.pt')

def test(
        model, tokenizer,
        prompt, bio_length = 60,
        top_p = 0.8, temperature = 1.00):
    ''' Test loop. 

    Parameters
    ----------
    model : transformer.GPT2LMHeadModel
        Model to test.
    tokenizer : transformers.GPT2Tokenizer
        Tokenizer for the data.
    prompt : str
        The beginning of the test bio to test on.
    bio_length : int 
        The maximum number of words to make the bio. 
    top_p : float 
        Percentage value controlling the diversity of the generated text. Once the 
        cumulative distribution is generated, it is cut off once the CDF exceeds top_p. 
    temperature : float 
        Used to control the randomness of the generated tokens. 
    
    Returns
    -------
    str
        The generated string. 
    '''

    model.eval()

    # filter value to eliminate everything that falls outside our top_prob 
    filter = -float('inf')

    with torch.inference_mode():

        finished = False 

        # tokenize the prompt 
        prompt_toks_ids = torch.tensor(tokenizer.encode(prompt), device = model.device).unsqueeze(0)
        # number of tokens in the prompt 
        num_token_ids = prompt_toks_ids.shape[-1]

        for word in range(bio_length):

            # get prediction for the next word 
            outputs = model(prompt_toks_ids, labels = prompt_toks_ids).to_tuple()
            # unpack the output
            loss = outputs[0]
            logits = outputs[1] 
            # test
            hidden_state = outputs[2]
            # slice just the predictions for the last word and then divide by the temperature  
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            # sort the logits for the most likely first 
            sorted_logits, sorted_indices = torch.sort(logits, descending = True) 
            # apply the softmax function to the logits to convert them to probabilties
            # then apply the cumulative sum function along the column 
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

            # creates a boolean tensor to indicate which indices to set to the filter value 
            remove_indices = cum_probs > top_p
            # we never want to remove the first token as to not have an empty tensor causing an error 
            # shift the values to the right (last indices will always be greater than top_p since it equals 1)
            remove_indices[..., 1:] = remove_indices[..., :-1].clone() 
            # set the first indices to False (0) so it will never get dropped 
            remove_indices[..., 0] = 0 
            # use `remove_indices` as a boolean mask on the sorted indices 
            indices_to_remove = sorted_indices[remove_indices]
            # replace the selected logits to be removed with the filter value (-inf)
            logits[:, indices_to_remove] = filter 

            # after the correct filter values have been assigned, re-compute the probabilities and then sample one
            next_token = torch.multinomial(F.softmax(logits, dim = -1), num_samples = 1)
            # concatenate the new predicted token id to the original encoded prompt 
            prompt_toks_ids = torch.cat((prompt_toks_ids, next_token), dim = 1)

            # boolean to determine if the bio has finished or not 
            finished = (next_token.item() == tokenizer.eos_token_id)
            if finished:
                break 
        
        num_generated = (prompt_toks_ids.shape[-1] - num_token_ids)
        # print(f'sanity check: {num_generated == (word + 1)}')

        output_list = list(prompt_toks_ids.cpu().squeeze().numpy())
        # only grab the generated text 
        generated_list = output_list[-num_generated:]
        generated_text = f"{tokenizer.decode(generated_list)}{'' if finished else tokenizer.eos_token}"
    
    return generated_text

# generate bios for the test set 
generated_bios = [''] * test_set.shape[0]
for i in trange(test_set.shape[0], leave = False):
    generated_bios[i] = test(model, tokenizer, test_set.bio.iloc[i])

test_set.insert(test_set.shape[1], 'generated_bio', generated_bios)
test_set.to_csv('test.csv')