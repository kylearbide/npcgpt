import numpy as np 
import pandas as pd 
import os 
import random 
import torch 
import torch.nn.functional as F
from transformers import GPT2Tokenizer

gpt2_type = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)

cuda = torch.cuda.is_available()
if cuda:
    DEVICE = 'cuda'
    torch.cuda.manual_seed(42)

# read in seed data 
seed_data = pd.read_csv('data/bio_seed_data.csv')

# number of seed prompts 
num_seeds = seed_data.shape[0]

# pick a random seed 
seed = random.randint(0, num_seeds - 1)

# load in model 
model = torch.load('code/models/character_bio_generation.pt')
model = model.to(DEVICE)

def generate(
    model, tokenizer,
    prompt, bio_length = 60,
    top_p = 0.8, temperature = 1.15):
    '''
    '''

    model.eval()
    filter = -float('inf')

    with torch.inference_mode():

        finished = False 

        # tokenize the prompt 
        generated = torch.tensor(tokenizer.encode(prompt), device = model.device).unsqueeze(0)

        for word in range(bio_length):

            # get prediction for next word
            outputs = model(generated, labels = generated).to_tuple()
            logits = outputs[1]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending = True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

            remove_indices = cum_probs > top_p
            remove_indices[..., 1:] = remove_indices[..., :-1].clone() 
            remove_indices[..., 0] = 0
            indices_to_remove = sorted_indices[remove_indices]
            logits[:, indices_to_remove] = filter 

            next_token = torch.multinomial(F.softmax(logits, dim = -1), num_samples = 1)
            generated = torch.cat((generated, next_token), dim = 1)

            finished = (next_token.item() == tokenizer.eos_token_id)
            if finished:
                break 
        
        output_list = list(generated.cpu().squeeze().numpy())
        generated_text = f"{tokenizer.decode(output_list)}{'' if finished else tokenizer.eos_token}"

    return generated_text

print(generate(model, tokenizer, f'{seed_data.seed.iloc[seed]} a fun and outoing'))