import numpy as np 
import pandas as pd 
import sys
import random 
import torch 
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import spacy 

gpt2_type = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)

cuda = torch.cuda.is_available()
if cuda:
    DEVICE = 'cuda'
    torch.cuda.manual_seed(42)

# read in seed data 
seed_data = pd.read_csv('data/bio_seed_data.csv')

# number of names
num_names = seed_data.name.shape[0]

# get adjectives in a list 
adjs = list(seed_data.adjective.dropna())
# number of adjectives 
num_adjs = len(adjs)

nlp = spacy.load('en_core_web_md')

# load in model 
model = torch.load('code/models/character_bio_generation.pt')
model = model.to(DEVICE)

def generate(
        model, tokenizer,
        prompt, bio_length = 70,
        top_p = 0.8, temperature = 1.15):

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

def single_bio():

    # pick a random name 
    name_seed = random.randint(0, num_names - 1)

    adj_one = ''
    adj_two = ''
    adj_one_idx = -1
    adj_two_idx = -1
    repick = True 

    while repick:
        adj_one_idx = random.randint(0, num_adjs - 1)
        adj_two_idx = random.randint(0, num_adjs - 1)
        adj_one = adjs[adj_one_idx]
        adj_two = adjs[adj_two_idx]

        if adj_one_idx == adj_two_idx:
            continue 

        adj_one_nlp = nlp(adj_one)
        adj_two_nlp = nlp(adj_two)
        sim_score = adj_one_nlp.similarity(adj_two_nlp)

        # print(adj_one)
        # print(adj_two)
        # print(sim_score)
        
        if sim_score >= 0.2:
            repick = False 
        else:
            continue 

    print(generate(model, tokenizer, f'{seed_data.name.iloc[name_seed]} is a {adj_one.lower()} and {adj_two.lower()}'))

def batch_bios(num_bios):

    bios = []

    for _ in range(num_bios):
        adj_one = ''
        adj_two = ''
        adj_one_idx = -1
        adj_two_idx = -1 
        name_seed = random.randint(0, num_names - 1)
        repick = True 

        while repick:
            adj_one_idx = random.randint(0, num_adjs - 1)
            adj_two_idx = random.randint(0, num_adjs - 1)
            adj_one = adjs[adj_one_idx]
            adj_two = adjs[adj_two_idx]

            if adj_one_idx == adj_two_idx:
                continue 

            adj_one_nlp = nlp(adj_one)
            adj_two_nlp = nlp(adj_two)
            sim_score = adj_one_nlp.similarity(adj_two_nlp)

            # print(adj_one)
            # print(adj_two)
            # print(sim_score)
            
            if sim_score >= 0.2:
                repick = False 
            else:
                continue
        
        bios.append(generate(model, tokenizer, f'{seed_data.name.iloc[name_seed]} is a {adj_one.lower()} and {adj_two.lower()}'))
    
    bios_df = pd.DataFrame(columns = ['bio'], data = bios)
    bios_df.to_csv('./data/generated_bios.csv')

def main(args):
    if len(args) < 2: 
        print('Not enough arguments')
        sys.exit(1)
    try:
        arg = int(args[1])
    except:
        print('Invalid argument')
        sys.exit(1)
    if arg == 1:
        single_bio()
    else:
        batch_bios(arg)

if __name__ == '__main__':
    main(sys.argv)