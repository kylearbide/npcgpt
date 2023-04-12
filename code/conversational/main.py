from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transfer_learning_conversational import SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN, add_special_tokens, build_input_from_segments, get_dataset
# from build_raw_data import get_bios
from itertools import chain
import torch
import torch.nn.functional as F
import random
import pandas as pd
import nltk.data
import re
from argparse import ArgumentParser
import json 
import subprocess
import sys

device = 'cuda' if torch.cuda.is_available() else "cpu" 

tokenizer = GPT2Tokenizer.from_pretrained('code/conversational/models/dialoGPT')
model = GPT2LMHeadModel.from_pretrained('code/conversational/models/dialoGPT')
model.to(device)
add_special_tokens(model,tokenizer, ATTR_TO_SPECIAL_TOKEN)

with open('data/main.json', 'r') as f:
    data = json.load(f)

def generate_new_bio():
    def generate_bio(path):
        command = [sys.executable, path, str(1)]
        output = subprocess.check_output(command)
        return output.decode('utf-8')

    new_bio = generate_bio('code/character_generation/generate.py')
    data['regenerate'] = False
    data['persona'] = new_bio
    with open('data/main.json', 'w') as f:
        json.dump(data, f)

if data['regenerate'] == True:
    generate_new_bio()

default_persona = data['persona']

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

parser = ArgumentParser()
parser.add_argument("--persona", type=str, default=default_persona, help="Unformatted persona from Model 1.")
parser.add_argument("--temperature", type=float, default=0.7, help="Response temperature")
parser.add_argument("--min_length", type=float, default=5.0, help="Minimum length of model response")
parser.add_argument("--max_length", type=float, default=50, help="Maximum length of model response")
parser.add_argument("--no_sample", type=bool, default=True, help="tbh forgot what this does but I had it set to true")
parser.add_argument("--top_k", type=float, default=0.0, help="<=0: no filtering, >0: keep only top k tokens with highest probability.")
parser.add_argument("--top_p", type=float, default=0.0, help="<=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset whose total probability mass is greater than or equal to the threshold top_p.")
args = parser.parse_args()

def format_input(text:str) -> str:
    """
    Takes a raw text and formats for the model input. For dialogue and personality, this means all lowercase with a space before any punctuation.
    Args:
        text: raw text input, cam be dialogue or bio
    Returns:
        text: text formatted for the model
    """
    text = text.lower()
    text = re.sub('([.,!?()])', r' \1 ', text)
    text = re.sub('\s{2,}', ' ', text)
    text = text.strip()
    return(text)

def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)

def format_persona(persona:str) -> list:
     """
     Formats the persona for model consumption.
     """
     persona = persona.lower().split("\n",1)[0]
     persona = sent_tokenizer.tokenize(persona)
     persona = [format_input(sent) for sent in persona]
     persona = tokenize(persona)
     return(persona)

def top_filtering(logits, top_k=0., top_p=0.8, threshold=-float('Inf'), filter_value=-float('Inf')):
        """ 
        Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
    
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits

def sample_sequence(personality, history, tokenizer, model, max_length, min_length, no_sample, temp, top_k, top_p):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    current_output = []
    for i in range(max_length):
        instance =  build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], device=device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=device).unsqueeze(0)
        output = model(input_ids, token_type_ids=token_type_ids)
        logits = output.logits
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / temp
        logits = top_filtering(logits, top_k, top_p)
        probs = F.softmax(logits, dim=-1)
        
        prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
        if i < min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def main(persona = args.persona,
         temp = args.temperature,
         min_length = args.min_length,
         max_length = args.max_length,
         no_sample = args.no_sample,
         top_k = args.top_k,
         top_p = args.top_p
         ):
    

    # Format persona for model consumption
    persona = format_persona(args.persona)

    # Create the history, this is what gets fed into the model as a prompt
    history = []

    # Define if we want the bot to talk first, if yes leave this true
    bot_talks_first = True
    
    print("Conversation Starting...")
    
    talking = True

    while talking:
        if bot_talks_first:
            # Using this so the convo starts out with a model output, like in the game
            history.append(tokenizer.encode("tell me about yourself ."))
            bot_talks_first = False
            # Outputs the results if the bot is supposed to start the convo
            with torch.no_grad():
                output = sample_sequence(persona, history, tokenizer, model, max_length, min_length, no_sample, temp, top_k, top_p)
                out_text = tokenizer.decode(output, skip_special_tokens=True)
                history.append(output)
                print("CaptAIn: " + out_text)
        else:
            print("Talk, Restart, Return, or Stop: ")
            action = input().lower()
            if action == "talk":
                # Interface for interacting with the conversational model, keeps adding to the history unless `restart` is selected
                user_input = input("You: ")[6:]
                formatted_input = tokenizer.encode(format_input(user_input))
                history.append(formatted_input)
                with torch.no_grad():
                    output = sample_sequence(persona, history, tokenizer, model, max_length, min_length, no_sample, temp, top_k, top_p)
                    out_text = tokenizer.decode(output, skip_special_tokens=True)
                    history.append(output)
                    print("CaptAIn: " + out_text)
            elif action == "restart":
                # Resets the history, allows you to pick if you want to start of if you want the AI to start
                history = []
                bot_talks_first_request = input("Bot talks first? y/n: ")
                if bot_talks_first_request.lower() == "y":
                    bot_talks_first = True
            elif action == "return":
                # return last bot output
                return(tokenizer.decode(history[-1]))
            else:
                new_character = input('Should a new character persona be generated on next run? y/n: ')
                if new_character.lower() == 'y':
                    with open('data/main.json', 'r') as f:
                        main_json = json.load(f)
                    main_json['regenerate'] = True 
                    with open('data/main.json', 'w') as f:
                        json.dump(main_json, f)
                # Exits the application
                talking = False
            
if __name__ == "__main__":
    main()  


    

