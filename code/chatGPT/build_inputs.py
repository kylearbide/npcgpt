import pandas as pd
import re
import nltk

personalities = pd.read_csv("../../data/generative_model_output.csv")

class convoSample():
    def __init__(self, personality:list, player_input:str, response:str):
        self.persona = personality
        self.utterances = [{"history":[player_input]}]
        self.response = response
        self.history = [player_input, response]
    def add_candidates(self,candidates:list):
        self.candidates = candidates
        candidates_w_response = candidates + [self.response]
        self.utterances[0].update({"candidates":candidates_w_response})
    def add_dialog(self,player_input,response):
        candidates_w_response = self.candidates + [response]
        self.utterances.append({"history":self.history + [player_input], "candidates":candidates_w_response})
        self.history.append(player_input)
        self.history.append(response)
    def to_json(self) -> dict:
        output = {}
        output["personality"] = self.persona
        output["utterances"] = self.utterances
        return(output)
        

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

def format_personalities(personalities:pd.DataFrame):
    """
    Transform the model outputs into a list to be parsed through
    """
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    full_bios = list(personalities['bio'])
    full_bios = [bio.lower().split("\n",1)[0] for bio in full_bios]
    full_bios = [bio.split("<|endoftext|>")[0] for bio in full_bios]
    full_bios = [sent_tokenizer.tokenize(bio) for bio in full_bios]
    full_bios = [[format_input(sent) for sent in sents] for sents in full_bios]
    return(full_bios)

def build_general_dialogue(persona:str, player_input:str):
    """
    Build a prompt for a general conversation based on a personality
    """
    messages = [
    {"role": "system", "content": "You are a helpful assistant that generates dialogue from a given Stardew Valley persona."},
    {"role": "user", "content": f'Create a conversation between a Stardew Valley character and a player. Each should speak three times. The persona is "{persona}". The first line from the player is "{player_input}"' }
    ]
    return(messages)

def build_item_input(item:str, item_type:str):
    """
    Build the prompt for generating an item description to be used in fake conversations 
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that will return a 3 sentence definition for an item in Stardew Valley."},
        {"role": "user", "content": f'The item is the {item} and it is a {item_type}.'}
        ]
    return(messages)

def build_mob_input(mob:str):
    """
    Build the prompt for generating a mob description to be used in fake conversations 
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that will return a 3 sentence definition for a mob in Stardew Valley."},
        {"role": "user", "content": f'The mob is the {mob}.'}
        ]
    return(messages)

def build_location_input(location:str):
    """
    Build the prompt for generating a location description to be used in fake conversations 
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that will return a 3 sentence definition for a location in Stardew Valley."},
        {"role": "user", "content": f'The location is the {location}.'}
        ]
    return(messages)

def build_item_quest_input(persona:str, item:str):
    """
    Build a prompt for generic conversation where the npc offers the player a quest to retrieve a certain item
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates dialogue from a given Stardew Valley persona."},
        {"role": "user", "content": f'Write a one sentence dialogue line from a Stardew Valley Character to a player. The characters persona is {persona}. The character should give the player a quest to collect {item}. Specify the number of items.' }
        ]
    return(messages)

def build_mob_quest_input(persona:str, mob:str):
    """
    Build a prompt for a generic conversation where the npc offers the player a quest to slay certain mobs
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates dialogue from a given Stardew Valley persona."},
        {"role": "user", "content": f'Write a one sentence dialogue line from a Stardew Valley Character to a player. The characters persona is {persona}. The character should give the player a quest to slay {mob} mobs. Specify the number of mobs.' }
        ]
    return(messages)

def parse_conversation_output(output:dict):
    """
    Format the conversational output from chatGPT
    """
    content = output['choices'][0]['message']['content']
    lines = content.split("\n")
    player_inputs, bot_inputs = [],[]
    for line in lines:
        if line:
            if line.startswith("Player"):
                player_inputs.append(line.split(": ",1)[1])
            else:
                bot_inputs.append(line.split(": ",1)[1])
    return(player_inputs,bot_inputs)