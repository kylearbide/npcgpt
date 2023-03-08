import json
import pandas as pd
import os.path
import random
import re
import nltk.data
from sklearn.model_selection import train_test_split

def get_candidates(personachat:str) -> list:
    """
    Reads the personachat data and returns a list of candidates to be used as noise for the multiple choice options
    Args:
        personachat: Path to the personachat training and valid datasets
    Returns:
        choice_dialogue: list of options to be used for multiple choice 
    """
    training = os.path.join(personachat, "training.jsonl")
    valid = os.path.join(personachat, "valid.jsonl")
    data = []
    # Load both personachat datasets
    with open(training, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    with open(valid, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    
    # Extract all the `candidate` dialogue entries
    choice_dialogue = []
    for entry in data:
        for utterance in entry['utterances']:
            choice_dialogue += utterance['candidates']

    return(choice_dialogue)

def get_bios(data_path:str = "../../data/stardew_valley_villiagers.csv") -> dict:
    """
    Grabs the character bios from csv and formats them by character name 
    Args:
        data_path: path where the character bios are stored
    Returns:
        bios_dict: dictionary linking the character names to their bios
    """
    bios = pd.read_csv(data_path)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    bios_dict = {}
    for bio in bios['bio']:
        if bio.startswith("The") or bio.startswith("Professor"):
            name = bio.split(" ")[1]
            name = name.replace(",","")
        elif bio.startswith("Mr."):
            name = "Mister Qi"
        else:
            name = bio.split(" ")[0]
        
        split_bio = tokenizer.tokenize(bio)
        form_split_bio = [format_input(sent) for sent in split_bio]

        bios_dict.update({name:form_split_bio})
    return(bios_dict)

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

def create_multiple_choice(candidates:list,true_dialogue:str) -> list: 
    """
    Creates the candidates section using noise data from the personachat dataset
    Args:
        true_dialogue: The actual expected output
        candidates: List of the filler phrases taken from the personachat dataset 
    Returns:
        candidates: a list of incorrect outputs with the correct output being the last entry
    """
    # Get random dialogue
    choices = random.choices(candidates, k=20)
    choices.append(true_dialogue)
    return({'candidates':choices})


with open("../../data/data.json", "r") as f:
    data = json.load(f)
# Get a dictionary of the character bios
bios_dict = get_bios()
# Names to be matched with the bios
character_names = ['Abigail', 'Alex', 'Caroline', 'Clint', 'Demetrius', 'Dwarf', 'Elliott', 'Emily', 'Evelyn', 'George', 'Gil', 'Gus', 'Haley', 'Harvey', 'Jas', 'Jodi', 'Kent', 'Krobus', 'Leah', 'Leo', 'LeoMainland', 'Lewis', 'Linus', 'Marnie', 'Maru', 'Penny', 'Sam', 'Sebastian', 'Shane', 'Mister Qi', 'Pam', 'Pierre', 'Robin', 'Sandy', 'Vincent', 'Willy', 'Wizard']
# Grab the noise data
choice_dialogue = get_candidates("../../data/personachat")
# Train set
dataset = []
# If we want to merge the datasets
add_persona_chat = False

for line in data['text']:
    name = list(line.keys())[0]
    # Fix edge case with Leo
    if name == "LeoMainland":
        true_name = "Leo"
    else:
        true_name = name

    if len(line[name]) <= 15:
        continue

    if name in character_names:
        # Make the multiple choice options
        candidates = create_multiple_choice(choice_dialogue, line[name])
        # Build the history based on context
        context = line['_Context'].lower()
        if "winter" in context:
            history = {"history": ["it is winter ."]}
        elif "summer" in context:
            history = {"history": ["it is summer ."]}
        elif "spring" in context:
            history = {"history": ["it is spring ."]}
        elif "fall" in context:
            history = {"history": ["it is fall ."]}
        elif "rain" in context:
            history = {"history": ["it is raining ."]}
        elif "resort" in context:
            history = {"history": ["we are at the resort ."]}
        else:
            history = {"history":[""]}
        # history = {"history":[""]}
        # Create the entry
        candidates.update(history)
        entry = {"utterances": [candidates]}
        entry.update({"personality":bios_dict[true_name]})
        dataset.append(entry)

if add_persona_chat:
    personachat = "../../data/personachat"
    training = os.path.join(personachat, "training.jsonl")
    valid = os.path.join(personachat, "valid.jsonl")
    data = []
    with open(training, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    with open(valid, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    
    dataset += data


train, test = train_test_split(dataset, random_state=0, test_size=0.2)

with open("../../data/dialogue_stardew_train.jsonl", "w") as f:
    for sample in train:
        f.write(json.dumps(sample))
        f.write("\n")

with open("../../data/dialogue_stardew_test.jsonl", "w") as f:
    for sample in test:
        f.write(json.dumps(sample))
        f.write("\n")



