import openai
import json
import tiktoken
import pandas as pd
import re
import nltk
import random
from build_inputs import *

with open("./config.json", "r") as f:
    CONFIG = json.loads(f.read())

with open("../../data/knowledge_base/kb.json", "r") as f:
    kb = json.loads(f.read())
    
personas_df = pd.read_csv("../../data/generated_bios.csv")
openai.api_key = CONFIG['openai_apikey']

player_inputs = ['Make a comment about the rain.',
                 'Tell me about yourself.',
                 'Make a comment about spring time.',
                 'Make a comment about summer time.',
                 'Make a comment about winter time.',
                 'Make a comment about fall.']


model = "gpt-3.5-turbo"

def run_kb_calls(kb:dict):
    outputs = []
    # Create the personalities list
    personas = format_personalities(personas_df)
    for key in kb.keys():
        if key == "locations":
            for item in kb[key]:
                prompt = build_location_input(item)
                response = openai.ChatCompletion.create(model = model,
                                                        messages = prompt,
                                                        max_tokens = 300)
                response_text = response['choices'][0]['message']['content']
                personality = random.choice(personas)
                output = convoSample(personality, f"tell me about the {item.lower()} location .", format_input(response_text))
                output.add_candidates([])
                print(output.to_json())
                outputs.append(output.to_json())
                
        elif key == "mobs":
            for item in kb[key]:
                prompt = build_location_input(item)
                response = openai.ChatCompletion.create(model = model,
                                                        messages = prompt,
                                                        max_tokens = 300)
                response_text = response['choices'][0]['message']['content']
                personality = random.choice(personas)
                output = convoSample(personality, f"tell me about the {item.lower()} mob .", format_input(response_text))
                output.add_candidates([])
                print(output.to_json())
                outputs.append(output.to_json())
                
        else:
            for item in kb[key]:
                item_type = key.replace("_"," ")
                if item_type.endswith("s"):
                    item_type = item_type[:-1]
                prompt = build_item_input(item,item_type)
                response = openai.ChatCompletion.create(model = model,
                                                        messages = prompt,
                                                        max_tokens = 300)
                response_text = response['choices'][0]['message']['content']
                personality = random.choice(personas)
                output = convoSample(personality, f"tell me about the {item.lower()} {item_type.lower()} .", format_input(response_text))
                output.add_candidates([])
                print(output.to_json())
                outputs.append(output.to_json())
                
            
    with open("../../data/dialogue_datasets/kb_convos.jsonl", "w") as f:
        for line in outputs:
            f.write(json.dumps(line))
            f.write("\n")          

def run_conversational_calls(personas:pd.DataFrame):
    outputs = []
    num_responses = 0
    # Get personas and cleaned personas
    personas_list = personas['bio']
    personas_format = format_personalities(personas)
    # Loop through personas
    for persona_format, persona in zip(personas_format, personas_list):

        persona_clean = persona.split("\n",1)[0]
        # Loop through input options
        for player_input in player_inputs:
            prompt = build_general_dialogue(persona_clean,player_input)
            try:
                response = openai.ChatCompletion.create(model = model,
                                                        messages = prompt,
                                                        max_tokens = 300)
                player_lines, bot_lines = parse_conversation_output(response)
            except:
                print("Unknown Error")
                break
            num_responses += 1
            # Ensure there are the same number of player and bot lines
            if len(player_lines) == len(bot_lines):
                ix = 0
                for player_line, bot_line in zip(player_lines, bot_lines):
                    if ix == 0:
                        output = convoSample(persona_format, format_input(player_line), format_input(bot_line))
                        output.add_candidates(candidates=[])
                    else:
                        output.add_dialog(format_input(player_line), format_input(bot_line))
                    ix += 1
                print(output.to_json())
                outputs.append(output.to_json())
            # Sometimes they insert an extra player line so thats an easy fix
            elif len(player_lines[:-1]) == len(bot_lines):
                ix = 0
                for player_line, bot_line in zip(player_lines[:-1], bot_lines):
                    if ix == 0:
                        output = convoSample(persona_format, format_input(player_line), format_input(bot_line))
                        output.add_candidates(candidates=[])
                    else:
                        output.add_dialog(format_input(player_line), format_input(bot_line))
                    ix += 1
                print(output.to_json())
                outputs.append(output.to_json())
            # Otherwise, save the outputs up to this point and exit
            else:
                print("LENGTH ERROR")
                print(player_lines)
                print(bot_lines)
                print(response)
                break
                
            if num_responses >= 10:
                num_responses = 0
                with open("../../data/dialogue_datasets/chatgpt_convos.jsonl", "a") as f:
                    for line in outputs:
                        f.write(json.dumps(line))
                        f.write("\n")
                
                outputs = []
                
    with open("../../data/dialogue_datasets/chatgpt_convos.jsonl", "a") as f:
        for line in outputs:
            f.write(json.dumps(line))
            f.write("\n")

def run_quest_calls(kb:dict):
    outputs = []
    # Create the personalities list
    personas = format_personalities(personas_df)
    # Get the prompt friendly format
    personas_list = personas_df['bio']
    # Makes personas indexes to align these too for random choice
    persona_indices = range(0,len(personas_list))
    for key in kb.keys():
        if key == "locations":
            continue
        elif key == "mobs":
            for item in kb[key]:
                num_iter = 0
                while num_iter <= 5:
                    num_iter += 1
                    # Build the personalities
                    personality_index = random.choice(persona_indices)
                    persona = personas_list[personality_index]
                    persona_form = personas[personality_index]
                    # Get the response
                    prompt = build_mob_quest_input(persona, item)
                    print(item)
                    timeout = True
                    while timeout:
                        try:
                            response = openai.ChatCompletion.create(model = model,
                                                                messages = prompt,
                                                                max_tokens = 300)
                        except:
                            print("Timeout")
                        else:
                            timeout = False
                    response_text = response['choices'][0]['message']['content']
                    output = convoSample(persona_form, f"give me a quest to slay a mob in stardew valley .", format_input(response_text))
                    output.add_candidates([])
                    print(output.to_json())
                    outputs.append(output.to_json())
        else:
            item_type = key.replace("_"," ")
            if item_type.endswith("s"):
                item_type = item_type[:-1]
            print(item_type)
            for item in kb[key]:
                num_iter = 0
                while num_iter < 5:
                    num_iter += 1
                    # Build the personalities
                    personality_index = random.choice(persona_indices)
                    persona = personas_list[personality_index]
                    persona_form = personas[personality_index]
                    # Get the response
                    prompt = build_item_quest_input(persona, item)
                    print(item)
                    timeout = True
                    while timeout:
                        try:
                            response = openai.ChatCompletion.create(model = model,
                                                                messages = prompt,
                                                                max_tokens = 300)
                        except:
                            print("Timeout")
                        else:
                            timeout = False
                    
                    response_text = response['choices'][0]['message']['content']
                    output = convoSample(persona_form, f"give me a quest to collect a type of {item_type.lower()} .", format_input(response_text))
                    output.add_candidates([])
                    print(output.to_json())
                    outputs.append(output.to_json())
    with open("../../data/dialogue_datasets/kb_quests.jsonl", "a") as f:
        for line in outputs:
            f.write(json.dumps(line))
            f.write("\n")

if __name__ == "__main__":
    run_conversational_calls(personas_df)