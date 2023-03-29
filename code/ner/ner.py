import pandas as pd 
import json
import spacy 
from spacy.matcher import Matcher 

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

with open('data/knowledge_base/kb.json') as f:
    knowledge_base = json.load(f)

# print(knowledge_base)
# print(knowledge_base.keys())

#### lists 
items = (knowledge_base['fall_crops'] + knowledge_base['fish'] + knowledge_base['food'] + 
         knowledge_base['minerals'] + knowledge_base['special_crops'] + 
         knowledge_base['spring_crops'] + knowledge_base['summer_crops'])
items = sorted(items)
lower_items = [x.lower() for x in items]
locations = knowledge_base['locations']
locations = sorted(locations)
lower_locations = [x.lower() for x in locations]
mobs = knowledge_base['mobs']
mobs = sorted(mobs)
lower_mobs = [x.lower() for x in mobs]

### patterns 
buy_patterns = [
    [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             # matches on the words 'buy', 'purchase', or 'get' (required)
    {'POS': 'NUM', 'OP': '?'},                                  # matches if a quantity (decimal) was included (optional)
    {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   # matches in case if a player says something general such as 'I would like to buy an [item]' (optional)
    {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            # matches in case if a player says something general such as 'I would like to buy an item' (optional)
    {'LOWER': {'IN': ['of']}, 'OP': '?'},                       # matches in case if a player says something general such as 'I would like to buy a bowl of fish stew' (optional)
    {'LEMMA': {'IN': items}}],                                  # matches lemma of item names against the items list (capitilized)
    [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
    {'POS': 'NUM', 'OP': '?'},                                  
    {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
    {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
    {'LOWER': {'IN': ['of']}, 'OP': '?'},                       
    {'LEMMA': {'IN': lower_items}}]                             # matches lemma of item names against the items list (lower case)
]
matcher.add('BUY_PATTERN', buy_patterns)

item_quest_patterns = [
    [{'LEMMA': {'IN': ['bring', 'brought', 'retrieve', 'retrived', 'get', 'got']}}, # matches on the request key word (required)
    {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some']}, 'OP': '?'},                # matches on if the request if followed by 'me', 'some', 'a', 'an', 'some' 
    {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                              # matches on the request qualifier  
    {'POS': 'NUM', 'OP': '?'},                                                      # matches on if a quantity was requested 
    {'LEMMA': {'IN': items}}],                                                      # matches lemma of item names against the items list (capitilized)
    [{'LEMMA': {'IN': ['bring', 'retrieve', 'get']}},
    {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some']}, 'OP': '?'},
    {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},
    {'POS': 'NUM', 'OP': '?'},
    {'LEMMA': {'IN': lower_items}}]                                                 # matches lemma of item names against the items list (lower case)
]
matcher.add('ITEM_QUEST_PATTERN', item_quest_patterns)

mob_quest_patterns = [
    
]

# testing
test_dialogue = 'I would love if you brought me 100 Quartz'
test_doc = nlp(test_dialogue)
for token in test_doc:
    print(f'Token: \'{token.text}\', Token POS: {token.pos_}, Token lemma: {token.lemma_}')
matches = matcher(test_doc)
for match_id, start, end in matches:
    print('---------------------------------------------')
    print(f'Match tuple: ({match_id}, {start}, {end})')
    print(f'Match: \'{test_doc[start:end]}\'')
    string_id = nlp.vocab.strings[match_id]
    print(f'String ID: {string_id}')