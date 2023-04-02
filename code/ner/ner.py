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

### lists 

# items
items = (knowledge_base['fall_crops'] + knowledge_base['fish'] + knowledge_base['food'] + 
         knowledge_base['minerals'] + knowledge_base['special_crops'] + 
         knowledge_base['spring_crops'] + knowledge_base['summer_crops'])
items = sorted(items)
lower_items = [x.lower() for x in items]

multi_word_items = [x for x in items if ' ' in x]
items_first_word = [x.split()[0] for x in multi_word_items]
items_second_word = [x.split()[1] for x in multi_word_items]
items_first_word_lower = [x.lower() for x in items_first_word]
items_second_word_lower = [x.lower() for x in items_second_word]

# locations
locations = knowledge_base['locations']
locations = sorted(locations)
lower_locations = [x.lower() for x in locations]

multi_word_locations = [x for x in locations if ' ' in x]
locations_first_word = [x.split()[0] for x in multi_word_locations]
locations_second_word = [x.split()[1] for x in multi_word_locations]
locations_first_word_lower = [x.lower() for x in locations_first_word]
locations_second_word_lower = [x.lower() for x in locations_second_word]

# mobs 
mobs = knowledge_base['mobs']
mobs = sorted(mobs)
lower_mobs = [x.lower() for x in mobs]

### patterns 

# buy patterns
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
    {'LEMMA': {'IN': lower_items}}],                            # matches lemma of item names against the items list (lower case)

    [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
    {'POS': 'NUM', 'OP': '?'},                                  
    {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
    {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
    {'POS': 'NOUN', 'OP': '?'}, 
    {'LOWER': {'IN': ['of']}, 'OP': '?'},                      
    {'LEMMA': {'IN': items_first_word}},                        # matches on the first word of multi-word items (capitalized)
    {'LEMMA': {'IN': items_second_word}}],                      # matches on the second word of multi-word items (capitalized)

    [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
    {'POS': 'NUM', 'OP': '?'},                                  
    {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
    {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
    {'POS': 'NOUN', 'OP': '?'}, 
    {'LOWER': {'IN': ['of']}, 'OP': '?'},                      
    {'LEMMA': {'IN': items_first_word_lower}},                  # matches on the first word of multi-word items (capitalized)
    {'LEMMA': {'IN': items_second_word_lower}}]                 # matches on the second word of multi-word items (capitalized)
]
matcher.add('BUY_PATTERN', buy_patterns)

# item quest patterns
item_quest_patterns = [
    [{'LEMMA': {'IN': ['bring', 'need', 'brought', 'retrieve', 'retrived', 'get', 'got', 'gather']}}, # matches on the request key word (required)
    {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some']}, 'OP': '?'},                # matches on if the request if followed by 'me', 'some', 'a', 'an', 'some' 
    {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                              # matches on the request qualifier  
    {'POS': 'NUM', 'OP': '?'},                                                      # matches on if a quantity was requested 
    {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},                              # matches on if a supplemental quantity was included 
    {'LOWER': {'IN': ['of']}, 'OP': '?'},
    {'LEMMA': {'IN': items}}],                                                      # matches lemma of item names against the items list (capitilized)

    [{'LEMMA': {'IN': ['bring', 'need', 'brought', 'retrieve', 'retrived', 'get', 'got', 'gather']}}, 
    {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some']}, 'OP': '?'},                 
    {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             
    {'POS': 'NUM', 'OP': '?'},                                                     
    {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},    
    {'LOWER': {'IN': ['of']}, 'OP': '?'},                                                 
    {'LEMMA': {'IN': lower_items}}],                                                # matches lemma of item names against the items list (lower case)

    [{'LEMMA': {'IN': ['bring', 'need', 'brought', 'retrieve', 'retrived', 'get', 'got', 'gather']}}, 
    {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some']}, 'OP': '?'},                 
    {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             
    {'POS': 'NUM', 'OP': '?'},                        
    {'POS': 'NOUN', 'OP': '?'},                         
    {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},         
    {'LOWER': {'IN': ['of']}, 'OP': '?'},                                            
    {'LEMMA': {'IN': items_first_word}},                                            # matches on the first word of multi-word items (capitalized)
    {'LEMMA': {'IN': items_second_word}}],                                          # matches on the second word of multi-word items (capitalized)

    [{'LEMMA': {'IN': ['bring', 'need', 'brought', 'retrieve', 'retrived', 'get', 'got', 'gather']}}, 
    {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some']}, 'OP': '?'},                 
    {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             
    {'POS': 'NUM', 'OP': '?'},                        
    {'POS': 'NOUN', 'OP': '?'},                         
    {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},         
    {'LOWER': {'IN': ['of']}, 'OP': '?'},                                            
    {'LEMMA': {'IN': items_first_word_lower}},                                      # matches on the first word of multi-word items (lower case)
    {'LEMMA': {'IN': items_second_word_lower}}],                                    # matches on the second word of multi-word items (lower case)
]
matcher.add('ITEM_QUEST_PATTERN', item_quest_patterns)

# mob quest patterns 
mob_quest_patterns = [
    [{'LEMMA': {'IN': ['slay', 'kill', 'defeat', 'find']}},
    {'LOWER': {'IN': ['a', 'an']}, 'OP': '?'},
    {'POS': 'NUM', 'OP': '?'},
    {'LEMMA': {'IN': mobs}}],
    [{'LEMMA': {'IN': ['slay', 'kill', 'defeat', 'find']}},
    {'LOWER': {'IN': ['a', 'an']}, 'OP': '?'},
    {'POS': 'NUM', 'OP': '?'},
    {'LEMMA': {'IN': lower_mobs}}]
]
matcher.add('MOB_QUEST_PATTERN', mob_quest_patterns)

### testing
test_dialogue = 'hey there, I heard you\'re quite the adventurer, would you be willing to gather 10 pieces of wheat for me?'
test_doc = nlp(test_dialogue)
print(f'Dialogue item: {test_dialogue}')
# for token in test_doc:
#     print(f'Token: \'{token.text}\', Token POS: {token.pos_}, Token lemma: {token.lemma_}')
matches = matcher(test_doc)
for match_id, start, end in matches:
    print('---------------------------------------------')
    print(f'Match tuple: ({match_id}, {start}, {end})')
    print(f'Match: \'{test_doc[start:end]}\'')
    string_id = nlp.vocab.strings[match_id]
    print(f'String ID: {string_id}')