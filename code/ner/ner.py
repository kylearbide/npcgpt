import pandas as pd 
import json
import spacy 
from spacy.matcher import Matcher 

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
secondary_matcher = Matcher(nlp.vocab)

with open('data/knowledge_base/kb.json') as f:
    knowledge_base = json.load(f)

# print(knowledge_base)
# print(knowledge_base.keys())

### lists 

# items
items = (knowledge_base['fall_crops'] + knowledge_base['fish'] + knowledge_base['food'] + 
         knowledge_base['minerals'] + knowledge_base['special_crops'] + 
         knowledge_base['spring_crops'] + knowledge_base['summer_crops'])
idx = items.index('Cranberries')
items[idx] = 'Cranberry'
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

multi_word_mobs = [x for x in mobs if ' ' in x]
mobs_first_word = [x.split()[0] for x in multi_word_mobs]
mobs_second_word = [x.split()[1] for x in multi_word_mobs]
mobs_first_word_lower = [x.lower() for x in mobs_first_word]
mobs_second_word_lower = [x.lower() for x in mobs_second_word] 

### initial patterns 

# buy patterns
buy_patterns = [
    [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             # matches on the words 'buy', 'purchase', or 'get' (required)
     {'POS': 'NUM', 'OP': '?'},                                 # matches if a quantity (decimal) was included (optional)
     {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},  # matches in case if a player says something general such as 'I would like to buy an [item]' (optional)
     {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},           # matches in case if a player says something general such as 'I would like to buy an item' (optional)
     {'LOWER': {'IN': ['of']}, 'OP': '?'},                      # matches in case if a player says something general such as 'I would like to buy a bowl of fish stew' (optional)
     {'LEMMA': {'IN': items}}],                                 # matches lemma of item names against the items list (capitilized)
    [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
     {'POS': 'NUM', 'OP': '?'},                                  
     {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
     {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
     {'LOWER': {'IN': ['of']}, 'OP': '?'},                       
     {'LEMMA': {'IN': lower_items}}],                           # matches lemma of item names against the items list (lower case)
    [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
     {'POS': 'NUM', 'OP': '?'},                                  
     {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
     {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
     {'POS': 'NOUN', 'OP': '?'}, 
     {'LOWER': {'IN': ['of']}, 'OP': '?'},                      
     {'LEMMA': {'IN': items_first_word}},                       # matches on the first word of multi-word items (capitalized)
     {'LEMMA': {'IN': items_second_word}}],                     # matches on the second word of multi-word items (capitalized)
    [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
     {'POS': 'NUM', 'OP': '?'},                                  
     {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
     {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
     {'POS': 'NOUN', 'OP': '?'}, 
     {'LOWER': {'IN': ['of']}, 'OP': '?'},                      
     {'LEMMA': {'IN': items_first_word_lower}},                 # matches on the first word of multi-word items (capitalized)
     {'LEMMA': {'IN': items_second_word_lower}}]                # matches on the second word of multi-word items (capitalized)
]
matcher.add('BUY_PATTERN', buy_patterns)

# item quest patterns
item_quest_patterns = [
    [{'LEMMA': {'IN': ['bring', 'need', 'retrieve', 'get', 'gather', 'collect']}},  # matches on the request key word (required)
     {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some', 'of']}, 'OP': '?'},         # matches on if the request if followed by 'me', 'some', 'a', 'an', 'some' 
     {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             # matches on the request qualifier  
     {'POS': 'ADJ', 'OP': '?'},                                                     # matches on an adjective 
     {'POS': 'NUM', 'OP': '?'},                                                     # matches on if a quantity was requested 
     {'POS': 'NOUN', 'OP': '?'},                                                    # matches on a noun
     {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},                              # matches on if a supplemental quantity was included 
     {'LOWER': {'IN': ['of']}, 'OP': '?'},
     {'POS': 'DET', 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},
     {'POS': 'CCONJ', 'OP': '?'},
     {'POS': 'ADV', 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},
     {'LEMMA': {'IN': items}}],                                                     # matches lemma of item names against the items list (capitilized)
    [{'LEMMA': {'IN': ['bring', 'need', 'retrieve', 'get', 'gather', 'collect']}}, 
     {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some', 'of']}, 'OP': '?'},                 
     {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             
     {'POS': 'ADJ', 'OP': '?'},
     {'POS': 'NUM', 'OP': '?'},
     {'POS': 'NOUN', 'OP': '?'},                                                     
     {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},    
     {'LOWER': {'IN': ['of']}, 'OP': '?'},                                                 
     {'POS': 'DET', 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},
     {'POS': 'CCONJ', 'OP': '?'},
     {'POS': 'ADV', 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},
     {'LEMMA': {'IN': lower_items}}],                                               # matches lemma of item names against the items list (lower case)
    [{'LEMMA': {'IN': ['bring', 'need', 'retrieve', 'get', 'gather', 'collect']}}, 
     {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some', 'of']}, 'OP': '?'},                 
     {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             
     {'POS': 'ADJ', 'OP': '?'},
     {'POS': 'NUM', 'OP': '?'},                        
     {'POS': 'NOUN', 'OP': '?'},                         
     {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},         
     {'LOWER': {'IN': ['of']}, 'OP': '?'},                                            
     {'POS': 'DET', 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},
     {'POS': 'CCONJ', 'OP': '?'},
     {'POS': 'ADV', 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},
     {'LEMMA': {'IN': items_first_word}},                                           # matches on the first word of multi-word items (capitalized)
     {'LEMMA': {'IN': items_second_word}}],                                         # matches on the second word of multi-word items (capitalized)
    [{'LEMMA': {'IN': ['bring', 'need', 'retrieve', 'get', 'gather', 'collect']}}, 
     {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some', 'of']}, 'OP': '?'},                 
     {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},                             
     {'POS': 'NUM', 'OP': '?'},                        
     {'POS': 'NOUN', 'OP': '?'},                         
     {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},         
     {'LOWER': {'IN': ['of']}, 'OP': '?'},                                            
     {'POS': 'DET', 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},
     {'POS': 'CCONJ', 'OP': '?'},
     {'POS': 'ADV', 'OP': '?'},
     {'POS': 'ADJ', 'OP': '?'},
     {'LEMMA': {'IN': items_first_word_lower}},                                     # matches on the first word of multi-word items (lower case)
     {'LEMMA': {'IN': items_second_word_lower}}],                                   # matches on the second word of multi-word items (lower case)
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
     {'LEMMA': {'IN': lower_mobs}}],
    [{'LEMMA': {'IN': ['slay', 'kill', 'defeat', 'find']}},
     {'LOWER': {'IN': ['a', 'an']}, 'OP': '?'},
     {'POS': 'NUM', 'OP': '?'},
     {'LEMMA': {'IN': mobs_first_word}},
     {'LEMMA': {'IN': mobs_second_word}}],
    [{'LEMMA': {'IN': ['slay', 'kill', 'defeat', 'find']}},
     {'LOWER': {'IN': ['a', 'an']}, 'OP': '?'},
     {'POS': 'NUM', 'OP': '?'},
     {'LEMMA': {'IN': mobs_first_word_lower}},
     {'LEMMA': {'IN': mobs_second_word_lower}}]
]
matcher.add('MOB_QUEST_PATTERN', mob_quest_patterns)

### secondary patterns 
target_item = [
    [{'LEMMA': {'IN': items}}],
    [{'LEMMA': {'IN': lower_items}}],
    [{'LEMMA': {'IN': items_first_word}},
     {'LEMMA': {'IN': items_second_word}}],
    [{'LEMMA': {'IN': items_first_word_lower}},
     {'LEMMA': {'IN': items_second_word_lower}}]
]
secondary_matcher.add('TARGET_ITEM', target_item)

target_quantity = [
    [{'POS': 'NUM'}]
]
secondary_matcher.add('TARGET_QUANTITY', target_quantity)

target_mob = [
    [{'LEMMA': {'IN': mobs}}],
    [{'LEMMA': {'IN': lower_mobs}}],
    [{'LEMMA': {'IN': mobs_first_word}},
     {'LEMMA': {'IN': mobs_second_word}}],
    [{'LEMMA': {'IN': mobs_first_word_lower}},
     {'LEMMA': {'IN': mobs_second_word_lower}}]
]
secondary_matcher.add('TARGET_MOB', target_mob)

### testing
# test_dialogue = 'hey there, I heard you\'re quite the adventurer, would you be willing to help me out by slaying 10 lava crabs for me?'
test_dialogue = 'hey there, i heard you\'re quite the adventurer, would you be willing to collect 10 pieces of bixite for me? i need them for a new art project i\'m working on.'
test_doc = nlp(test_dialogue)
# for token in test_doc:
#     print(f'Token: \'{token.text}\', Token POS: {token.pos_}, Token lemma: {token.lemma_}')
print(f'Dialogue item: {test_dialogue}')
matches = matcher(test_doc)
for match_id, start, end in matches:
    print('---------------------------------------------')
    print(f'Match tuple: ({match_id}, {start}, {end})')
    match1 = test_doc[start:end]
    print(f'Match: \'{match1}\'')
    string_id = nlp.vocab.strings[match_id]
    print(f'String ID: {string_id}')

    secondary_matches = secondary_matcher(match1)
    for match_id2, start2, end2 in secondary_matches:
        string_id2 = nlp.vocab.strings[match_id2]
        if string_id == 'MOB_QUEST_PATTERN' and string_id2 == 'TARGET_ITEM':
            continue
        print('\t----------------------')
        print(f'\tSecondary match tuple: ({match_id2}, {start2}, {end2})')
        match2 = match1[start2:end2]
        print(f'\tMatch: \'{match2}\'')
        print(f'\tString ID: {string_id2}')