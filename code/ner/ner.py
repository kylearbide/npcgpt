import pandas as pd 
import json
import spacy 
from spacy.matcher import Matcher 

# nlp = spacy.load('en_core_web_sm')
# matcher = Matcher(nlp.vocab)

with open('data/knowledge_base/kb.json') as f:
    knowledge_base = json.load(f)

# print(knowledge_base)
# print(knowledge_base.keys())

# lists 
items = (knowledge_base['fall_crops'] + knowledge_base['fish'] + knowledge_base['food'] + 
         knowledge_base['minerals'] + knowledge_base['special_crops'] + 
         knowledge_base['spring_crops'] + knowledge_base['summer_crops'])
locations = knowledge_base['locations']
mobs = knowledge_base['mobs']
print(locations)
print()
print(mobs)