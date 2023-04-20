import pandas as pd 
import json
import spacy 
from spacy.matcher import Matcher 

class NERMatcher():

    def __init__(self) -> None:

        # general setup 
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)
        self.secondary_matcher = Matcher(self.nlp.vocab)

        with open('data/knowledge_base/kb.json') as f:
            self.knowledge_base = json.load(f)

        # lists 
        self.items = None
        self.lower_items = None 
        self.items_first_word = None 
        self.items_second_word = None 
        self.items_first_word_lower = None
        self.items_second_word_lower = None 

        self.locations = None
        self.lower_locations = None 
        self.locations_first_word = None 
        self.locations_second_word = None 
        self.locations_first_word_lower = None 
        self.locations_second_word_lower = None

        self.mobs = None 
        self.lower_mobs = None
        self.mobs_first_word = None 
        self.mobs_second_word = None
        self.mobs_first_word_lower = None 
        self.mobs_second_word_lower = None 

        self._create_lists()

        # initial patterns 
        self.buy_patterns = None 
        self.item_quest_patterns = None 
        self.mob_quest_patterns = None 

        # secondary patterns
        self.target_items = None 
        self.target_quantity = None 
        self.target_mob = None 

        self._create_patterns() 

        # add initial patterns 
        self.matcher.add('BUY_PATTERN', self.buy_patterns)
        self.matcher.add('ITEM_QUEST_PATTERN', self.item_quest_patterns)
        self.matcher.add('MOB_QUEST_PATTERN', self.mob_quest_patterns)

        # add secondary patterns 
        self.secondary_matcher.add('TARGET_ITEM', self.target_item)
        self.secondary_matcher.add('TARGET_QUANTITY', self.target_quantity)
        self.secondary_matcher.add('TARGET_MOB', self.target_mob)

    def _create_lists(self) -> None:

        self.items = (self.knowledge_base['fall_crops'] + self.knowledge_base['fish'] + self.knowledge_base['food'] + 
                      self.knowledge_base['minerals'] + self.knowledge_base['special_crops'] + 
                      self.knowledge_base['spring_crops'] + self.knowledge_base['summer_crops'])
        idx = self.items.index('Cranberries')
        self.items[idx] = 'Cranberry'
        self.items = sorted(self.items)
        self.lower_items = [x.lower() for x in self.items]
        multi_word_items = [x for x in self.items if ' ' in x]
        self.items_first_word = [x.split()[0] for x in multi_word_items]
        self.items_second_word = [x.split()[1] for x in multi_word_items]
        self.items_first_word_lower = [x.lower() for x in self.items_first_word]
        self.items_second_word_lower = [x.lower() for x in self.items_second_word]

        self.locations = self.knowledge_base['locations']
        self.locations = sorted(self.locations)
        self.lower_locations = [x.lower() for x in self.locations]
        multi_word_locations = [x for x in self.locations if ' ' in x]
        self.locations_first_word = [x.split()[0] for x in multi_word_locations]
        self.locations_second_word = [x.split()[1] for x in multi_word_locations]
        self.locations_first_word_lower = [x.lower() for x in self.locations_first_word]
        self.locations_second_word_lower = [x.lower() for x in self.locations_second_word]

        self.mobs = self.knowledge_base['mobs']
        self.mobs = sorted(self.mobs)
        self.lower_mobs = [x.lower() for x in self.mobs]
        multi_word_mobs = [x for x in self.mobs if ' ' in x]
        self.mobs_first_word = [x.split()[0] for x in multi_word_mobs]
        self.mobs_second_word = [x.split()[1] for x in multi_word_mobs]
        self.mobs_first_word_lower = [x.lower() for x in self.mobs_first_word]
        self.mobs_second_word_lower = [x.lower() for x in self.mobs_second_word]

    def _create_patterns(self) -> None:

        ### initial patterns 
        # buy patterns
        self.buy_patterns = [
            [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},            # matches on the words 'buy', 'purchase', or 'get' (required)
            {'POS': 'NUM', 'OP': '?'},                                 # matches if a quantity (decimal) was included (optional)
            {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},  # matches in case if a player says something general such as 'I would like to buy an [item]' (optional)
            {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},           # matches in case if a player says something general such as 'I would like to buy an item' (optional)
            {'LOWER': {'IN': ['of']}, 'OP': '?'},                      # matches in case if a player says something general such as 'I would like to buy a bowl of fish stew' (optional)
            {'LEMMA': {'IN': self.items}}],                            # matches lemma of item names against the items list (capitilized)
            [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
            {'POS': 'NUM', 'OP': '?'},                                  
            {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
            {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
            {'LOWER': {'IN': ['of']}, 'OP': '?'},                       
            {'LEMMA': {'IN': self.lower_items}}],                      # matches lemma of item names against the items list (lower case)
            [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
            {'POS': 'NUM', 'OP': '?'},                                  
            {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
            {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
            {'POS': 'NOUN', 'OP': '?'}, 
            {'LOWER': {'IN': ['of']}, 'OP': '?'},                      
            {'LEMMA': {'IN': self.items_first_word}},                  # matches on the first word of multi-word items (capitalized)
            {'LEMMA': {'IN': self.items_second_word}}],                # matches on the second word of multi-word items (capitalized)
            [{'LEMMA': {'IN': ['buy', 'purchase', 'get']}},             
            {'POS': 'NUM', 'OP': '?'},                                  
            {'LOWER': {'IN': ['new', 'a', 'an', 'some']}, 'OP': '?'},   
            {'LOWER': {'IN': ['item', 'items']}, 'OP': '?'},            
            {'POS': 'NOUN', 'OP': '?'}, 
            {'LOWER': {'IN': ['of']}, 'OP': '?'},                      
            {'LEMMA': {'IN': self.items_first_word_lower}},            # matches on the first word of multi-word items (capitalized)
            {'LEMMA': {'IN': self.items_second_word_lower}}]           # matches on the second word of multi-word items (capitalized)
        ]

        # item quest patterns
        self.item_quest_patterns = [
            [{'LEMMA': {'IN': ['bring', 'need', 'retrieve', 'get', 'gather', 'collect', 'take', 'catch']}}, # matches on the request key word (required)
            {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some', 'of', 'down']}, 'OP': '?'},         # matches on if the request if followed by 'me', 'some', 'a', 'an', 'some' 
            {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             # matches on the request qualifier  
            {'POS': 'ADJ', 'OP': '*'},                                                     # matches on an adjective 
            {'POS': 'NUM', 'OP': '?'},                                                     # matches on if a quantity was requested 
            {'POS': 'NOUN', 'OP': '?'},                                                    # matches on a noun
            {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},                              # matches on if a supplemental quantity was included 
            {'LOWER': {'IN': ['of']}, 'OP': '?'},
            {'POS': 'DET', 'OP': '?'},
            {'POS': 'ADJ', 'OP': '?'},
            {'POS': 'CCONJ', 'OP': '?'},
            {'POS': 'ADV', 'OP': '?'},
            {'POS': 'ADJ', 'OP': '?'},
            {'LEMMA': {'IN': self.items}}],                                                # matches lemma of item names against the items list (capitilized)
            [{'LEMMA': {'IN': ['bring', 'need', 'retrieve', 'get', 'gather', 'collect', 'take', 'catch']}}, 
            {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some', 'of', 'down']}, 'OP': '?'},                 
            {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             
            {'POS': 'ADJ', 'OP': '*'},
            {'POS': 'NUM', 'OP': '?'},
            {'POS': 'NOUN', 'OP': '?'},                                                     
            {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},    
            {'LOWER': {'IN': ['of']}, 'OP': '?'},                                                 
            {'POS': 'DET', 'OP': '?'},
            {'POS': 'ADJ', 'OP': '?'},
            {'POS': 'CCONJ', 'OP': '?'},
            {'POS': 'ADV', 'OP': '?'},
            {'POS': 'ADJ', 'OP': '?'},
            {'LEMMA': {'IN': self.lower_items}}],                                          # matches lemma of item names against the items list (lower case)
            [{'LEMMA': {'IN': ['bring', 'need', 'retrieve', 'get', 'gather', 'collect', 'take', 'catch']}}, 
            {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some', 'of', 'down']}, 'OP': '?'},                 
            {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},                             
            {'POS': 'ADJ', 'OP': '*'},
            {'POS': 'NUM', 'OP': '?'},                        
            {'POS': 'NOUN', 'OP': '?'},                         
            {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},         
            {'LOWER': {'IN': ['of']}, 'OP': '?'},                                            
            {'POS': 'DET', 'OP': '?'},
            {'POS': 'ADJ', 'OP': '?'},
            {'POS': 'CCONJ', 'OP': '?'},
            {'POS': 'ADV', 'OP': '?'},
            {'POS': 'ADJ', 'OP': '?'},
            {'LEMMA': {'IN': self.items_first_word}},                                      # matches on the first word of multi-word items (capitalized)
            {'LEMMA': {'IN': self.items_second_word}}],                                    # matches on the second word of multi-word items (capitalized)
            [{'LEMMA': {'IN': ['bring', 'need', 'retrieve', 'get', 'gather', 'collect', 'take', 'catch']}}, 
            {'LOWER': {'IN': ['me', 'some', 'a', 'an', 'some', 'of', 'down']}, 'OP': '?'},                 
            {'LOWER': {'IN': ['some', 'a', 'an']}, 'OP': '?'},
            {'POS': 'ADJ', 'OP': '*'},                             
            {'POS': 'NUM', 'OP': '?'},                        
            {'POS': 'NOUN', 'OP': '?'},                         
            {'LEMMA': {'IN': ['dozen', 'piece']}, 'OP': '?'},         
            {'LOWER': {'IN': ['of']}, 'OP': '?'},                                            
            {'POS': 'DET', 'OP': '?'},
            {'POS': 'ADJ', 'OP': '?'},
            {'POS': 'CCONJ', 'OP': '?'},
            {'POS': 'ADV', 'OP': '?'},
            {'POS': 'ADJ', 'OP': '?'},
            {'LEMMA': {'IN': self.items_first_word_lower}},                                # matches on the first word of multi-word items (lower case)
            {'LEMMA': {'IN': self.items_second_word_lower}}],                              # matches on the second word of multi-word items (lower case)
        ]

        # mob quest patterns 
        self.mob_quest_patterns = [
            [{'LEMMA': {'IN': ['slay', 'kill', 'defeat', 'find']}},
            {'LOWER': {'IN': ['a', 'an']}, 'OP': '?'},
            {'POS': 'NUM', 'OP': '?'},
            {'LEMMA': {'IN': self.mobs}}],
            [{'LEMMA': {'IN': ['slay', 'kill', 'defeat', 'find']}},
            {'LOWER': {'IN': ['a', 'an']}, 'OP': '?'},
            {'POS': 'NUM', 'OP': '?'},
            {'LEMMA': {'IN': self.lower_mobs}}],
            [{'LEMMA': {'IN': ['slay', 'kill', 'defeat', 'find']}},
            {'LOWER': {'IN': ['a', 'an']}, 'OP': '?'},
            {'POS': 'NUM', 'OP': '?'},
            {'LEMMA': {'IN': self.mobs_first_word}},
            {'LEMMA': {'IN': self.mobs_second_word}}],
            [{'LEMMA': {'IN': ['slay', 'kill', 'defeat', 'find']}},
            {'LOWER': {'IN': ['a', 'an']}, 'OP': '?'},
            {'POS': 'NUM', 'OP': '?'},
            {'LEMMA': {'IN': self.mobs_first_word_lower}},
            {'LEMMA': {'IN': self.mobs_second_word_lower}}]
        ]

        ### secondary patterns 
        self.target_item = [
            [{'LEMMA': {'IN': self.items}}],
            [{'LEMMA': {'IN': self.lower_items}}],
            [{'LEMMA': {'IN': self.items_first_word}},
            {'LEMMA': {'IN': self.items_second_word}}],
            [{'LEMMA': {'IN': self.items_first_word_lower}},
            {'LEMMA': {'IN': self.items_second_word_lower}}]
        ]

        self.target_quantity = [
            [{'POS': 'NUM'}]
        ]

        self.target_mob = [
            [{'LEMMA': {'IN': self.mobs}}],
            [{'LEMMA': {'IN': self.lower_mobs}}],
            [{'LEMMA': {'IN': self.mobs_first_word}},
            {'LEMMA': {'IN': self.mobs_second_word}}],
            [{'LEMMA': {'IN': self.mobs_first_word_lower}},
            {'LEMMA': {'IN': self.mobs_second_word_lower}}]
        ]
    
    def match_dialogue(self, dialogue):

        test_dialogue = self.nlp(dialogue)
        target_info = {}

        init_matches = self.matcher(test_dialogue)
        for match_id, start, end in init_matches:

            string_id = self.nlp.vocab.strings[match_id]
            match1 = test_dialogue[start:end]

            target_info[string_id] = {}

            secondary_matches = self.secondary_matcher(match1)
            for match_id2, start2, end2 in secondary_matches:

                string_id2 = self.nlp.vocab.strings[match_id2]
                match2 = match1[start2:end2]
                if string_id2 == 'MOB_QUEST_PATTERN' and string_id2 == 'TARGET_ITEM':
                    continue 
                if string_id2 == 'TARGET_ITEM':
                    target_info[string_id]['target_item'] = match2 
                elif string_id2 == 'TARGET_MOB':
                    target_info[string_id]['target_mob'] = match2
                elif string_id2 == 'TARGET_QUANTITY':
                    target_info[string_id]['target_quantity'] = match2
    
        return target_info

### test dialogue 
# test_dialogue = 'hey there, I heard you\'re quite the adventurer, would you be willing to help me out by slaying 10 lava crabs for me?'
# test_dialogue = 'hey there, i heard you\'re quite the adventurer, would you be willing to collect 10 pieces of bixite for me? i need them for a new art project i\'m working on.'
# test_dialogue = 'hey there, i heard you\'re quite the adventurer, would you be willing to collect 10 cranberry pips for me? i need them for a new recipe i\'m working on.'