# npcgpt
Leveraging large language models for video games and NPCs. This proof of concept is built out for the game [Stardew Valley](https://www.stardewvalley.net/). The project is made up of 3 parts, described below.

## Part 1: Character Generation 

To generate dynamic and unique NPC's, the first model generates a short character bio. The model is based on OpenAI's open source [GPT2LMHeadModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel). The model was fine tuned using sample character bios that were generated using similar formatting and description styles as the [Stardew Valley villager wiki pages](https://stardewvalleywiki.com/Villagers). The sample bios used for training data can be found at `data/character_bios.csv`. The generated character bio is then cleaned and processed to be used as input for the dialogue model. 

## Part 2: Dialogue Model  

## Part 3: Rule-Based Named-Entity Recognition 

To capture the transactional intention of dialogue with the NPC, two [Spacy Matchers](https://spacy.io/api/matcher) are used to identify and pull the relevant information from the dialogue. For example, if the user asks the NPC for a quest, the NPC might give the user an item quest or a mob related quest. The Matcher rules can identify that a quest has been given to the user and extract the target items (or mobs), the quantity requested, and the potential reward offered to the user upon completion of the request. This information is used to dynamically implement the transactions in the game, allowing for a more robust and complete user experience. 