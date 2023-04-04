# Character Bio Generation  
  
This code makes up the first step of the **npcgpt** project. By fine tuning the OpenAI open source [GPT2LMHeadModel](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel), we can create dynamic and unique NPC personas. 

## Training Data 

The training data can be found at `data/character_bios.csv`. It consists of sample bios for made up Stardew Valley characters that are formatted and styled after the [Stardew Valley villager wiki pages](https://stardewvalleywiki.com/Villagers). 

## Files 

| File                      | Description                                                                                                                       |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `bio_dataset.py`          | Custom pytorch dataset for the sample bio data.                                                                                   |
| `character_generation.py` | Trains and saves the model.                                                                                                       |
| `generate.py`             | Loads the saved model and creates character bios, can generate many bios at once (for testing) or a single bios (for production). |
| `test.py`                 | Calls the `generate.py` script to generate bio(s).                                                                                |