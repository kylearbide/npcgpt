# Data 

| File/Folder                  | Description                                                                                                                                                                 |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| convo_tests                  | Sample dialogue outputs from model 2.                                                                                                                                       |
| knowledge_base               | Knowledge base of items, locations, and mobs in the game.                                                                                                                   |
| `bio_seed_data.csv`          | Seed data used to generate new character bios. When generating a character bio, a random name and two non-contradicting adjectives are picked to feed into the first model. |
| `character_bios.csv`         | Training data for the first model.                                                                                                                                          |
| `convo_outputs_3_6.csv`      | First batch of conversational outputs.                                                                                                                                      |
| `generated_bios.csv`         | If generating many sample bios at once for testing from model 1, they are output to this csv file.                                                                          |
| `generative_model_output.csv`| Holdout testing data from model 1 training process.                                                                                                                         |
| `stardew_valley_villagers.csv`| The actual villager bios from the Stardew Valley villager wiki pages.                                                                                                       |
| `main.json`                  | Holds the current character persona and a flag to determine whether to generate a new persona on next run.                                                                  |