# chatGPT api Interface
  
This code makes calls to the [ChatGPT API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis) to generate training data for the dialogue model. The prompts for the API were carefully selected based on the desired outputs of the conversational model.

## Generated Data

The generated data files are stored in `data/dialogue_datasets`.

## Files 

| File                      | Description                                                                                                                       |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `build_inputs.py`         | Functions for generating prompts, cleaning outputs, and normalizing to the training format.                                       |
| `interface.py`            | Script for running API calls on loop and creating the training data.                                                              |