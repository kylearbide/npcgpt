# Conversational AI Model
  
Part 2 of the **npcgpt** project uses the PyTorch and huggingface libraries to create a conversational model for conversing with non playable characters in Stardew Valley. The model is trained off data created from the `code/chatGPT` files and has performed transfer learning on the [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium?text=Hey+my+name+is+Mariama%21+How+are+you%3F) and [GPT2](https://huggingface.co/gpt2) models. 

## Generated Data

The data files used for training are stored in `data/dialogue_datasets`. These files contain ~20,000 samples and are too large to host here, so please reach out to the repository owners for access.

## Trained Models

Trained models are stored in the `conversational/models` folder, but are hidden from the repo due to their size 

## Files 

| File                      | Description                                                                                                                       |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `build_raw_data.py`                   | Functions for formatting training data, building datasets, and cleaning personas.                                     |
| `eval.ipynb`                          | Notebook used for interacting with the trained models and created large sets of responses.                            |
| `example_entry.py`                    | Example of the training data format.                                                                                  |
| `transfer_learning_conversational.py` | Script used to perform transfer learning, includes data loaders, tokenization, and batch processing.                  |