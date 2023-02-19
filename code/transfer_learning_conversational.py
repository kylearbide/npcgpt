import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import chain

checkpoint = "bigscience/bloom-1b7"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, device=device)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16, offload_folder="offload")

# We will use 5 special tokens:
# - <bos> to indicate the start of the sequence
# - <eos> to indicate the end of the sequence
# - <speaker1> to indicate the beginning and the tokens of an utterance from the user
# - <speaker2> to indicate the beginning and the tokens of an utterance from the bot
# - <pad> as a padding token to build batches of sequences
SPECIAL_TOKENS = {"additional_special_tokens":["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]}

# Add special tokens
tokenizer.add_special_tokens(SPECIAL_TOKENS)

persona = [["i", "am", "a", "fisherman", "."],
           ["i", "am", "from", "DC", "."]]

history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]

reply = ["great", "to", "hear"]
# Let's add a distractor to our previously defined persona, history and reply
distractor = ["sorry", "to", "hear", "that"]

bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"

def build_inputs(persona, history, reply, num_speakers=1):
    # Build our sequence by adding delimiters and concatenating
    sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]

    if num_speakers == 2:
        sequence = [sequence[0]] + [ [speaker2 if (len(sequence)-i) % 2 else speaker1] + s # For two speakers
                                    for i, s in enumerate(sequence[1:])]
        # Build our word, segments and position inputs from the sequence
        words = list(chain(*sequence))                          # word tokens
        segments = [speaker2 if i % 2 else speaker1             # segment tokens
                    for i, s in enumerate(sequence) for _ in s]

    # For one speaker
    elif num_speakers == 1:
        sequence = [sequence[0]] + [ [speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        words = list(chain(*sequence))  
        
        segments = [speaker1 for s in sequence for _ in s] # For if we only have one speaker
        position = list(range(len(words)))                      # position tokens
    return words, segments, position, sequence

words, segments, position, sequence = build_inputs(persona, history, reply, 1)

# Tokenize
words = tokenizer.convert_tokens_to_ids(words)
segments = tokenizer.convert_tokens_to_ids(segments)

# Build & tokenize inputs ending with our distractor like we did with the gold reply
words_distractor, segments_distractor, _, _ = build_inputs(persona, history, distractor, 1)
words_distractor = tokenizer.convert_tokens_to_ids(words_distractor)
segments_distractor = tokenizer.convert_tokens_to_ids(segments_distractor)



