import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, cached_path
from itertools import chain
from argparse import ArgumentParser
from ignite.engine import Engine, Events

"""
NOTE:
Things to add:
    Distributed training and fp16 training?
    format dataset
"""

parser = ArgumentParser()
parser.add_argument("--train_path", type=str, default="", help="Path or url of the dataset.")
parser.add_argument("--eval_path", type=str, default="", help="Path or url of the dataset.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
args = parser.parse_args()

checkpoint = "bigscience/bloom-1b7"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, device=device)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16, offload_folder="offload")

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}

persona = [["i", "am", "a", "fisherman", "."],
           ["i", "am", "from", "DC", "."]]

history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]

reply = ["great", "to", "hear"]
# Let's add a distractor to our previously defined persona, history and reply
distractor = ["sorry", "to", "hear", "that"]

bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"

def add_special_tokens(model, tokenizer, tokens_dict):
    # We will use 5 special tokens:
    # - <bos> to indicate the start of the sequence
    # - <eos> to indicate the end of the sequence
    # - <speaker1> to indicate the beginning and the tokens of an utterance from the user
    # - <speaker2> to indicate the beginning and the tokens of an utterance from the bot
    # - <pad> as a padding token to build batches of sequences
    # Add special tokens
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(tokens_dict)
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

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

def get_dataset(tokenizer, dataset_path):
    """
    Pull the dataset and tokenize. Needs cache system 
    """
    chat_file = cached_path(dataset_path)
    with open(chat_file, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    dataset = tokenize(dataset)
    return dataset

def get_data_loaders(args, tokenizer):
    train = get_dataset(tokenizer, args.train_path)
    eval = get_dataset(tokenizer, args.eval_path)

    

words, segments, position, sequence = build_inputs(persona, history, reply, 1)

# Tokenize
words = tokenizer.convert_tokens_to_ids(words)
segments = tokenizer.convert_tokens_to_ids(segments)

# Build & tokenize inputs ending with our distractor like we did with the gold reply
words_distractor, segments_distractor, _, _ = build_inputs(persona, history, distractor, 1)
words_distractor = tokenizer.convert_tokens_to_ids(words_distractor)
segments_distractor = tokenizer.convert_tokens_to_ids(segments_distractor)

# Add special tokens
add_special_tokens(model, tokenizer, ATTR_TO_SPECIAL_TOKEN)

# Optimizer
optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
