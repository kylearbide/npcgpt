import json
import math
import os
import torch
from pprint import pformat
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, CONFIG_NAME, WEIGHTS_NAME
from itertools import chain
from argparse import ArgumentParser
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_pretrained_bert import cached_path
from collections import defaultdict

"""
NOTE:
Things to add:
    Distributed training and fp16 training?
    format dataset
"""
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

persona = [["i", "am", "a", "fisherman", "."],
           ["i", "am", "from", "DC", "."]]

history = [["hello", "how", "are", "you", "?"],
           ["i", "am", "fine", "thanks", "."]]

reply = ["great", "to", "hear"]
# Let's add a distractor to our previously defined persona, history and reply
distractor = ["sorry", "to", "hear", "that"]

bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def add_special_tokens(model, tokenizer, tokens_dict):
    # We will use 5 special tokens:
    # - <bos> to indicate the start of the sequence
    # - <eos> to indicate the end of the sequence
    # - <speaker1> to indicate the beginning and the tokens of an utterance from the user
    # - <speaker2> to indicate the beginning and the tokens of an utterance from the bot
    # - <pad> as a padding token to build batches of sequences
    # Add special tokens
    orig_num_tokens = tokenizer.vocab_size
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

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True, num_speakers=1):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]
    instance = {}
    
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
        
    
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = segments
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def get_dataset(tokenizer, dataset_path, dataset_type):
    """
    Pull the dataset and tokenize. Needs cache system 
    """
    chat_file = cached_path(dataset_path)
    dataset = {dataset_type:[]}
    with open(chat_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            entry = json.loads(line)
            dataset[dataset_type].append(entry)

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    dataset = tokenize(dataset)
    return dataset

def get_data_loaders(args, tokenizer):
    train = get_dataset(tokenizer, args.train_path, 'train')
    eval = get_dataset(tokenizer, args.eval_path, 'valid')

    num_candidates = len(train['train'][0]["utterances"][0]["candidates"])
    if args.num_candidates > 0:
        num_candidates = min(args.num_candidates, num_candidates)
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset in train['train']:
        persona = dataset["personality"].copy()
        for _ in range(args.personality_permutations):
            for utterance in dataset["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                    for input_name, input_array in instance.items():
                        datasets["train"][input_name].append(input_array)
                datasets["train"]["mc_labels"].append(num_candidates - 1)
                datasets["train"]["n_candidates"] = num_candidates
            persona = [persona[-1]] + persona[:-1]  # permuted personalities
    for dataset in eval['valid']:
        persona = dataset["personality"].copy()
        for _ in range(args.personality_permutations):
            for utterance in dataset["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    instance = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels)
                    for input_name, input_array in instance.items():
                        datasets["valid"][input_name].append(input_array)
                datasets["valid"]["mc_labels"].append(num_candidates - 1)
                datasets["valid"]["n_candidates"] = num_candidates
            persona = [persona[-1]] + persona[:-1]  # permuted personalities
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_loader = DataLoader(train_dataset, sampler=None, batch_size=args.train_batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, sampler=None, batch_size=args.valid_batch_size, shuffle=False)
    return train_loader, valid_loader

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def train():
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="../data/personachat/training.jsonl", help="Path or url of the dataset.")
    parser.add_argument("--eval_path", type=str, default="../data/personachat/valid.jsonl", help="Path or url of the dataset.")
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
    parser.add_argument("--model_checkpoint", type=str, default="../log", help="Path, url or short name for logging")
    args = parser.parse_args()

    checkpoint = "bigscience/bloom-1b7"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device=device)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16, offload_folder="offload")
    # Add special tokens
    print("Adding Special Tokens")
    add_special_tokens(model, tokenizer, ATTR_TO_SPECIAL_TOKEN)

    print("Loading Optimizer")
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    
    print("Getting Data")
    train_loader, val_loader = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            lm_logits, mc_logits, *_ = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = args.model_checkpoint
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    print("Run training")
    print(torch.is_tensor(train_loader))
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint 
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()
