# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import argparse
import csv
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQualityEstimation, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 src,
                 mt,
                 label = None):
        self.src = src
        self.mt = mt
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "src: {}".format(self.src),
            "mt: {}".format(self.mt)
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 choices_features,
                 label

    ):
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_swag_examples(src_file, mt_file, hter_file, is_training):

    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = f.readlines()

    with open(mt_file, 'r', encoding='utf-8') as f:
        mt_sentences = f.readlines()

    with open(hter_file, 'r', encoding='utf-8') as f:
        hters = f.readlines()
    print("src_sentences:%d, mt_sentences:%d, hters:%d"%(len(src_sentences),len(mt_sentences),len(hters)))

    examples = [
        SwagExample(
            src = s.strip(),
            mt = m.strip(),
            label = float(h.strip()) if is_training else None
        ) for s, m, h in zip(src_sentences, mt_sentences, hters) # we skip the line with the column names
    ]

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        src_tokens = tokenizer.tokenize(example.src)
        mt_tokens = tokenizer.tokenize(example.mt)

        choices_features = []

        _truncate_seq_pair(src_tokens, mt_tokens, max_seq_length - 3)

        tokens = ["[CLS]"] + src_tokens + ["[SEP]"] + mt_tokens + ["[SEP]"]
        segment_ids = [0] * (len(src_tokens) + 2) + [1] * (len(mt_tokens) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 5:
            logger.info("*** Example ***")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            if is_training:
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--hidden_dim",
                        default=128,
                        type=int,
                        help="The lstm's hidden_dim.") # 可调
    parser.add_argument("--patience",
                        default=5,
                        type=int,
                        help="Patience.")
    parser.add_argument("--steps_per_eval",
                        default=50,
                        type=int,
                        help="Steps_per_eval.")
    parser.add_argument("--steps_per_stats",
                        default=10,
                        type=int,
                        help="Steps_per_stats.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.") # 可调
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.") # 可调
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_swag_examples(os.path.join(args.data_dir, 'train.src'),
                                            os.path.join(args.data_dir, 'train.mt'),
                                            os.path.join(args.data_dir, 'train.hter'),
                                            is_training = True)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size() # Returns the number of processes in the current process group

    # Prepare model
    model = BertForQualityEstimation.from_pretrained(args.bert_model,
        cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(args.local_rank)),hidden_dim=args.hidden_dim)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]] # ???

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # ???
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    best_pearson = 0
    wait = 0
    Loss_list = []
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in range(int(args.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                Loss_list.append(loss.item())
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                #每隔一定步数就输出loss
                if step % args.steps_per_stats == 0:
                    print("step:%d, loss:%f"%(step, loss * args.gradient_accumulation_steps))

                每隔一定步数就进行验证
                if (step+1) % args.steps_per_eval == 0:
                    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                        eval_examples = read_swag_examples(os.path.join(args.data_dir, 'dev.src'),
                                                            os.path.join(args.data_dir, 'dev.mt'),
                                                            os.path.join(args.data_dir, 'dev.hter'),
                                                            is_training = True)
                        eval_features = convert_examples_to_features(
                            eval_examples, tokenizer, args.max_seq_length, True)
                        logger.info("***** Running evaluation *****")
                        logger.info("  Num examples = %d", len(eval_examples))
                        logger.info("  Batch size = %d", args.eval_batch_size)
                        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.float)
                        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
                        # Run prediction for full data
                        eval_sampler = SequentialSampler(eval_data)
                        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                        model.eval()
                        eval_loss = 0
                        nb_eval_steps, nb_eval_examples = 0, 0
                        pred = np.array([])
                        ref = np.array([])
                        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            segment_ids = segment_ids.to(device)
                            label_ids = label_ids.to(device)

                            with torch.no_grad():
                                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                                qe_scores = model(input_ids, segment_ids, input_mask)

                            qe_scores = qe_scores.detach().cpu().numpy()
                            label_ids = label_ids.to('cpu').numpy()
                            qe_scores = qe_scores.reshape((qe_scores.shape[0],))
                            pred = np.concatenate((pred,qe_scores))
                            ref = np.concatenate((ref,label_ids))

                            eval_loss += tmp_eval_loss.item()

                            nb_eval_examples += input_ids.size(0)
                            nb_eval_steps += 1

                        eval_loss = eval_loss / nb_eval_steps
                        pearson = np.corrcoef(pred, ref)[0, 1]

                        result = {'eval_loss': eval_loss,
                                  'pearson': pearson,
                                  'global_step': global_step,
                                  'loss': tr_loss/nb_tr_steps}

                        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                        with open(output_eval_file, "a") as writer:
                            logger.info("***** Eval results *****")
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))

                        if pearson > best_pearson:
                            best_pearson = pearson
                            wait = 0

                            # Save a trained model and the associated configuration
                            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                            torch.save(model_to_save.state_dict(), output_model_file)
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())
                        else:
                            wait += 1
                        if wait >= args.patience:
                            break # 跳出steps

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if wait >= args.patience:
                print("Early stop !!!")
                break # 跳出epoch

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_loss_file = os.path.join(args.output_dir, 'loss')
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # 保存loss
        with open(output_loss_file, 'w') as f:
            for l in Loss_list:
                f.write(str(l)+'\n')
        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForQualityEstimation(config, hidden_dim=args.hidden_dim)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForQualityEstimation.from_pretrained(args.bert_model, hidden_dim=args.hidden_dim)

    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    config = BertConfig(output_config_file)
    model = BertForQualityEstimation(config, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)


    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_swag_examples(os.path.join(args.data_dir, 'test.src'),
                                            os.path.join(args.data_dir, 'test.mt'),
                                            os.path.join(args.data_dir, 'test.hter'),
                                            is_training = True)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.float)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred = np.array([])
        ref = np.array([])
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                qe_scores = model(input_ids, segment_ids, input_mask)

            qe_scores = qe_scores.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            qe_scores = qe_scores.reshape((qe_scores.shape[0],))
            pred = np.concatenate((pred,qe_scores))
            ref = np.concatenate((ref,label_ids))

            eval_loss += tmp_eval_loss.item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        pearson = np.corrcoef(pred, ref)[0, 1]

        result = {'test_loss': eval_loss,
                  'pearson': pearson}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
