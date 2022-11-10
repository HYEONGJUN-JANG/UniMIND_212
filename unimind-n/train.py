import argparse
import glob
import logging
import os
import random
from transformers import BartForConditionalGeneration, BertTokenizer, BartConfig, AdamW
from pytorch_transformers import WarmupLinearSchedule, WEIGHTS_NAME
import torch
import utils
import data_reader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import metrics
from model import UniMind
import math
from pytz import timezone
from datetime import datetime
import sys

def get_time_kst(): return datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H%M%S')


class DataFrame(Dataset):
    def __init__(self, data, args):
        self.source_ids = data['source_ids']
        self.target_ids = data['target_ids']
        self.item_ids = data['item_ids']
        self.item_num = args.item_num
        self.max_len = args.max_seq_length
        self.max_tgt_len = args.max_target_length

    def __getitem__(self, index):
        return self.source_ids[index][-self.max_len:], self.target_ids[index][-self.max_tgt_len:], self.item_ids[index], \
               self.item_num

    def __len__(self):
        return len(self.source_ids)


def collate_fn(data):
    source_ids, target_ids, item_ids, item_num = zip(*data) # DataFrame.getitem으로받은것 (4개)
    batch_size = len(source_ids)

    input_ids = [torch.tensor(source_id).long() for source_id in source_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    attention_mask = input_ids.ne(0)
    labels = [torch.tensor(target_id).long() for target_id in target_ids]
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    item_ids = torch.tensor(item_ids).long()

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'item_ids': item_ids,
            }


def train(args, train_dataset, model, tokenizer, task=None):
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, len(args.device_id))
    train_dataloader = DataLoader(DataFrame(train_dataset, args), batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                 args.train_batch_size * args.gradient_accumulation_steps)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    if task is None:
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                                bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')
    else:
        train_iterator = trange(int(args.num_ft_epochs), desc="Epoch",
                                bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')
    utils.set_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    global_step = 0

    # total_rouge = evaluate(args, model, tokenizer, save_output=True)
    best_f1 = 0  # total_rouge[2]

    for e in train_iterator:
        logging.info("training for epoch {} ...".format(e))
        print("training for epoch {} ...".format(e))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'labels': batch['labels'].to(args.device),
                      'item_ids': batch['item_ids'].to(args.device)}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            ## HJ 멀티GPU때
            if len(args.device_id) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # HJ Optimizer를 scheduler보다 먼저 step 하라는 torch 글을 보고 위아래순서바꿈
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            global_step += 1

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / (step + 1), global_step)
        print('loss: {}'.format((tr_loss - logging_loss) / (step + 1)))
        logging_loss = tr_loss

        # Log metrics
        # total_rouge = evaluate(args, model, tokenizer, task, save_output=True) #Default
        total_rouge = evaluate(args, model, tokenizer, task, save_output=False)

        if total_rouge[2] > best_f1:
            # Save model checkpoint
            if task is not None:
                output_dir = os.path.join(args.output_dir, task, 'best_checkpoint')
            else:
                output_dir = os.path.join(args.output_dir, 'best_checkpoint')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            # model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logging.info("Saving model checkpoint to %s", output_dir)

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_task=None, save_output=False):
    # results = {}
    eval_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, len(args.device_id))

    all_metrics = []
    if eval_task is None:
        tasks = ['goal', 'know', 'item', 'resp']  # HJ know == topic??
    else:
        tasks = [eval_task]
    for task in tasks:
        task_eval_dataset = eval_dataset[task]
        eval_dataloader = DataLoader(DataFrame(task_eval_dataset, args), batch_size=args.eval_batch_size, shuffle=False,
                                     collate_fn=collate_fn)
        # Eval!
        logging.info("***** Running evaluation {} *****".format(task))
        logging.info("  Num examples = %d", len(task_eval_dataset))
        logging.info("  Batch size = %d", args.eval_batch_size)
        count = 0
        preds = []
        targets = []
        model_to_eval = model.module if hasattr(model, 'module') else model
        for batch in tqdm(eval_dataloader, desc="Evaluating", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            model.eval()
            with torch.no_grad():
                if task == 'item':
                    outputs = model_to_eval(
                        input_ids=batch['input_ids'].to(args.device),
                        attention_mask=batch['attention_mask'].to(args.device))
                    targets.extend(batch['item_ids'].tolist())
                    _, ranks = torch.topk(outputs, 50, dim=-1)
                    preds.extend(ranks.cpu().tolist())
                    # for rank in ranks.cpu().tolist():
                    #    preds.append([x for x in rank if x != 0][:50])
                else:
                    if task == 'resp':
                        beam_size = 1  # args.beam_size
                        max_length = args.max_target_length
                    else:  # task == goal
                        beam_size = 1
                        max_length = 32
                    generated_ids = model_to_eval.bart.generate(
                        input_ids=batch['input_ids'].to(args.device),
                        attention_mask=batch['attention_mask'].to(args.device),
                        num_beams=beam_size,
                        max_length=max_length,
                        # repetition_penalty=2.5,
                        # length_penalty=1.5,
                        early_stopping=True,
                    )
                    preds.extend([
                        tokenizer.decode(
                            g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for g in generated_ids
                    ])

                    targets.extend([
                        tokenizer.decode(
                            g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for g in batch['labels']
                    ])

        if save_output:
            if eval_task is None:
                output_dir = args.output_dir
            else:
                output_dir = os.path.join(args.output_dir, task)
            with open(os.path.join(args.output_dir, '{}_{}.decoded'.format(args.data_name, task)), 'w',
                      encoding='utf-8') as outfile, \
                    open(os.path.join(args.output_dir, '{}_{}.reference'.format(args.data_name, task)), 'w',
                         encoding='utf-8') as reffile:
                for p, t in zip(preds, targets):
                    outfile.write("{}\n".format(p))
                    reffile.write("{}\n".format(t))

        auto_scores = metrics.calculate(preds, targets, args.data_name, task, logging)
        logging.info(auto_scores)
        print(auto_scores)
        all_metrics += auto_scores
    return all_metrics

## HJ 여기가 Goal -> Topic -> Item -> Resp 로 Chain of Thought 해주는부분이다
def pipeline(args, model, tokenizer, save_output=False):
    # results = {}
    eval_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, len(args.device_id))

    all_metrics = []
    all_preds = {}
    for task in ['goal', 'know', 'item', 'resp']:
        if task == 'goal': # HJ: Goal 예측시에는 그대로 가져다가 씀
            task_eval_dataset = eval_dataset[task]
        else: # HJ: 그 이후것들은 이전것을 받아서 오는것 (all_pred)
            task_eval_dataset = data_reader.process_pipeline_data(args, tokenizer, eval_dataset, all_preds, task)
        eval_dataloader = DataLoader(DataFrame(task_eval_dataset, args), batch_size=args.eval_batch_size, shuffle=False,
                                     collate_fn=collate_fn)
        # Eval!
        logging.info("***** Pipeline Running evaluation {} *****".format(task))
        logging.info("  Num examples = %d", len(task_eval_dataset))
        logging.info("  Batch size = %d", args.eval_batch_size)
        count = 0
        preds = []
        targets = []
        tr_loss = []

        output_dir = os.path.join(args.output_dir, task, 'best_checkpoint')
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        model_to_eval = model.module if hasattr(model, 'module') else model

        for batch in tqdm(eval_dataloader, desc="Evaluating", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            model.eval()
            with torch.no_grad():
                if task == 'item':
                    outputs = model_to_eval(
                        input_ids=batch['input_ids'].to(args.device),
                        attention_mask=batch['attention_mask'].to(args.device))
                    targets.extend(batch['item_ids'].tolist())
                    _, ranks = torch.topk(outputs, 50, dim=-1)
                    preds.extend(ranks.cpu().tolist())
                    # for rank in ranks.cpu().tolist():
                    #    preds.append([x for x in rank if x != 0][:50])
                else:
                    if task == 'resp':
                        beam_size = args.beam_size
                        max_length = args.max_target_length
                    else: ## HJ : task == 'goal'
                        beam_size = 1
                        max_length = 32
                    generated_ids = model_to_eval.bart.generate(
                        input_ids=batch['input_ids'].to(args.device),
                        attention_mask=batch['attention_mask'].to(args.device),
                        num_beams=beam_size,
                        max_length=max_length,
                        # repetition_penalty=2.5,
                        # length_penalty=1.5,
                        early_stopping=True,
                    )
                    preds.extend([
                        tokenizer.decode(
                            g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for g in generated_ids
                    ])

                    targets.extend([
                        tokenizer.decode(
                            g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for g in batch['labels']
                    ])
                if task == 'resp':
                    model.train()
                    inputs = {'input_ids': batch['input_ids'].to(args.device),
                              'attention_mask': batch['attention_mask'].to(args.device),
                              'labels': batch['labels'].to(args.device),
                              'item_ids': batch['item_ids'].to(args.device)}
                    outputs = model(**inputs)
                    loss = outputs[2]
                    if len(args.device_id) > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    tr_loss.append(loss.item())
        # if save_output:
        #     with open(os.path.join(args.output_dir, task, '{}_{}.pipeline'.format(args.data_name, task)),
        #               'w') as outfile, \
        #             open(os.path.join(args.output_dir, task, '{}_{}.reference'.format(args.data_name, task)),
        #                  'w') as reffile:
        #         for p, t in zip(preds, targets):
        #             outfile.write("{}\n".format(p))
        #             reffile.write("{}\n".format(t))

        all_preds[task] = preds
        auto_scores = metrics.calculate(preds, targets, args.data_name, task, logging)
        if task == 'resp':
            print(sum(tr_loss) / len(tr_loss))
            ppl = math.exp(sum(tr_loss) / len(tr_loss))
            auto_scores += [ppl]
        logging.info(auto_scores)
        print(auto_scores)
        all_metrics += auto_scores
    return all_metrics


def main():

    parser = argparse.ArgumentParser(description="train.py")

    ## Required parameters
    parser.add_argument('--data_name', default='durecdial', type=str,
                        help="dataset name")
    parser.add_argument("--model_name_or_path", default='fnlp/bart-base-chinese',
                        type=str, help="model name or path")
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='../data', type=str,
                        help="The data directory.")
    parser.add_argument("--cache_dir", default='../temp_cache/bart', type=str,  # /projdata1/info_fil/ydeng/bert
                        help="The cache directory.")
    # HJ 특별추가 Parameters
    parser.add_argument("--log_name", default='', type=str,
                        help="Log fileName")  # HJ : Log file middle Name
    parser.add_argument("--log_dir", default='output/logs', type=str,
                        help="Log dir")  # HJ : Log dir
    parser.add_argument("--use_cached_data", default=False, type=bool,
                        help="Use cached data (tokenized dataset 활용여부)")  # HJ : Use cached data (tokenized dataset 활용여부)
    parser.add_argument("--save_tokenized_data", default=False, type=bool,
                        help="tokenized dataset 저장여부")  # HJ : tokenized dataset 저장여부
    parser.add_argument("--in_goal_with_goal_seq", default='T', type=str,
                        help="tokenizing with goal_sequence")  # HJ : tokenizing with goal_sequence
    parser.add_argument("--in_topic_with_goal_seq", default='T', type=str,
                        help="HJ : tokenized with topic_sequence")  # HJ : in_topic_with_goal_seq tokenized with topic_sequence
    parser.add_argument("--in_topic_with_topic_seq", default='T', type=str,
                        help="HJ : tokenized with topic_sequence")  # HJ : in_topic_with_topic_seq tokenized with topic_sequence

    ## Other parameters

    # parser.add_argument("--do_pure_generate", action='store_true',
    #                    help="Whether to use original model.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval.")
    parser.add_argument("--do_pipeline", action='store_true',
                        help="Whether to run pipeline eval.")
    parser.add_argument("--do_finetune", action='store_true',
                        help="Whether to run finetune.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    # parser.add_argument('--logging_steps', type=int, default=500,
    #                    help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=2000,
    #                    help="Save checkpoint every X updates steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_lower_case", action='store_false',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=100, type=int,
                        help="The maximum total output sequence length.")
    parser.add_argument('--beam_size', default=1, type=int)

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default="0", type=str,
                        help="Use CUDA on the device.")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=15, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_ft_epochs", default=5, type=float,
                        help="Total number of finetuning epochs to perform.")
    # parser.add_argument("--max_steps", default=-1, type=int,
    #                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    args = parser.parse_args()

    # HJ Desktop and Server Settings
    from platform import system as sysChecker
    if sysChecker() == 'Linux':  # HJ KT-server
        args.do_train, args.do_eval, args.do_finetune, args.overwrite_output_dir = True, True, True, True
        args.do_pipeline=True
        # args.gpu = '0'
        # args.gpu, args.num_train_epochs, args.num_ft_epochs = '0', 1, 1
        args.per_gpu_train_batch_size, args.per_gpu_eval_batch_size = 32, 32
        args.cache_dir = '../temp_cache/bart'
        args.data_dir = '/home/work/CRSTEST/UniMIND/data'
        args.use_cached_data, args.save_tokenized_data = False, False
        args.in_goal_with_goal_seq, args.in_topic_with_goal_seq = args.in_goal_with_goal_seq.upper()[0], args.in_topic_with_goal_seq.upper()[0]
        args.in_topic_with_topic_seq = args.in_topic_with_topic_seq.upper()[0]
        pass
    elif sysChecker() == "Windows":  # HJ local
        args.do_train, args.do_eval, args.do_finetune, args.overwrite_output_dir = False, False, False, True
        args.do_pipeline=True
        args.gpu, args.num_train_epochs, args.num_ft_epochs = '0', 1, 1
        args.per_gpu_train_batch_size = 1
        args.use_cached_data, args.save_tokenized_data = True, False
        pass
    else:
        print("Check Your Platform Setting")
        exit()

    args.output_dir = os.path.join(args.output_dir, args.data_name)
    # Create output directory if needed
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

    logging.basicConfig(level=logging.DEBUG,
                        filename=args.log_dir + f'/{get_time_kst()}_{args.log_name + "_"}log.txt',
                        filemode='a',
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        datefmt='%Y/%m/%d_%p_%I:%M:%S ',
                        )
    logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    device, device_id = utils.set_cuda(args)
    args.device = device
    args.device_id = device_id

    # Set seed
    utils.set_seed(args.seed)

    config = BartConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case,
                                              cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[goal]', '[user]', '[system]', '[knowledge]', '[item]',
                                                                '[profile]', '[history]']})

    ft_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=False)
    train_dataset = data_reader.merge_dataset(ft_dataset) # train_dataset은 다 때려박아주려고 만드는것이고, ['resp','goal','know','item']을 다 동시에 넣은것 맞음
    item_dict = train_dataset['item_dict']
    args.item_num = len(item_dict)

    model = UniMind(args, config, item_num=len(item_dict))
    model.bart.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    logging.info(get_time_kst())
    logging.info("Training/evaluation parameters %s", args)
    output_dir = os.path.join(args.output_dir, 'best_checkpoint')

    # Training
    if args.do_train:
        logging.info("")
        logging.info("")
        logging.info(get_time_kst())
        logging.info("do_train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        tokenizer.save_pretrained(output_dir)

    # Fine-tuning
    if args.do_finetune:
        for task in ['goal', 'know', 'item', 'resp']:
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
            else:
                model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))

            logging.info("")
            logging.info("")
            logging.info(get_time_kst())
            logging.info(" Fine-tuning task %s", task)
            global_step, tr_loss = train(args, ft_dataset[task], model, tokenizer, task)
            logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    if args.do_eval:
        # Load a trained model and vocabulary that you have fine-tuned
        # model = BartForConditionalGeneration.from_pretrained(output_dir)
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            # model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'))) # Default
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'),
                                             map_location=str(args.device).split()[0]))  # HJ
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)
        logging.info("")
        logging.info("")
        logging.info(get_time_kst())
        logging.info(" Eval task ")
        logging.info("")
        evaluate(args, model, tokenizer, save_output=True) # Evaluation

    # Pipeline Evaluation
    if args.do_pipeline:
        tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)
        pipeline(args, model, tokenizer, save_output=True)
    logging.info("THE END")


if __name__ == "__main__" :
    # logging.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
    main()
