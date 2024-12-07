import os
import torch
import json
import time
import logging
import random
import argparse
import numpy as np
import itertools
from typing import List
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer, \
    GPT2LMHeadModel, AutoModel
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from arguments import get_args
from policy_marian import Policy
from data_pool import DataPool
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness
# from token_gpt2 import Distill_Tuning
# from Sentiment.main_disc import Scorer
from Sentiment.main_disc import Scorer_topic, Scorer_sent

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self):
        dataset = json.load(open('data/double/pos_asian.json'))
        dataset = random.sample(dataset, 20000)
        dataset = [i.strip() for i in dataset]

        self.dataset = [i for i in dataset if i != '']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class TrainCollator(object):
    def __init__(self, tokenizer, max_source_length):
        self.max_source_length = max_source_length
        self.tokenizer = tokenizer
        self.scorer1 = Scorer_sent('cuda')
        self.scorer2 = Scorer_topic('cuda')
        
    def __call__(self, sequences):
        res_input = self.tokenizer.batch_encode_plus(sequences, max_length=self.max_source_length, return_tensors="pt",
                                                     truncation=True, padding="max_length")
        res_input = {k: v.cuda() for k, v in res_input.items()}
        # scores1=torch.zeros_like(res_input['input_ids']).cuda()
        # scores2=torch.zeros_like(res_input['input_ids']).cuda()
        scores1, scores2 = [], []
        for i in range(res_input['input_ids'].shape[-1]):
            # if (res_input['attention_mask'][:,i:]==0).all():
            #     break
            score_1 = self.scorer1.score(res_input['input_ids'][:, :i])
            score_2 = self.scorer2.score(res_input['input_ids'][:, :i])

            scores1.append(score_1.detach())
            scores2.append(score_2.detach())

        scores1 = torch.stack(scores1).T
        scores2 = torch.stack(scores2).T
        scores = torch.cat([scores1[..., None], scores2[..., None]], dim=-1)
        return res_input['input_ids'], res_input['attention_mask'], scores


class Weight_Classifier(nn.Module):
    def __init__(self, hidden_size, label_nums):
        super().__init__()
        self.hidden_size = hidden_size
        self.label_nums = label_nums
        self.classifier_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.regression_head = nn.Linear(self.hidden_size, self.label_nums, bias=False)

    def forward(self, hidden_states):
        states = self.classifier_layer(hidden_states)
        weight = self.regression_head(states)
        weight = torch.softmax(weight, dim=-1)
        # weight = (torch.sigmoid(weight)-0.5)*2
        return weight


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'device:{device}')

    time = datetime.now()
    # date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
    date_time = time.strftime("%m-%d-%Y")
    args.save_dir = os.path.join(args.output_dir, date_time)
    args.reward_dir = os.path.join(args.save_dir, 'reward')
    args.model_dir = os.path.join(args.save_dir, 'model')
    args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
        ensure_dir(d)
    log.info(f'Write to output directory: {args.save_dir}')

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    log.info(f'Initializing models ...')
    model = AutoModel.from_pretrained('/home/lwd/gpt2-large').cuda()
    model.eval()
    weight_classifier = Weight_Classifier(model.config.hidden_size, 2).cuda()
    tokenizer = AutoTokenizer.from_pretrained('/home/lwd/gpt2-large')
    tokenizer.pad_token = tokenizer.eos_token

    log.info(f'Initialization done!')

    # prompt_collator = TrainCollator(tokenizer=tokenizer,max_source_length=128)
    prompt_collator = TrainCollator(tokenizer=tokenizer, max_source_length=128)
    train_dataset = TrainDataset()
    log.info(f'Load train set with {len(train_dataset)} examples')

    # set up optimizer and scheduler
    parameters2update = [para for name, para in weight_classifier.named_parameters()]
    optimizer = Adam(parameters2update, lr=args.lr, eps=1e-5)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=args.total_steps)

    step = 0
    epoch = 4
    for i in range(epoch):
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size // 4, shuffle=False, drop_last=True,
                                      collate_fn=prompt_collator)

        for input_ids, attn_mask, scores in tqdm(train_dataloader):

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            weight = weight_classifier(outputs['last_hidden_state'][:, 1:, :])

            scores = scores[:, 1:, :] - scores[:, :-1, :]

            loss = - ((weight * scores).sum(-1) * attn_mask[:, 1:]).sum()  # /attn_mask[:,1:].sum(-1)).sum()
            if step % 50 == 0:
                print(f"{i}:current_loss={-loss}")
            loss.backward()
            optimizer.step()
            scheduler.step()

        torch.save(weight_classifier.state_dict(), f'checkpoints/weight_classifier/model_large_{i}.pt')


if __name__ == "__main__":
    main()
