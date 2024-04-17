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
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
# from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup,AutoModelForSequenceClassification,AutoTokenizer

from arguments import get_args
# from policy_marian import Policy
from data_pool import DataPool
# from reward import Reward, reward_to_toxicity
# from fudge_reward import RewardScore
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum, distinctness
from token_gpt2 import Distill_Tuning
from Sentiment.main_disc import Classifier
# from nltk import sent_tokenize

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger(__name__)

# def process_openwebtext_into_train_prompts:
#     from datasets import load_dataset
#     dataset = load_dataset('openwebtext')['train'][:50000]
#     dataset = [i['text'] for i in dataset]
#     neutral_num,negative_num,positive_num = 10000,10000,10000
#     model = AutoModelForSequenceClassification.from_pretrained(
#         'cardiffnlp/twitter-roberta-base-sentiment-latest')
#     token = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
#     input_ids_list,attention_mask_list=[],[]
#     for text in dataset:
#         t = ' '.join(text.split()[:20])
#         inp = token(t,return_tensors='pt')
#         input_ids_list.append('inp')
class PromptDataset(Dataset):
    # def __init__(self, args, mode):
    #     # self.prompts = [json.loads(s.strip())["prompt"]["text"].strip() for s in open(path, 'r').readlines()][:100]
    #     num_reference = 2
    #
    #     valid_es = []
    #     train_es = []
    #     test_es = []
    #     train_en = []
    #     valid_en = [[]] * num_reference
    #     test_en = [[]] * num_reference
    #
    #     with open('../NADO/fisher-callhome-corpus/corpus/ldc/fisher_test.es', 'r') as f:
    #         for line in f:
    #             test_es.append(line.strip())
    #
    #     with open('../NADO/fisher-callhome-corpus/corpus/ldc/fisher_train.es', 'r') as f:
    #         for line in f:
    #             train_es.append(line.strip())
    #
    #     with open('../NADO/fisher-callhome-corpus/corpus/ldc/fisher_dev.es', 'r') as f:
    #         for line in f:
    #             valid_es.append(line.strip())
    #
    #     with open('../NADO/fisher-callhome-corpus/corpus/ldc/fisher_train.en', 'r') as f:
    #         for line in f:
    #             train_en.append(line.strip())
    #
    #     for i in range(num_reference):
    #         with open('../NADO/fluent-fisher/noids/dev.noid.cleaned_%d' % (i), 'r') as f:
    #             for line in f:
    #                 # clean_line = line.strip().split()[1:]
    #                 # clean_line = " ".join(clean_line)
    #                 valid_en[i].append(line.strip())
    #
    #     for i in range(num_reference):
    #         with open('../NADO/fluent-fisher/noids/test.noid.cleaned_%d' % (i), 'r') as f:
    #             for line in f:
    #                 # clean_line = line.strip().split()[1:]
    #                 # clean_line = " ".join(clean_line)
    #                 test_en[i].append(line.strip())
    #
    #     if mode=='train':
    #         self.dataset = [i for i in train_es]
    #         self.ref = train_en
    #         self.dataset = self.dataset[:320]
    #     elif mode =='valid':
    #         self.dataset = [i for i in valid_es]
    #         self.ref = valid_en
    #     else:
    #         self.dataset = [i for i in test_es]
    #         self.ref = test_en
    def __init__(self,mode='positive',train=True):
        if train==True:
            dataset = json.load(open('Sentiment/data/train_prompts_v1.json'))
            dataset = random.sample(dataset,12000)
            self.dataset = [i for _ in range(1) for i in dataset]
        else:
            path = 'Sentiment/data/sentiment_prompts-10k/' + mode + '_prompts.jsonl'
            with open(path) as f:
                dataset = [json.loads(line)['prompt']['text'] for line in f]
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class PromptCollator(object):
    def __init__(self, tokenizer, max_source_length):
        self.max_source_length = max_source_length
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        res_input = self.tokenizer.batch_encode_plus(sequences, max_length=self.max_source_length, return_tensors="pt",
                                               truncation=True, padding="max_length")
        return res_input['input_ids'],res_input['attention_mask']

class SequenceDataset(Dataset):
    def __init__(self, data_pool: DataPool):
        self.ids, self.masks, self.cat_mask = data_pool.get_data()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {'input_ids': self.ids[idx],
                'output_mask':self.masks[idx],
                'cat_mask': self.cat_mask[idx]
                }

class SequenceCollator(object):
    def __init__(self, args,tokenizer):
        self.tokenizer = tokenizer
        self.device = args.device
    # def __call__(self, sequences):
    #     queries = [sequence['query'] for sequence in sequences]
    #     responses = [sequence['response'] for sequence in sequences]
    #     # cat_ids = [self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens']) for sequence in sequences]
    #
    #     query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
    #     query_input_ids = query_encodings_dict['input_ids']
    #     query_mask = query_encodings_dict['attention_mask']
    #     # query_input_ids = torch.cat([query_input_ids.new(cat_ids)[:, None], query_input_ids], dim=1)
    #     # query_mask = torch.cat([query_mask.new([1] * len(query_mask))[:, None], query_mask], dim=1)
    #
    #     response_encodings_dict = self.tokenizer(responses, return_tensors="pt", padding=True)
    #     response_input_ids = response_encodings_dict['input_ids']
    #     response_mask = response_encodings_dict['attention_mask']
    #
    #     cat_mask = [sequence['cat_mask'] for sequence in sequences]
    #     mask_tensor,len_list = self.pad_tensor(cat_mask)
    #     assert len_list == response_mask.sum(-1).to_list()
    #
    #     return query_input_ids, query_mask, response_input_ids, response_mask, mask_tensor
    #
    def __call__(self,sequences):
        input_ids = torch.tensor([sequence['input_ids'] for sequence in sequences],device=self.device)
        input_mask = input_ids != self.tokenizer.eos_token_id
        output_mask = torch.tensor([sequence['output_mask'] for sequence in sequences],device=self.device)
        reward_mask = torch.tensor([sequence['cat_mask'] for sequence in sequences],device=self.device)
        # punish_mask = torch.tensor([sequence['cat_mask'] for sequence in sequences],device=self.device) == -1
        return input_ids,input_mask,output_mask,reward_mask

    def pad_tensor(self,mask_list):
        len_list = [len(mask) for mask in mask_list]
        max_len = max(len_list)
        tensor_stack = [mask+[0]*(max_len-len(mask)) for mask in mask_list]
        padded_tensor = torch.tensor(tensor_stack)
        return padded_tensor,len_list

class FixedController:
    def __init__(self, coef):
        self.value = coef

    def update(self, current, n_steps, lower_bound):
        pass


class AdaptiveController:
    def __init__(self, init_coef, target, horizon):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps, lower_bound):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        if lower_bound:
            mult = 1 + proportional_error * n_steps / self.horizon
        else:
            mult = 1 - proportional_error * n_steps / self.horizon
        self.value *= mult


class Evaluation():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('/home/lwd/distilbert-base-uncased-finetuned-sst-2-english')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            '/home/lwd/distilbert-base-uncased-finetuned-sst-2-english')

    def eval(self,text,target='positive'):
        inputs = self.tokenizer(text,return_tensors='pt',padding=True,truncation=True)
        output = self.model(**inputs)
        predicted_class_id = output.logits.argmax(-1)
        labels = [self.model.config.id2label[i] for i in predicted_class_id.tolist()]
        nums = [1 for i in labels if i.lower()==target]
        return sum(nums),len(labels)

    def score(self,text,target='POSITIVE'):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        output = self.model(**inputs)
        #todo check whether has been softmax?
        id = self.model.config.label2id[target]
        scores = output.logits[:,id]
        return scores

class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy,
                 ref_policy,
                 data_pool: DataPool,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR):

        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.data_pool = data_pool
        # self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # self.writer = SummaryWriter()
        self.q_record=[]

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveController(self.params.kl_coef, self.params.target_kl, self.params.horizon)
        else:
            self.kl_ctl = FixedController(self.params.kl_coef)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        if self.params.adaptive_entropy:
            self.entropy_ctl = AdaptiveController(self.params.entropy_coef, self.params.target_entropy,
                                                  self.params.horizon)
        else:
            self.entropy_ctl = FixedController(self.params.entropy_coef)

        # self.tree_tokens = tree_tokens
        # self.best_cat = self.tree_tokens[0]
        # self.best_cat_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat)
        # self.pos_id,self.neg_id,self.pad_id=special_ids
        # self.special_tokens_num = len(special_ids)-1

        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceCollator(self.params,tokenizer=policy.tokenizer)
        self.classifier = Evaluation()
        self.best_correctness = 0
        self.best_distinct = []


    def add_control_code(self, input_ids, attention_mask):
        input_ids = torch.cat([input_ids.new([self.pad_id] * len(input_ids))[:, None], input_ids], dim=1)
        pos_ids = torch.cat([input_ids.new([self.pos_id] * len(input_ids))[:, None], input_ids], dim=1)
        neg_ids = torch.cat([input_ids.new([self.neg_id] * len(input_ids))[:, None], input_ids], dim=1)
        attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
        attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
        return input_ids,neg_ids,attention_mask

    def decode(self, query_input_ids, response_input_ids=None):
        query = [self.policy.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for p in query_input_ids]

        if response_input_ids is None:
            return query

        response = [self.policy.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for r in response_input_ids]
        return query, response

    def sample(self, step):
        if step % self.params.sample_interval != 0:
            return
        log.info(f"[step {step}] Sampling ...")

        # prompts, responses = [], []
        # q_inputs,d_inputs,d_len = [],[],[]
        text_list,input_list,mask_list,q_values=[],[],[],[]
        for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader),
                                       desc='Sampling from current policy')):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.params.device)
            attention_mask = attention_mask.to(self.params.device)

            if step == 0:
                # rollouts = self.ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask, top_p=self.params.top_p)
                rollouts = self.ref_policy.sample(prompts_ids=input_ids, max_length=20+self.params.max_prompt_length,mode=None)
                text,input_ids,mask,q_value = rollouts['text'],rollouts['input_ids'],rollouts['output_mask'],rollouts['q_values']
                # prompt, response = rollouts['query/text'], rollouts['response/text']
                # d_len.extend((rollouts['response/input_ids'][:, 1:] != self.ref_policy.tokenizer.pad_token_id).sum(-1).tolist())
            else:
                # pos_ids,neg_ids,attention_mask = self.add_control_code(input_i
                # ds, attention_mask)
                rollouts = self.policy.sample(prompts_ids=input_ids, max_length=20+self.params.max_prompt_length, mode='positive')
                text, input_ids, mask,q_value = rollouts['text'], rollouts['input_ids'], rollouts['output_mask'],rollouts['q_values']
                # prompt = rollouts['query/text']
                # d_len.extend((rollouts['response/input_ids'][:, 2:] != self.policy.tokenizer.pad_token_id).sum(-1).tolist())

            # prompts.extend(prompt)
            # responses.extend(response)
            text_list.extend(text)
            input_list.extend(input_ids.tolist())
            mask_list.extend(mask.tolist())
            q_values.extend(q_value.tolist())
            #todo log_probs
        # input_ids,rewards,sen_mask = self.score_model.score(input_list,mode='positive')
        # aa=1
        # assert input_ids.tolist() == input_list
        self.data_pool.add(input_list, mask_list, q_values, pos=True)
        self.q_record.append(self.data_pool.r_limit)
        print(self.q_record)
        sample_dataset = SequenceDataset(data_pool=self.data_pool)
        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                            shuffle=False, drop_last=True, collate_fn=self.seq_collator)
        self.sampler = iter(self.sample_dataloader)

    def step(self, step_num):

        with torch.no_grad():
            self.eval(step=step_num)
            self.sample(step=step_num)
        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.sampler = iter(self.sample_dataloader)
            batch = next(self.sampler)

        self.optimizer.zero_grad()
        ppo_loss = self.loss(step_num, *batch)
        ppo_loss.backward()
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()


    def loss(self, step, input_ids, input_mask, output_mask, reward_tensor):
        self.policy.model.train()
        outputs = self.policy.forward_pass(input_ids, input_mask, output_mask, reward_tensor, mode='positive')
        lm_loss, logits,entropy = outputs['loss'],outputs['logits'],outputs['entropy']
        # logits = outputs['response/logits'][:, :, :-len(self.special_tokens_num)]
        kl_mask = outputs['kl_mask']

        with torch.no_grad():
            ref_outputs = self.ref_policy.forward_pass(input_ids, input_mask, output_mask,mode=None)
            ref_logits = ref_outputs['logits']
            pad_logits = torch.zeros(ref_logits.shape[0],self.policy.spell_length,ref_logits.shape[-1],device=self.params.device)
            ref_logits = torch.cat([pad_logits,ref_logits],dim=1)

        # kl = torch.sum(self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1)), dim=-1)
        kl = torch.sum(
            torch.softmax(ref_logits, dim=-1) * (F.log_softmax(ref_logits, dim=-1) - F.log_softmax(logits, dim=-1)),
            dim=-1)

        loss = lm_loss + reduce_mean(- self.entropy_ctl.value * entropy, kl_mask) + reduce_mean(self.kl_ctl.value * kl, torch.ones_like(kl_mask))

        # kl_loss = reduce_mean(self.kl_ctl.value * kl, kl_mask)
        # loss = lm_loss + kl_loss



        # queries = self.decode(input_ids)
        # self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1),
        #                    logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step=step)
        # self.print_samples(queries=queries, lm_loss=reduce_mean(lm_loss, masks, axis=1),
        #                    loss=loss,step=step)
        # r_loss = reduce_mean(lm_loss,masks)
        # r_kl_loss = reduce_mean(self.kl_ctl.value * kl,masks)
        if step % self.params.log_interval ==0:
            log.info(f"[step {step}] lm_loss={lm_loss:.4f}, kl={reduce_mean(kl,kl_mask):.4f},entropy={reduce_mean(- self.entropy_ctl.value * entropy, kl_mask):.4f}")


        return loss

    def record_step_stats(self, data):
        masks = data['masks']
        stats = {}
        # kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        # mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        # mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        # stats = {
        #     'objective/kl': mean_kl.item(),
        # }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
        })
        # stats = {
        #     'objective/kl': mean_kl.item(),
        #     'objective/entropy': mean_entropy.item(),
        # }
        # stats.update({
        #     'loss/total': data['total_loss'].item(),
        #     'loss/kl': data['kl_loss'].item(),
        #     'loss/lm': data['lm_loss'].item(),
        #     'loss/entropy': data['entropy'].item(),
        # })
        return stats

    def print_samples(self, queries, lm_loss, loss, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(3, len(queries))):
            # sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(queries[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            # print(f"  total_loss = {loss[i].item():+.2f}")
            # print(f"  kl = {sample_kl:+.2f}")
            # print(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

    def save(self, mode):
        # if step % self.params.save_interval != 0:
        #     return
        torch.save({
            'prompt_encoder_pos': self.policy.prompt_encoder_pos.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, f'{self.params.model_dir}/ckp_{mode}.pth')
        log.info(f"model checkpoint saved once")

    def load(self,load_dir):
        load_dic = torch.load(load_dir)
        self.policy.prompt_encoder_pos.load_state_dict(load_dic['prompt_encoder_pos'])
        self.optimizer.load_state_dict(load_dic['optimizer'])
        self.scheduler.load_state_dict(load_dic['scheduler'])
        log.info(f"Load Model Successfully from {load_dir}")

    def eval(self, step):
        if  step % self.params.eval_interval != 0:
            return
        self.policy.model.eval()
        log.info(f"[step {step}] evaluating ...")
        generations, perplexities, toxicities = [], [], []
        correct,count = 0,0
        for i, (input_ids, attention_mask) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                input_ids = input_ids.to(self.params.device)
                attention_mask = attention_mask.to(self.params.device)
                # input_ids, attention_mask = self.add_control_code(input_ids, attention_mask)
                rollouts = self.policy.sample(prompts_ids=input_ids,max_length=20 + self.params.max_prompt_length,mode='positive',gen=True)
                cur_cast_mask = torch.ones_like(rollouts['input_ids'])[:,:-1]
                forward_inputs = {'x_hs':rollouts['input_ids'],
                                  'att_mask':rollouts['input_ids']!=self.policy.tokenizer.pad_token_id,
                                  'out_mask':rollouts['output_mask'],
                                  'reward_mask':cur_cast_mask}
                outputs = self.policy.forward_pass(**forward_inputs,mode='positive',gen=True)
                ref_logprobs = outputs['logprob']
                ## prompt = self.decode(rollouts['query/input_ids'][:, 1:])
                # response = rollouts['response/text']
                # score = self.score_model.get_reward(prompt, response, f'step{step}_eval{i}')
                # toxicity = [reward_to_toxicity(x) for x in score if x is not None]
                # toxicities.extend(toxicity)
                generations.extend(rollouts['text'])

                x1,x2 = self.classifier.eval(rollouts['text'],target=self.params.target_mode)
                correct += x1
                count += x2

        correctness = correct/count

        dist_1, dist_2, dist_3, dist_4 = distinctness(generations)
        log.info('*******************************')
        log.info(f"  correctness = {correctness:+.4f}")
        log.info(f'dist-1={dist_1:.3f}, dist-2={dist_2:.3f}, dist-3={dist_3:.3f}, dist-4={dist_4:.3f}')
        log.info('***example***')
        log.info(generations[-1])
        log.info(generations[-2])
        log.info(generations[-3])
        log.info(generations[-4])
        log.info(generations[-5])
        log.info(generations[-6])
        log.info('******************************')

        result = f"cor={correctness:+.3f}+step={step}+dist={dist_1:.3f}-{dist_2:.3f}-{dist_3:.3f}-{dist_4:.3f}"
        # if correctness > self.best_correctness:
        self.save(result)
        self.best_correctness = correctness
        # self.writer.add_scalar('Evaluation/perplexity', ppl_score, step)
        # self.writer.add_scalar('Evaluation/toxicity', toxicity_score, step)
        # self.writer.add_scalar('Evaluation/Dist-1', dist_1, step)
        # self.writer.add_scalar('Evaluation/Dist-2', dist_2, step)
        # self.writer.add_scalar('Evaluation/Dist-3', dist_3, step)



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


    device= 'cuda'

    time = datetime.now()
    # date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
    date_time = time.strftime("%m-%d-%Y")
    args.save_dir = os.path.join(args.output_dir, date_time)
    args.reward_dir = os.path.join(args.save_dir, 'reward')
    args.model_dir = os.path.join(args.save_dir, 'model-fudge-way')
    args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
    for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
        ensure_dir(d)
    log.info(f'Write to output directory: {args.save_dir}')

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    log.info(f'Initializing models ...')

    label_token = {"positive": 'good', "negative": 'bad','neutral':'neutral'}
    ref_policy = Distill_Tuning(args,args.template,label_token)
    policy = Distill_Tuning(args, args.template, label_token)

    data_pool = DataPool()
    log.info(f'Initialization done!')

    prompt_collator = PromptCollator(tokenizer=ref_policy.tokenizer,max_source_length=args.max_prompt_length)
    train_dataset = PromptDataset(mode='neutral',train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=prompt_collator)
    log.info(f'Load train set with {len(train_dataset)} examples')

    val_dataset = PromptDataset(mode='neutral',train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=prompt_collator)
    log.info(f'Load val set with {len(val_dataset)} examples')

    # set up optimizer and scheduler
    parameters2update = [para for name,para in policy.named_parameters() if 'prompt' in name]
    optimizer = Adam(parameters2update, lr=args.lr, eps=1e-5)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)

    # special_ids = [policy.tokenizer.get_vocab()['__pos__'],policy.tokenizer.get_vocab()['__neg__'],policy.tokenizer.pad_token_id]
    trainer = ConditionTrainer(params=args, policy=policy, ref_policy=ref_policy, data_pool=data_pool,
                               train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               optimizer=optimizer, scheduler=scheduler)


    for step_num in range(100000):
        if step_num>30000:
            trainer.save_result()
            break
        trainer.step(step_num)



if __name__ == "__main__":
    main()
