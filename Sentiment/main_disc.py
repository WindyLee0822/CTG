import json
import os
import torch
import argparse
import random
import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from transformers import pipeline, set_seed

from os.path import join, abspath, dirname
from Sentiment.data import Classification_Dataset, SentimentPrompt, DetoxicDataset, Sentiment_Suffix, GPT2Label
from Sentiment.discriminator import PTuneForLAMA
# from data import Classification_Dataset, SentimentPrompt, DetoxicDataset, Sentiment_Suffix, GPT2Label
# from discriminator import PTuneForLAMA

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--model_name_or_path", type=str, default='/home/lwd/gpt2-base')

    parser.add_argument("--data_path", type=str, default='data/pos_neg')

    parser.add_argument("--embedding_checkpoint", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="sentiment", choices=["detoxic", "sentiment"])

    parser.add_argument("--pseudo_token", type=str, default='xxx')

    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--epoch", type=int, default=50)

    parser.add_argument("--template", type=str, default="(2, 2)")
    parser.add_argument("--early_stop", type=int, default=20)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=True)

    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), './checkpoint'))
    # MegatronLM 11B

    ## generation configure
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--max_prompt_length", type=int, default=10)

    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--prompt_type", type=str, default="negative")
    parser.add_argument("--target_type", type=str, default="positive")

    parser.add_argument("--prompt_pad_length", type=int, default=10)
    # parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--ranking_scope", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--file_name", type=str, default="./eval")
    parser.add_argument("--mode", type=str, default="train", choices=["ctg", "train", "classifer"])
    parser.add_argument("--evaluate_file", type=str, default="../our_text")
    parser.add_argument("--evaluate_outfile", type=str, default="./eval/our/result.csv")
    parser.add_argument("--iter_num", type=int, default=10)
    parser.add_argument("--corpus_type", type=str, default="positive")
    parser.add_argument("--tuning_name", type=str, default="disc_tuning",
                        choices=["prompt_tuning", "disc_tuning", "distill_tuning"])

    ## discriminator information for distilled tuning
    parser.add_argument("--disc_embedding_checkpoint", type=str, default=None)
    parser.add_argument("--template_disc", type=str, default="(2, 3)")

    args = parser.parse_args()
    # post-parsing args

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    args.template_disc = eval(args.template_disc) if type(args.template_disc) is not tuple else args.template_disc

    assert type(args.template) is tuple

    seed_everything(args.seed)

    return args

class Scorer():
    def __init__(self,device):
        n_args = construct_generation_args()
        n_args.device = device
        self.args = n_args
        self.label_token = {"positive": 'good', "negative": 'bad'}
        self.model = PTuneForLAMA(n_args, n_args.template, label_token=self.label_token)
        # ckpt = torch.load('/home/lwd/quark/Sentiment/checkpoint/fudge/disc_tuning_positive_temperature0.01_scope_50_epoch_5_f1_0.88_(2,2).ckpt')['embedding']
        ckpt = torch.load('/home/lwd/quark/Sentiment/checkpoint/train-in-fudge-way/disc_tuning_positive_temperature0.01_scope_50_epoch_2_f1_0.85_(2,2).ckpt')['embedding']
        self.model.load_state_dict(ckpt)
        self.tokenizer = self.model.tokenizer

    def score(self,input_ids,mode='positive'):
        score = self.model._predict_scores(input_ids, input_ids!=self.tokenizer.pad_token_id, reward=True)
        return score
    # def score(self,input_ids,mode='positive'):
    #     dataset = Classification_Dataset(tokenizer=self.tokenizer, data_dir=input_ids,label_token=self.label_token,max_length=35)
    #
    #     data_loader = DataLoader(dataset, self.args.batch_size, shuffle=False)
    #     input_list,rewards_list=[],[]
    #     with torch.no_grad():
    #         self.model.eval()
    #         for batch in data_loader:
    #             self.model.eval()
    #             x = batch[0].to(self.args.device).squeeze(1)
    #             musk = batch[1].to(self.args.device).long().squeeze(1)
    #             y = batch[2]
    #             max_length = musk.sum(1).max()
    #             scores=torch.zeros_like(x,dtype=torch.float32)
    #             sen_mask = self.model._predict_scores(x, musk)
    #             if mode == 'positive':
    #                 sen_mask = sen_mask == 11274
    #             else:
    #                 sen_mask = sen_mask == 14774
    #             for l in range(max_length):
    #                 state_score = self.model._predict_scores(x[:,:l+1], musk[:,:l+1], reward=True)
    #                 scores[:,l] = state_score
    #             rewards = scores[:,1:] - scores[:,:-1]
    #             input_list.extend(x.tolist())
    #             rewards_list.extend(rewards.tolist())
    #     return input_list,rewards_list,sen_mask

class Classifier():
    def __init__(self,args):
        n_args = construct_generation_args()
        n_args.device = args.device
        self.args = n_args
        self.label_num = 2
        if self.label_num==3:
            self.label_token = {"positive": 'good', "negative": 'bad','neutral':'neutral'}
        elif self.label_num==2:
            self.label_token = {"positive": 'good', "negative": 'bad'}

        self.model = PTuneForLAMA(n_args, n_args.template, label_token=self.label_token)
        # used for quark
        # ckpt = torch.load(
        #     'Sentiment/checkpoint/prompt_model/disc_tuning_positive_temperature0.01_scope_50_epoch_2_f1_0.81_(2,2).ckpt')['embedding']
        # self.model.load_state_dict(ckpt)
        #used for fudge
        ckpt = torch.load(
            '/home/lwd/quark/Sentiment/checkpoint/fudge/disc_tuning_positive_temperature0.01_scope_50_epoch_5_f1_0.88_(2,2).ckpt')['embedding']
        self.model.load_state_dict(ckpt)
        self.tokenizer = self.model.tokenizer

    def get_past_key_values(self,input_ids):
        attn_mask = input_ids != self.tokenizer.pad_token_id
        pkv=self.model.get_past_key_values(input_ids,attn_mask)
        return pkv

    def get_q_value(self,input_ids,add_ids,past_key_values,mean=True):
        past_key_values_list = []
        bsz = add_ids.shape[0]
        sub_bsz = add_ids.shape[1]
        for i in range(bsz):
            pkv_ele =[]
            for tp in past_key_values:
                pkv_ele.append((tp[0][i].unsqueeze(0).repeat(sub_bsz,1,1,1),tp[1][i].unsqueeze(0).repeat(sub_bsz,1,1,1)))
            past_key_values_list.append(pkv_ele)

        r_storage = []
        for i in range(bsz):
            attn_mask = torch.cat([input_ids[i]!=self.tokenizer.pad_token_id,torch.ones(1,device=input_ids.device)],dim=-1)
            attn_mask = attn_mask.unsqueeze(0).repeat(sub_bsz,1)
            r_value = self.model.forward_with_pkv(add_ids[i],attn_mask,past_key_values_list[i],mean).view(-1)
            r_storage.append(r_value)

        return torch.stack(r_storage)

    def get_next_contrast_token(self,input_ids,add_ids,past_key_values,mean=False):
        past_key_values_list = []
        bsz = add_ids.shape[0]
        # print(input_ids.shape,add_ids.shape)
        sub_bsz = add_ids.shape[1]
        for i in range(bsz):
            pkv_ele =[]
            for tp in past_key_values:
                pkv_ele.append((tp[0][i].unsqueeze(0).repeat(sub_bsz,1,1,1),tp[1][i].unsqueeze(0).repeat(sub_bsz,1,1,1)))
            past_key_values_list.append(pkv_ele)

        r_storage = []
        for i in range(bsz):
            attn_mask = torch.cat([input_ids[i]!=self.tokenizer.pad_token_id,torch.ones(1,device=input_ids.device)],dim=-1)
            attn_mask = attn_mask.unsqueeze(0).repeat(sub_bsz,1)
            r_value = self.model.forward_with_pkv(add_ids[i][...,None],attn_mask,past_key_values_list[i],mean).view(-1)
            r_storage.append(r_value)

        return torch.stack(r_storage)



class Trainer(object):
    def __init__(self, args):
        self.args = args

        # self.label_token ={
        #   "positive":'good',
        #   "negative":'bad'
        # }
        label_kind = 2
        assert self.args.tuning_name == "disc_tuning"
        if label_kind ==2:
            self.label_token = {"positive": 'good', "negative": 'bad'}
        elif label_kind==3:
            self.label_token = {"positive": 'good', "negative": 'bad','neutral':'neutral'}
        self.model = PTuneForLAMA(args, args.template, label_token=self.label_token)

        self.tokenizer = self.model.tokenizer
        data_path = args.data_path

        if self.args.task_name == "sentiment":
            print(self.args.tuning_name)

            if self.args.tuning_name == "disc_tuning" or self.args.tuning_name == "distill_tuning":
                all_dataset = Classification_Dataset(tokenizer=self.tokenizer, data_dir=data_path, max_length=30,
                                                     type_path="train", label_token=self.label_token)

            else:

                all_dataset = Sentiment_Suffix(tokenizer=self.tokenizer, data_dir=data_path, max_length=30,
                                               task_type=self.args.corpus_type, label_token=self.label_token)

        elif self.args.task_name == "detoxic":
            print("load detoxic dataset!!!")

            if self.args.tuning_name == "disc_tuning" or self.args.tuning_name == "distill_tuning":

                all_dataset = DetoxicDataset(tokenizer=self.tokenizer, data_dir=data_path, max_length=30,
                                             type_path="train", label_token=self.label_token)

            else:
                all_dataset = Sentiment_Suffix(tokenizer=self.tokenizer, data_dir=data_path, max_length=30,
                                               task_type=self.args.corpus_type, label_token=self.label_token)
        # all_dataset = GPT2Label(tokenizer=self.tokenizer, data_dir=data_path, max_length=20)

        train_size = int(len(all_dataset) * 0.9)
        test_size = len(all_dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.split(all_dataset, [train_size, test_size])
        train_dataset = torch.utils.data.Subset(all_dataset, range(train_size))
        test_dataset = torch.utils.data.Subset(all_dataset, range(train_size, train_size + test_size))
        self.train_loader = DataLoader(train_dataset, args.batch_size, num_workers=2, shuffle=True)
        self.test_loader = DataLoader(test_dataset, args.batch_size, num_workers=2, shuffle=True)

    def evaluate(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
        else:
            loader = self.dev_loader
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for batch in loader:
                self.model.eval()
                x = batch[0].cuda().squeeze(1)
                musk = batch[1].cuda().long().squeeze(1)
                y = batch[2]

                pred_ids = self.model._predict_scores(x, musk)

                preds += pred_ids
                labels += y.tolist()

            result = self.disc_metric(labels, preds)
            print('*********precision:{}**********'.format(result))
        return result

    # def evaluate(self, epoch_idx, evaluate_type):
    #     self.model.eval()
    #     if evaluate_type == 'Test':
    #         loader = self.test_loader
    #     else:
    #         loader = self.dev_loader
    #     scores,totals = 0,0
    #     with torch.no_grad():
    #         self.model.eval()
    #         for batch in loader:
    #             self.model.eval()
    #
    #             x = batch[0].cuda().squeeze(1)
    #             musk = batch[1].cuda().long().squeeze(1)
    #             y = batch[2]
    #
    #             ave,nums = self.model(x, musk)
    #             scores+=ave
    #             totals+=nums
    #
    #         print('*********precision:{}**********'.format(scores/totals))
    #     return scores/totals

    def disc_metric(self,labels,preds):
        correct=0.
        sum=0.
        for l,p in zip(labels,preds):
            if l==p:
                correct+=1.
            sum+=1.
        return correct/sum

    def get_save_path(self):
        return join(self.args.out_dir, 'train-in-fudge-way')

    def get_checkpoint(self, epoch_idx, f1_score):
        ckpt_name = "{}_{}_temperature{}_scope_{}_epoch_{}_f1_{}_{}.ckpt".format(self.args.tuning_name,
                                                                                 self.args.corpus_type,
                                                                                 self.args.temperature,
                                                                                 self.args.ranking_scope, epoch_idx,
                                                                                 str(f1_score),
                                                                                 str(self.args.template).replace(" ",
                                                                                                                 ""))
        return {'embedding': self.model.state_dict(),
                'ckpt_name': ckpt_name,
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))

        if self.args.use_lm_finetune:
            self.model.model.save_pretrained(str(join(path, ckpt_name))[:-5])
        print("Checkpoint {} saved.".format(ckpt_name))

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters(), 'lr': self.args.lr}]
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': 1e-5})

        optimizer = torch.optim.Adam(params, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        stop_count = 0
        best_result = 0.0
        for epoch_idx in range(self.args.epoch):

            tot_loss = 0
            count = 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader),total=len(self.train_loader),desc='epoch:{}'.format(epoch_idx)):
                self.model.train()
                x = batch[0].cuda().squeeze(1)
                musk = batch[1].long().cuda().squeeze(1)
                y = batch[2].long().cuda()

                loss = self.model(x, y, musk)

                tot_loss += loss.item()

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()

            print(f"epoch index is {epoch_idx}, and total loss is {tot_loss}")

            my_lr_scheduler.step()

            #             if epoch_idx > -1:
            #                 result = self.evaluate(epoch_idx, 'Test')
            #                 weight_avg =result["weighted avg"]
            #                 f1_score = weight_avg["f1-score"]

            #                 if f1_score > best_result:
            #                     best_ckpt = self.get_checkpoint(epoch_idx,best_result)
            #                     best_result =  f1_score
            #                     stop_count = 0
            #                     continue
            #                 else:
            #                     stop_count += 1
            #                     if stop_count>5:
            #                         self.save(best_ckpt)
            #                         break

            if epoch_idx >= -1:

                if self.args.tuning_name == "prompt_tuning" or self.args.tuning_name == "disc_tuning":
                    result = self.evaluate(epoch_idx, 'Test')
                    # weight_avg = result["weighted avg"]
                    # f1_score = round(weight_avg["f1-
                    best_ckpt = self.get_checkpoint(epoch_idx, round(result, 2))
                else:
                    best_ckpt = self.get_checkpoint(epoch_idx, round(tot_loss, 2))

                self.save(best_ckpt)
    # def train(self):
    #     best_dev, early_stop, has_adjusted = 0, 0, True
    #     best_ckpt = None
    #     params = [{'params': self.model.prompt_encoder.parameters(), 'lr': self.args.lr}]
    #     if self.args.use_lm_finetune:
    #         params.append({'params': self.model.model.parameters(), 'lr': 1e-5})
    #
    #     optimizer = torch.optim.Adam(params, weight_decay=self.args.weight_decay)
    #     my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)
    #
    #     stop_count = 0
    #     best_result = 0.0
    #     for epoch_idx in range(self.args.epoch):
    #
    #         tot_loss = 0
    #         count = 0
    #         for batch_idx, batch in tqdm(enumerate(self.train_loader),total=len(self.train_loader),desc='epoch:{}'.format(epoch_idx)):
    #             self.model.train()
    #             x = batch[0].cuda().squeeze(1)
    #             musk = batch[1].long().cuda().squeeze(1)
    #             y = batch[2].long().cuda()
    #
    #             loss = self.model(x,musk,y)
    #
    #             tot_loss += loss.item()
    #
    #             loss.backward()
    #             torch.cuda.empty_cache()
    #             optimizer.step()
    #             torch.cuda.empty_cache()
    #             optimizer.zero_grad()
    #
    #         print(f"epoch index is {epoch_idx}, and total loss is {tot_loss}")
    #
    #         my_lr_scheduler.step()
    #
    #         #             if epoch_idx > -1:
    #         #                 result = self.evaluate(epoch_idx, 'Test')
    #         #                 weight_avg =result["weighted avg"]
    #         #                 f1_score = weight_avg["f1-score"]
    #
    #         #                 if f1_score > best_result:
    #         #                     best_ckpt = self.get_checkpoint(epoch_idx,best_result)
    #         #                     best_result =  f1_score
    #         #                     stop_count = 0
    #         #                     continue
    #         #                 else:
    #         #                     stop_count += 1
    #         #                     if stop_count>5:
    #         #                         self.save(best_ckpt)
    #         #                         break
    #
    #         if epoch_idx >= -1:
    #
    #             if self.args.tuning_name == "prompt_tuning" or self.args.tuning_name == "disc_tuning":
    #                 result = self.evaluate(epoch_idx, 'Test')
    #                 # weight_avg = result["weighted avg"]
    #                 # f1_score = round(weight_avg["f1-
    #                 best_ckpt = self.get_checkpoint(epoch_idx, round(result, 2))
    #             else:
    #                 best_ckpt = self.get_checkpoint(epoch_idx, round(tot_loss, 2))
    #
    #             self.save(best_ckpt)

def main(relation_id=None):
    args = construct_generation_args()

    # train stage
    trainer = Trainer(args)
    trainer.train()

    ## generation stage


if __name__ == '__main__':
    main()
