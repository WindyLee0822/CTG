import itertools
import json
import linecache
import os
import pickle
import re
import socket
import string
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List
from transformers import AutoModel,AutoTokenizer,AutoModelForSequenceClassification

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
# import jsonlines

class Evaluation():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'cardiffnlp/twitter-roberta-base-sentiment-latest')

    def eval(self,text,target='positive'):
        inputs = self.tokenizer(text,return_tensors='pt',padding=True)
        # inputs = {k:v.to('cuda') for k,v in inputs.items()}
        output = self.model(**inputs)
        predicted_class_id = output.logits.argmax(-1)
        labels = [self.model.config.id2label[i] for i in predicted_class_id.tolist()]
        nums = [1 for i in labels if i.lower()==target]
        return sum(nums)/len(labels)

    def score(self,text,target='POSITIVE'):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        output = self.model(**inputs)
        #todo check whether has been softmax?
        pid = self.model.config.label2id['positive']
        nid = self.model.config.label2id['negative']
        logits = torch.softmax(output.logits,dim=-1)
        scores = logits[:,pid] - logits[:,nid]
        return scores


class TextDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
    ):
        super().__init__()

        self.src_file = Path(data_dir)
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_length
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")  # +self.tokenizer.bos_token
        source_line = source_line.replace("xxx", '')

        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt",
                                               truncation=True, padding="max_length")
        return [res_input["input_ids"], res_input["attention_mask"]]

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]


class ToxicPrompt(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
            n_obs=None,
            prefix="",
    ):
        super().__init__()

        self.src_file = data_dir

        self.prompts = []
        with open(str(self.src_file), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                prompt = item["prompt"]["text"]
                self.prompts.append(prompt)

        self.tokenizer = tokenizer
        self.max_lens = max_length
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        index = index  # linecache starts at 1
        source_line = self.prompts[index].rstrip("\n")
        source_line = source_line.replace("xxx", '')

        res = self.tokenizer.encode_plus(source_line, max_length=self.max_lens, return_tensors="pt", truncation=True,
                                         padding="max_length")

        return (res["input_ids"], res["attention_mask"])

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]


class SentimentPrompt(Dataset):

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
            prompt_type="negative",
            n_obs=None,
            prefix="",
    ):
        super().__init__()

        self.src_file = data_dir + "/" + str(prompt_type) + '_prompts.jsonl'

        self.prompts = []
        with open(str(self.src_file), "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                prompt = item["prompt"]["text"]
                self.prompts.append(prompt)

        self.tokenizer = tokenizer
        self.max_lens = max_length
        self.tokenizer.padding_side = "left"

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        index = index  # linecache starts at 1
        source_line = self.prompts[index].rstrip("\n")
        source_line = source_line.replace("xxx", '')

        assert source_line, f"empty source line for index {index}"

        res = self.tokenizer.encode_plus(source_line, max_length=self.max_lens, return_tensors="pt", truncation=True,
                                         padding="max_length")

        return (res["input_ids"], res["attention_mask"])

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]


class DetoxicDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
            type_path="train",
            n_obs=None,
            src_lang=None,
            tgt_lang=None,
            prefix="",
            label_token={}
    ):
        super().__init__()

        self.src_file = Path(data_dir).joinpath(type_path + ".src")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")

        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_length
        self.max_target_length = max_length

        self.label_token = label_token

        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"

        self.tokenizer = tokenizer
        self.prefix = prefix

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.padding_side = "left"

    def token_wrapper(args, token):
        if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
            return 'Ġ' + token
        else:
            return token

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
            "\n")  # +self.tokenizer.bos_token
        source_line = source_line.replace("xxx", '')

        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        tgt_line = str(tgt_line)
        if "1" in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['positive']))
        else:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['negative']))

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt",
                                               truncation=True, padding="max_length")

        return [res_input["input_ids"], res_input["attention_mask"], tgt_line]

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]


class GPT2Label(Dataset):
    def __init__(self,tokenizer,data_dir,max_length):
        self.device = 'cuda'
        self.model = AutoModel.from_pretrained('gpt2').to(self.device)
        self.classify = Evaluation()
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_source_length = max_length
        self.data = self.process_data()

    def get_q(self,input_ids):
        input_ids = input_ids.to(self.device)
        output = self.model(input_ids = input_ids, attention_mask = (input_ids != self.pad_token_id))
        logits = torch.matmul(output.last_hidden_state[:,-1,:],self.model.wte.weight.T)
        s_logits = logits.sort(-1,descending=True)
        ids = s_logits.indices[...,:32]
        possibility = s_logits.values[...,:32]
        batch_input = torch.cat([input_ids.repeat(32,1),ids.T],dim=-1)
        batch_text = [self.tokenizer.decode(i) for i in batch_input]
        batch_score = self.classify.score(batch_text)
        q_value = batch_score * torch.softmax(possibility,-1).to('cpu')
        q_value = q_value.sum()
        return q_value

    def process_data(self):
        with open(Path(self.data_dir).joinpath("train.src")) as f:
            all_text = f.readlines()

        data=[]
        for raw_text in tqdm(all_text[:8],desc='Calculating Q-values:'):
            # text_list = raw_text.strip().split()
            encoding = self.tokenizer.encode_plus(raw_text,return_tensors="pt")
            input_ids,attn_mask = encoding['input_ids'],encoding['attention_mask']
            q_list = []
            q_len = min(input_ids.shape[-1],20)
            pad_len = 20
            for ilen in range(q_len):
                assert 0 not in attn_mask
                q_value = self.get_q(input_ids[:,:ilen+1])
                q_list.append(q_value.item())
            assert len(q_list)== q_len
            if input_ids.shape[-1]<pad_len:
                input_ids = torch.cat([input_ids,torch.tensor([[self.pad_token_id]*(pad_len-len(q_list))])],dim=-1)
                attn_mask = torch.cat([attn_mask,torch.tensor([[0]*(pad_len-len(q_list))])],dim=-1)
                q_list +=  [0.]*(pad_len-len(q_list))
            else:
                input_ids = input_ids[...,:pad_len]
                attn_mask = attn_mask[...,:pad_len]

            data.append([input_ids[0,:pad_len+1].tolist(),attn_mask[0,:pad_len+1].tolist(),q_list[:pad_len+1]])
        with open('processed_data.json','w') as f:
            json.dump(data,f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x,mask,y = self.data[index]
        return torch.tensor(x),torch.tensor(mask),torch.tensor(y)

class Classification_Dataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
            type_path="train",
            n_obs=None,
            src_lang=None,
            tgt_lang=None,
            prefix="",
            label_token={}
    ):
        super().__init__()

        # self.src_file = Path(data_dir).joinpath(type_path + ".src")
        # self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")
        # transfer to fudge form
        self.data_dir = data_dir
        if type(data_dir) != list:
            with open(Path(data_dir).joinpath(type_path + ".src")) as f:
                all_text = f.readlines()
            with open(Path(data_dir).joinpath(type_path + ".tgt")) as f:
                labels = f.readlines()
            assert len(all_text)==len(labels)
            self.dataset=[]
            for raw_text,label in zip(all_text,labels):
                text_list=raw_text.strip().split()
                if len(text_list)>10:#todo  it's 10 for quark
                    for ilen in range(10,len(text_list)):
                        text = ' '.join(text_list[:ilen+1])
                        self.dataset.append((text,label))
                else:
                    self.dataset.append((text,label))
                # self.dataset.append((raw_text.strip(),label))
        else:
            self.dataset = [(text,'') for text in data_dir]

        # self.src_lens = self.get_char_lens(self.src_file)
        self.src_lens = self.get_char_lens([x[0] for x in self.dataset])
        self.max_source_length = max_length
        self.max_target_length = max_length

        self.label_token = label_token

        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"

        self.tokenizer = tokenizer
        self.prefix = prefix

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.padding_side = "left"

    def token_wrapper(args, token):
        if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
            return 'Ġ' + token
        else:
            return token

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        if type(self.data_dir) == list:
            return [torch.tensor(self.dataset[index][0]), torch.tensor(self.dataset[index][0])!=self.tokenizer.pad_token_id, 0]
        # index = index + 1  # linecache starts at 1
        # source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
        #     "\n")  # +self.tokenizer.bos_token
        #
        # tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        source_line,tgt_line = self.dataset[index]

        if "positive" in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['positive']))
        elif "negative" in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['negative']))
        else:
            raise ValueError

        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"

        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt",
                                               truncation=True, padding="max_length")

        return [res_input["input_ids"], res_input["attention_mask"], tgt_line]

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in data_file]
        # return [len(x) for x in Path(data_file).open().readlines()]

class Classification_Dataset_double_sent(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
            type_path="train",
            n_obs=None,
            src_lang=None,
            tgt_lang=None,
            prefix="",
            label_token={}
    ):
        super().__init__()

        # self.src_file = Path(data_dir).joinpath(type_path + ".src")
        # self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")
        # transfer to fudge form
        self.data_dir = data_dir
        if type(data_dir) != list:
            with open(Path(data_dir).joinpath(type_path + ".src")) as f:
                all_text = f.readlines()
            with open(Path(data_dir).joinpath(type_path + ".tgt")) as f:
                labels = f.readlines()
            assert len(all_text)==len(labels)
            self.dataset=[]
            for raw_text,label in zip(all_text,labels):
                text_list=raw_text.strip().split()
                # if len(text_list)>10:#todo  it's 10 for quark
                #     for ilen in range(10,len(text_list)):
                #         text = ' '.join(text_list[:ilen+1])
                #         self.dataset.append((text,label))
                # else:
                #     self.dataset.append((text,label))
                self.dataset.append((raw_text.strip(),label))
        else:
            self.dataset = [(text,'') for text in data_dir]

        # self.src_lens = self.get_char_lens(self.src_file)
        self.src_lens = self.get_char_lens([x[0] for x in self.dataset])
        self.max_source_length = max_length
        self.max_target_length = max_length

        self.label_token = label_token

        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"

        self.tokenizer = tokenizer
        self.prefix = prefix

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.padding_side = "left"

    def token_wrapper(args, token):
        if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
            return 'Ġ' + token
        else:
            return token

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        if type(self.data_dir) == list:
            return [torch.tensor(self.dataset[index][0]), torch.tensor(self.dataset[index][0])!=self.tokenizer.pad_token_id, 0]
        # index = index + 1  # linecache starts at 1
        # source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
        #     "\n")  # +self.tokenizer.bos_token
        #
        # tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        source_line,tgt_line = self.dataset[index]

        if "positive" in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['positive']))
        elif "negative" in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['negative']))
        else:
            raise ValueError

        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"

        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt",
                                               truncation=True, padding="max_length")

        return [res_input["input_ids"], res_input["attention_mask"], tgt_line]

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in data_file]
        # return [len(x) for x in Path(data_file).open().readlines()]


class Classification_Dataset_label_num_3(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
            type_path="train",
            n_obs=None,
            src_lang=None,
            tgt_lang=None,
            prefix="",
            label_token={}
    ):
        super().__init__()

        self.src_file = Path(data_dir).joinpath(type_path + ".src")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")
        # transfer to fudge form

        self.data_dir = data_dir
        if type(data_dir) != list:
            dataset = json.load(open(self.data_dir))
            self.dataset=[]
            for label,raw_text in dataset:
                if 'positive' in label:
                    label = 'positive'
                elif 'negative' in label:
                    label = 'negative'
                else:
                    assert label == 'neutral'
                self.dataset.append((raw_text,label))
        else:
            self.dataset = [(text,'') for text in data_dir]

        # self.src_lens = self.get_char_lens(self.src_file)
        self.src_lens = self.get_char_lens([x[0] for x in self.dataset])
        self.max_source_length = max_length
        self.max_target_length = max_length

        self.label_token = label_token

        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"

        self.tokenizer = tokenizer
        self.prefix = prefix

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.padding_side = "left"

    def token_wrapper(args, token):
        if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
            return 'Ġ' + token
        else:
            return token

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        if type(self.data_dir) == list:
            return [torch.tensor(self.dataset[index][0]), torch.tensor(self.dataset[index][0])!=self.tokenizer.pad_token_id, 0]
        # index = index + 1  # linecache starts at 1
        # source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
        #     "\n")  # +self.tokenizer.bos_token
        #
        # tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        source_line,tgt_line = self.dataset[index]

        if "positive" in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['positive']))
        elif 'negative' in tgt_line:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['negative']))
        else:
            tgt_line = torch.tensor(self.tokenizer.encode(self.label_token['neutral']))

        assert source_line, f"empty source line for index {index}"
        # assert tgt_line, f"empty tgt line for index {index}"

        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt",
                                               truncation=True, padding="max_length")

        return [res_input["input_ids"], res_input["attention_mask"], tgt_line]

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in data_file]
        # return [len(x) for x in Path(data_file).open().readlines()]


class Sentiment_Suffix(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
            task_type="positive",
            n_obs=None,
            src_lang=None,
            tgt_lang=None,
            prefix="",
            label_token={}
    ):
        super().__init__()

        self.src_file = data_dir

        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_length

        self.label_token = label_token
        self.task_type = task_type

        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"

        self.tokenizer = tokenizer
        self.prefix = prefix

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.padding_side = "left"

    def token_wrapper(args, token):
        if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
            return 'Ġ' + token
        else:
            return token

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index):
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
            "\n")  # +self.tokenizer.bos_token

        tgt_line = torch.tensor(self.tokenizer.encode(self.label_token[self.task_type]))

        if len(source_line) < 2:
            source_line = "Hello world! Today is nice!"

        res_input = self.tokenizer.encode_plus(source_line, max_length=self.max_source_length, return_tensors="pt",
                                               truncation=True, padding="max_length")

        return [res_input["input_ids"], res_input["attention_mask"], tgt_line]

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]
