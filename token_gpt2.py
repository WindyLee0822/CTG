import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelWithLMHead, AutoTokenizer

import re
import datetime

from transformers import AutoTokenizer

import torch.nn.functional as F
from utils.utils import logits_to_entropy
SMALL_CONST = 1e-10
BIG_CONST = -1e15

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertTokenizer,
    GPT2Tokenizer)

from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM
from Sentiment.main_disc import Classifier,Scorer

def create_model(args):
    if args.model_name_or_path:
        # config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path
        )
    else:
        print("Model path is not set!!!")

    return model

def _create_model(model_path):
    if model_path:
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        print("Model path is not set!!!")

    return model

def get_embedding_layer(args, model):
    embeddings = model.base_model.get_input_embeddings()

    return embeddings

class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, args):
        super().__init__()
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(args.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(args.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(args.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class Distill_Tuning(torch.nn.Module):

    def __init__(self, args, template, label_token=None):
        super().__init__()
        self.args = args
        self.target_mode = args.target_mode

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # model setting
        self.model = create_model(self.args)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.args.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # get model's embeddings
        self.embeddings = self.model.get_input_embeddings()

        # label information
        self.label_token = label_token
        self.label_token_ids = {}

        for k, v in self.label_token.items():
            print(k, v, self.tokenizer.encode(v))
            self.label_token_ids[k] = self.tokenizer.encode(v)

        self.template = template
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim

        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)

        self.spell_length = sum(self.template)
        self.prompt_encoder_pos = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
        self.prompt_encoder_pos = self.prompt_encoder_pos.to(self.args.device)
        # self.prompt_encoder_neg = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
        # self.prompt_encoder_neg = self.prompt_encoder_neg.to(self.args.device)
        self.fc_loss = CrossEntropyLoss(reduction='none')
        # self.classifier = Classifier(param)
        self.scorer = Scorer(param)
        self.kl_dropout = nn.Dropout(p=0.5)
        ### load discriminator
        # if self.args.disc_embedding_checkpoint != None:
        #     self.disc_model = _create_model(self.args.disc_embedding_checkpoint[:-5]).to(self.args.device)
        #     self.spell_length_disc = sum(self.args.template_disc)
        #     self.disc_embedding = self.disc_model.get_input_embeddings()
        #     self.prompt_encoder_disc = PromptEncoder(self.args.template_disc, self.disc_embedding.embedding_dim,
        #                                              self.tokenizer, args)
        #     self.prompt_encoder_disc = self.prompt_encoder_disc.to(self.args.device)
        #     self.prompt_encoder_disc.load_state_dict(self.load_prompt(self.args.disc_embedding_checkpoint))
        # else:
        #     self.disc_model = self.model
        #     self.prompt_encoder_disc = self.prompt_encoder

    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        raw_embeds = self.disc_embedding(queries_for_embedding)

        replace_embeds = self.prompt_encoder_disc()

        replace_embeds = replace_embeds.unsqueeze(0).expand(bz, -1, -1)

        raw_embeds[:, -self.prompt_encoder_disc.spell_length:, :] = replace_embeds

        return raw_embeds

    def get_query_head(self, x_h, prompt_tokens, x_t=None):

        prompt_tensor_head = torch.tensor(prompt_tokens * (self.spell_length)).to(self.args.device)

        trans_inputs = []

        index_musk = (x_h == self.tokenizer.pad_token_id).type(torch.uint8)  # only calculte the token which is not eos

        valid_number_length = torch.sum(index_musk, 1)

        for index, seq in zip(valid_number_length, x_h):
            trans_inputs.append(torch.cat([prompt_tensor_head, seq]))
            # if index == x_h.shape[1]:
            #     trans_inputs.append(torch.cat([prompt_tensor_head, seq]))
            # else:
            #     trans_inputs.append(torch.cat([seq[:index], prompt_tensor_head, seq[index:]]))

        res = torch.stack(trans_inputs, dim=0)
        if x_t != None:
            # x_t = x_t.unsqueeze(1)
            return torch.cat([res, x_t], dim=1)
        else:
            return res

    def embed_input_head(self, queries,mode='positive'):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()

        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        try:
            blocked_indices = (queries == self.pseudo_token_id).type(torch.uint8).nonzero().reshape(
                (bz, self.spell_length, 2))[:, :, 1]  # bz
        except:
            print(bz)
            print(queries.shape)
            print(queries[0])
            print(queries[-1])
            print((queries == self.pseudo_token_id).type(torch.uint8).nonzero().shape)
            raise ValueError

        if mode=='positive':
            prompt_encoder = self.prompt_encoder_pos
        elif mode=='negtive':
            prompt_encoder = self.prompt_encoder_neg
        else:
            raise ValueError

        replace_embeds = prompt_encoder()
        for bidx in range(bz):
            for i in range(prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds


    def top_k_top_p_filtering(self,logits,top_k=0,top_p=1.0,filter_value=BIG_CONST,min_tokens_to_keep=1,):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits

    # def get_q_value(self,text_ids,logits,topk):
    #     past_key_values = self.classifier.get_past_key_values(text_ids)
    #     sort_logits = logits.sort(dim=-1,descending=True)
    #     indices = sort_logits.indices[:,:topk,None]
    #     p = torch.softmax(sort_logits.values[:,:topk],dim=-1)
    #     # cat_ids = torch.cat([text_ids[:,None,:].repeat(1,topk,1),indices[None,...].repeat(text_ids.shape[0],1,1)])
    #     reward = self.classifier.get_q_value(text_ids,indices,past_key_values)
    #     #todo whether maintain the interference
    #     # interfere = torch.normal(0,0.08,size=reward.shape)
    #     # interfere = interfere.masked_fill(reward!=0,0)
    #     # reward += interfere
    #     # q_value = p * reward
    #     # return q_value.sum(-1)
    #     return reward.view(-1)

    def get_q_value(self,text_ids):
        q=self.scorer.score(text_ids)
        return q if self.target_mode=='positive' else 1-q

    def sample(self, prompts_ids, max_length, mode='positive', gen=False):
        cur_len = prompts_ids.shape[1]
        logits = []
        output_ids = prompts_ids
        q_storage = torch.zeros_like(output_ids)[:, :-1]
        output_mask = torch.zeros_like(output_ids)
        return_dict = {}
        eos_flag = torch.ones([prompts_ids.shape[0]]).type(torch.uint8).to(self.args.device)

        if mode != None:
            prompt_tokens = [self.pseudo_token_id]
            queries = self.get_query_head(prompts_ids, prompt_tokens)
            inputs_embeds = self.embed_input_head(queries, mode)

            # attention_mask = torch.cat([prompts_ids != self.tokenizer.pad_token_id, torch.ones(
            #     [prompts_ids.shape[0], self.spell_length + max_length - prompts_ids.shape[1]]).long().to(
            #     self.args.device)], dim=1)
            attention_mask = torch.cat([torch.ones(prompts_ids.shape[0], self.spell_length, device=self.args.device),
                                        prompts_ids != self.tokenizer.pad_token_id,
                                        torch.ones(prompts_ids.shape[0], max_length - prompts_ids.shape[1] + 1,
                                                   device=self.args.device).long().to(self.args.device)], dim=-1)
        else:
            inputs_embeds = self.embeddings(prompts_ids)
            attention_mask = torch.cat([prompts_ids != self.tokenizer.pad_token_id, torch.ones(
                [prompts_ids.shape[0], max_length - prompts_ids.shape[1] + 1]).long().to(self.args.device)], dim=1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill_(attention_mask == 0, 0)

        if not gen:
            q_value = self.get_q_value(output_ids)
            q_storage = torch.cat([q_storage, q_value[..., None]], dim=-1)

        # start = datetime.datetime.now()
        # test generation time
        first_round = 1
        while cur_len <= max_length:
            outputs = self.model(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask[:, :inputs_embeds.shape[1]],
                                 position_ids=position_ids[:, :inputs_embeds.shape[1]],
                                 return_dict=True)

            if first_round:
                if mode == None:
                    last_non_masked_idx = torch.sum(prompts_ids != self.tokenizer.pad_token_id, dim=1) - 1
                else:
                    last_non_masked_idx = torch.sum(prompts_ids != self.tokenizer.pad_token_id,
                                                    dim=1) - 1 + self.spell_length
                next_token_logits = outputs.logits[range(prompts_ids.shape[0]), last_non_masked_idx, :]
                first_round = 0
            else:
                next_token_logits = outputs.logits[:, -1, :]

            # if gen == False:
            next_token_logits_ = self.top_k_top_p_filtering(next_token_logits, top_k=self.args.ranking_scope, top_p=1.0,
                                                            filter_value=BIG_CONST)
            # else:
            #     next_token_logits_ = self.top_k_top_p_filtering(next_token_logits, top_k=10,
            #                                                     top_p=1.0, filter_value=BIG_CONST)

            next_token_logits_prob = torch.softmax(next_token_logits_, dim=1)

            next_token_logits_prob[:, self.tokenizer.eos_token_id] = 0
            next_token_logits_prob[:, self.pseudo_token_id] = 0
            # if gen==False:
            next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
            # else:
            #     next_tokens = torch.argmax(next_token_logits, dim=-1)

            eos_flag = eos_flag.mul((next_tokens != self.tokenizer.eos_token_id).type(
                torch.uint8))  # if flag = 0, it means the generation is over
            next_tokens = next_tokens.mul(eos_flag)
            next_tokens[next_tokens == 0] = self.tokenizer.eos_token_id
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)
            # output_mask = torch.cat([output_mask,torch.ones_like(next_tokens.unsqueeze(1))],dim=1)
            output_mask = torch.cat([output_mask, (next_tokens.unsqueeze(1) != self.tokenizer.eos_token_id)], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, self.embeddings(next_tokens).unsqueeze(1)], dim=1)

            cur_len = cur_len + 1

            if not gen:
                q_value = self.get_q_value(output_ids)
                q_storage = torch.cat([q_storage, q_value[..., None]], dim=-1)

        #         end = datetime.datetime.now()
        #         print("runing time is:",end-start)
        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for output in output_ids]
        prompt_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for output in prompts_ids]

        if not gen:
            assert output_ids.shape[-1] == q_storage.shape[-1]
            reward = q_storage[:, 1:] - q_storage[:, :-1]
        else:
            reward = None
        # reward = q_storage[:, -1][..., None].repeat(1, q_storage.shape[-1] - 1)
        return_dict = {
            'text': response_text,
            'prompt': prompt_text,
            'input_ids': output_ids,
            'output_mask': output_mask,
            'q_values': reward
        }
        return return_dict


    def forward(self, x_hs, x_ts, att_mask):
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(x_hs, prompt_tokens)
        # construct label ids
        attention_mask = torch.cat(
            [att_mask, torch.ones([att_mask.shape[0], self.prompt_encoder.spell_length]).long().to(self.args.device)],
            dim=1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        labels = torch.clone(queries)

        labels.masked_fill_(attention_mask == 0, -100)
        labels.masked_fill_(queries == self.pseudo_token_id, -100)

        # get embedded input
        inputs_embeds = self.embed_input_head(queries)

        output = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=None)

        output_logits = output.logits
        # ce_loss =  self.contrast_crossEntry_loss(torch.softmax(output_logits, dim = -1), labels, sentence_labels = x_ts)

        _queries = queries.view(queries.size(0) * queries.size(1))
        _output_logits = output_logits.view(output_logits.size(0) * output_logits.size(1), -1)
        disc_logits = _output_logits.index_select(0, torch.nonzero(_queries != self.pseudo_token_id).squeeze(1)).view(
            output_logits.shape[0], -1, output_logits.shape[2])

        logits_candidate = self.get_candidate_logits(x_hs, att_mask)
        logits_candidate = self.top_k_top_p_filtering(
            logits_candidate.view(logits_candidate.shape[0] * logits_candidate.shape[1], -1),
            top_k=self.args.ranking_scope, top_p=self.args.top_p, filter_value=BIG_CONST).view(x_hs.shape[0],
                                                                                               x_hs.shape[1], -1)

        reank_output = self.get_ranked_logtis(x_hs, logits_candidate.detach().clone(), desired_att=None)

        reank_output = (logits_candidate > BIG_CONST + 10).mul(reank_output)

        kl_loss = self.KL_loss(torch.softmax(disc_logits, dim=-1), reank_output, att_mask)

        loss = kl_loss

        return loss

    def forward_pass(self, x_hs, att_mask, out_mask, reward_mask=None, mode='positive',gen=False):
        # construct query ids
        if mode!=None:
            prompt_tokens = [self.pseudo_token_id]
            queries = self.get_query_head(x_hs, prompt_tokens)
            inputs_embeds = self.embed_input_head(queries,mode)
            attention_mask = torch.cat(
                [torch.ones([att_mask.shape[0], self.spell_length]).long().to(self.args.device),att_mask],
                dim=1)

        else:
            queries = x_hs
            inputs_embeds = self.embeddings(queries)
            attention_mask = att_mask

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill_(attention_mask == 0, 0)

        # construct label ids
        labels = torch.clone(queries)
        # labels.masked_fill_(attention_mask == 0, -100)
        # labels.masked_fill_(queries == self.pseudo_token_id, -100)
        if mode != None:
            supplement = torch.zeros((out_mask.shape[0],self.spell_length),device=self.args.device)
            out_mask = torch.cat([supplement,out_mask],-1)
            # attention_mask = torch.cat([supplement,attention_mask],-1)
            reward_mask = torch.cat([supplement,reward_mask],-1)
            # punish_mask = torch.cat([supplement, punish_mask], -1)

        total_mask = attention_mask * out_mask
        labels = labels.masked_fill_(total_mask == 0 ,-100)
        labels = labels[:,1:]
        # reward_labels.masked_fill_(reward_mask==0, -100)
        # punish_labels.masked_fill_(punish_mask == 0, -100)

        outputs = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=None)

        logits = outputs.logits[:,:-1,:]

        if mode==None:
            output_logits = logits * (labels.unsqueeze(-1) != -100)
            return {'logits':output_logits}

        loss = self.fc_loss(logits.reshape(-1,logits.shape[-1]), labels.reshape(-1)) #- self.fc_loss(logits.reshape(-1,logits.shape[-1]),punish_labels.reshape(-1)) * 0.8
        loss = loss.reshape(x_hs.shape[0],-1)
        lm_loss = (loss * reward_mask).sum()

        log_prob = F.log_softmax(logits, dim=-1)
        labels_select = torch.where(labels==-100,0,labels)
        output_logprob = torch.gather(log_prob, -1, labels_select[...,None]).squeeze(2)
        output_logprob = output_logprob.masked_fill_(labels==-100,0)

        output_logits = logits * (labels.unsqueeze(-1)!=-100)
        # kl_mask = (labels!=-100) ^  ((reward_mask>0) & (reward_mask<0.01))
        kl_mask = (reward_mask!=0)

        # output_logprob = torch.gather(log_prob, 2, labels_select[...,None]).squeeze(2)
        output_entropy = logits_to_entropy(logits)
        # lm_loss = -1. * output_logprob

        output_dic={'logits':output_logits,'loss':lm_loss,'logprob':output_logprob,
                    'entropy':output_entropy,'kl_mask':kl_mask}

        return output_dic




