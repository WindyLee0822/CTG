import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import re
import datetime

from transformers import AutoTokenizer

from Sentiment.models import get_embedding_layer, create_model, _create_model
from Sentiment.prompt_encoder import PromptEncoder
# from models import get_embedding_layer, create_model, _create_model
# from prompt_encoder import PromptEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALL_CONST = 1e-10
BIG_CONST = -1e15


class PTuneForLAMA(torch.nn.Module):
    def __init__(self, args, template, label_token=None,label_num=2):
        super().__init__()
        self.args = args
        self.label_num = label_num
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # model setting
        self.model = create_model(self.args)
        # self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.args.device)
        # for param in self.model.parameters():
        #     param.requires_grad = self.args.use_lm_finetune

        self.template = template

        # get model's embeddings
        self.embeddings = self.model.get_input_embeddings()

        self.label_token = label_token
        self.label_map = {}
        self.label_token_ids = {}

        for k, v in self.label_token.items():
            print(k, v, self.tokenizer.convert_tokens_to_ids(v))
            self.label_map[self.tokenizer.convert_tokens_to_ids(v)] = k

            self.label_token_ids[k] = self.tokenizer.convert_tokens_to_ids(v)

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)

        # if self.args.disc_embedding_checkpoint == None:
        self.spell_length_disc = sum(self.template)
        self.prompt_encoder_disc = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
        self.prompt_encoder_disc = self.prompt_encoder_disc.to(args.device)
        self.prompt_encoder = self.prompt_encoder_disc
        self.disc_embedding = self.embeddings

        self.fc_loss = CrossEntropyLoss()

    def load_prompt(self, embedding_checkpoint):
        checkpoint = torch.load(embedding_checkpoint)
        prompt_embedding = checkpoint['embedding']
        return prompt_embedding

    def generate(self, prompts_ids, max_length, desired_att=None, beta=0.5):
        """
        generation forward based on given prompt tokens,
        Args:
            prompt_ids: the  prompt tokens
            max_length: the max len of the generation
        Returns:
            generated_texts:[generated tokens]
        """
        cur_len = prompts_ids.shape[1]
        logits = []
        output_ids = prompts_ids
        return_dict = {}
        eos_flag = torch.ones([prompts_ids.shape[0]]).type(torch.uint8).to(self.args.device)

        # start = datetime.datetime.now()
        past = None
        while cur_len <= max_length:
            past_k_v = past
            future_logits, past = self.generate_soft_tokens(output_ids, past_k_v)
            next_token_logits = future_logits.clone().detach().squeeze(1)
            perturb_logits = self.feedback_from_discriminator(output_ids, future_logits.unsqueeze(1), desired_att)

            next_token_logits_prob = torch.softmax(next_token_logits, dim=1)

            perturb_logits_prob = torch.softmax(perturb_logits, dim=1)

            next_token_logits_prob = perturb_logits_prob.mul(next_token_logits_prob)

            next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
            ## avoid eos token appeals continuely
            eos_flag = eos_flag.mul((next_tokens != self.tokenizer.eos_token_id).type(
                torch.uint8))  # if flag = 0, it means the generated is over
            next_tokens = next_tokens.mul(eos_flag)
            next_tokens[next_tokens == 0] = self.tokenizer.eos_token_id
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)
            print("cur_len is:", cur_len)
            cur_len = cur_len + 1

        # end = datetime.datetime.now()
        # print("runing time is:",end-start)

        return_dict = {"generated_tokens": output_ids}
        return return_dict

    def generate_soft_tokens(self, generated_tokens, past_key_values=None):

        if past_key_values != None:
            last_embeds = self.embeddings(generated_tokens[:, -1]).unsqueeze(1)  # get its embeddings
            # print("last_embeds：", last_embeds.shape)
            with torch.no_grad():
                outputs = self.model(inputs_embeds=last_embeds,
                                     past_key_values=past_key_values,
                                     return_dict=True)

        else:
            attention_mask = (generated_tokens != self.tokenizer.eos_token_id).type(torch.uint8)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill_(attention_mask == 0, 0)
            last_embeds = self.embeddings(generated_tokens)  # get its embeddings

            with torch.no_grad():
                outputs = self.model(inputs_embeds=last_embeds,
                                     past_key_values=past_key_values,
                                     attention_mask=attention_mask,
                                     position_ids=position_ids,
                                     return_dict=True)

        next_token_logits = outputs.logits[:, -1, :]

        next_token_logits = self.top_k_top_p_filtering(next_token_logits.squeeze(1), top_k=self.args.ranking_scope,
                                                       top_p=self.args.top_p, filter_value=BIG_CONST)

        return next_token_logits, outputs.past_key_values

    def discriminator_predict(self, input_ids):

        input_ids_left_pad, length_generated_tokens = self.pad_left_to_right(input_ids, self.tokenizer.eos_token_id)

        musk = (input_ids_left_pad != self.tokenizer.eos_token_id).type(torch.uint8)

        pred_ids = self.predict(input_ids_left_pad, musk)

        return pred_ids

    def scores_predict(self, input_ids):

        input_ids_left_pad, length_generated_tokens = self.pad_left_to_right(input_ids, self.tokenizer.eos_token_id)

        musk = (input_ids_left_pad != self.tokenizer.eos_token_id).type(torch.uint8)

        pred_scores = self.predict_scores(input_ids_left_pad, musk)

        return pred_scores

    def top_k_top_p_filtering(self,
                              logits,
                              top_k=0,
                              top_p=1.0,
                              filter_value=BIG_CONST,
                              min_tokens_to_keep=1,
                              ):
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

    def pad_left_to_right(self, inputs, pad_id):
        trans_inputs = torch.empty_like(inputs)

        input_remove_prompt = inputs[:, self.prompt_pad_length:]

        index_musk = (input_remove_prompt != pad_id).type(torch.uint8)  # only calculte the token which is not eos

        length_of_generated_text = torch.sum(index_musk, 1)

        valid_number_length = length_of_generated_text + self.prompt_pad_length

        count = 0
        for index, seq in zip(valid_number_length, inputs):

            if index == 0 or index == inputs.shape[1]:
                trans_inputs[count] = seq
            else:
                trans_inputs[count][-index:] = seq[:index]
                trans_inputs[count][:inputs.shape[1] - index] = seq[index:]
            count += 1

        return trans_inputs, length_of_generated_text

    def feedback_from_discriminator(self, input_ids, logits_seq, desired_att):
        logits_seq = logits_seq.squeeze(1)
        top_logits, top_indices = logits_seq.topk(self.args.ranking_scope, dim=1)  # batch x topk

        scores = []
        candidates = []
        for logit_id, ids in zip(top_indices, input_ids):
            data = ids.expand(self.args.ranking_scope, -1)
            new_input_candidates = torch.cat([data, logit_id.unsqueeze(1)], dim=1)  # batch x topk x seq+1
            candidates.append(new_input_candidates)
        candidates = torch.cat(candidates, dim=0)

        musk = (candidates != self.tokenizer.eos_token_id).type(torch.uint8)
        pred_scores = self._predict_scores(candidates, musk)
        pred_scores = pred_scores.reshape(input_ids.shape[0], -1)

        logits_seq.scatter_(-1, top_indices, pred_scores)

        indices_to_remove = logits_seq < torch.topk(logits_seq, 3)[0][..., -1, None]
        logits_seq[indices_to_remove] = BIG_CONST

        return logits_seq

    def gradient_feedback_from_discriminator(self, past, logits_seq, desired_att, lr):

        musk_value = torch.empty_like(logits_seq).fill_(0.0).long().to(self.args.device)
        indices_musk = (logits_seq == BIG_CONST)
        musk_value[indices_musk] = BIG_CONST
        musk_value.requires_grad = False

        index = torch.nonzero(logits_seq != BIG_CONST)

        logits_seqs = torch.empty_like(logits_seq)

        logits_seqs.fill_(0.0)

        update_logit = logits_seqs
        update_logit.requires_grad = True

        optimizer = torch.optim.AdamW([{"params": update_logit}], lr=lr, amsgrad=True, weight_decay=0.1)

        num_backward_iters = self.args.iter_num

        for i in range(num_backward_iters):
            update_logit_ = update_logit.mul(~indices_musk)  ##topk, value is orginal
            logits = update_logit_ + musk_value

            logit_softmax = torch.softmax(logits, dim=-1)

            soft_tokens = torch.matmul(logit_softmax, self.discrimirator_embedding.weight)

            loss_discrimlator = self.loss_for_desiredAtt(past, soft_tokens, desired_att)

            loss = loss_discrimlator  # + 0.0*l1_loss

            print("the loss is:", loss)

            loss.backward()
            torch.cuda.empty_cache()
            optimizer.step()
            torch.cuda.empty_cache()
            optimizer.zero_grad()

        update_logit_ = update_logit.mul(~indices_musk)  ##topk, value is orginal
        logits = update_logit_ + musk_value

        indices_to_remove = logits < torch.topk(logits, self.args.top_k)[0][..., -1, None]
        logits[indices_to_remove] = BIG_CONST

        return logits.squeeze(1)

    def get_query(self, x_h, prompt_tokens, x_t=None):

        prompt_tensor = torch.tensor(prompt_tokens * (self.spell_length_disc)).to(x_h.device)
        prompt_tensor = prompt_tensor.expand(x_h.shape[0], -1)
        if x_t != None:
            x_t = x_t.unsqueeze(1)
            return torch.cat([x_h, prompt_tensor, x_t], dim=1)
        else:
            return torch.cat([x_h, prompt_tensor], dim=1)

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        raw_embeds = self.disc_embedding(queries_for_embedding)

        replace_embeds = self.prompt_encoder_disc()

        replace_embeds = replace_embeds.unsqueeze(0).expand(bz, -1, -1)

        raw_embeds[:, -self.prompt_encoder_disc.spell_length:, :] = replace_embeds

        return raw_embeds

    def _predict_scores(self, x_hs, att_mask, reward=False,specific_label_for_topic=0):
        bz = len(x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]

        queries = self.get_query(x_hs, prompt_tokens)
        # construct label ids
        attention_mask = torch.cat([att_mask, torch.ones([att_mask.shape[0], self.prompt_encoder_disc.spell_length]).long().to(
                                        self.args.device)],dim=1)
        # get embedded input

        # print(queries.shape)
        inputs_embeds = self.embed_input(queries)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        # print(position_ids.shape, inputs_embeds.shape, attention_mask.shape)

        with torch.no_grad():

            output = self.model(inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask,
                                     position_ids=position_ids,
                                     labels=None)


        logits = output.logits[:, -1, :].squeeze(1)
        # last_non_masked_idx = position_ids.argmax(-1)
        # logits = output.logits[range(position_ids.shape[0]), last_non_masked_idx, :]

        if self.label_num == 2:
            serial_list = [self.label_token_ids['positive'],
                           self.label_token_ids['negative']]
        elif self.label_num == 3:
            serial_list = [self.label_token_ids['Asian'],
                        self.label_token_ids['American'],self.label_token_ids['Mexico']]


        tri_mask = torch.ones_like(logits,dtype=torch.bool)
        for i in serial_list:
            tri_mask[:,i]=0
        tri_prob = torch.masked_fill(logits,tri_mask,-torch.inf)

        # if self.args.target_type == "negative":
        #     return binary_prob[:, 1]
        # else:
        #     return binary_prob[:, 0]
        if reward==False:
            # results = torch.where(binary_prob[:,0]>binary_prob[:,1],11274,14774)
            # return results.unsqueeze(-1).tolist()
            rds = tri_prob.argmax(dim=-1).unsqueeze(-1).tolist()
            return rds

        else:
            # binary_prob = torch.softmax(binary_prob,dim=-1)
            if self.label_num ==2:
                tri_prob = torch.softmax(tri_prob[:,serial_list],dim=-1)
                return tri_prob[:,0]
            elif self.label_num ==3:
                tri_prob = torch.softmax(tri_prob[:,serial_list],dim=-1)
                return tri_prob[:,specific_label_for_topic]

    def forward(self, x_hs, x_ts, att_mask):
        bz = len(x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]

        queries = self.get_query(x_hs, prompt_tokens)

        # construct label ids
        attention_mask = torch.cat([att_mask,torch.ones([att_mask.shape[0], self.prompt_encoder_disc.spell_length]).long().to(
                                        att_mask.device)], dim=1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        # get embedded input
        inputs_embeds = self.embed_input(queries)

        output = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=None)

        logits = output.logits[:, -1, :].squeeze(1)
        # last_non_masked_idx = position_ids.argmax(-1)
        # # print(position_ids[0],last_non_masked_idx[0])
        # # print(position_ids[1], last_non_masked_idx[1])
        # # raise ValueError
        # logits = output.logits[range(position_ids.shape[0]), last_non_masked_idx, :]

        loss = self.fc_loss(logits, x_ts.squeeze(1))

        return loss

    def get_past_key_values(self, x_hs, att_mask):
        bz = len(x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]

        queries = self.get_query(x_hs, prompt_tokens)

        # construct label ids
        attention_mask = torch.cat([att_mask,torch.ones([att_mask.shape[0], self.prompt_encoder_disc.spell_length]).long().to(
                                        att_mask.device)], dim=1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill_(attention_mask == 0, 0)

        # get embedded input
        inputs_embeds = self.embed_input(queries)
        output = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True)
        return output.past_key_values

    def forward_with_pkv(self,input_ids,attn_mask,pkv,mean=True):
        # construct label ids
        attention_mask = torch.cat([attn_mask,torch.ones([attn_mask.shape[0], self.prompt_encoder_disc.spell_length]).long().to(
                                        attn_mask.device)],dim=1)

        # no need to specify position ids. (will add past_key_values length automatically)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill_(attention_mask == 0, 0)
        position_ids = position_ids[:,-1]
        output = self.model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            past_key_values = pkv,position_ids=position_ids)

        logits = output.logits[:, -1, :].squeeze(1)
        # last_non_masked_idx = position_ids.argmax(-1)
        # logits = output.logits[range(position_ids.shape[0]), last_non_masked_idx, :]

        if self.label_num==3:
            serial_list = [self.label_token_ids['positive'],
                       self.label_token_ids['negative'],
                       self.label_token_ids['neutral']]
            tri_prob = torch.softmax(logits[:, serial_list], dim=-1)

            results = tri_prob.argmax(dim=-1)
            reward = torch.where(results == 0, 1, torch.where(results == 1, -1, 0))
        elif self.label_num==2:
            serial_list = [self.label_token_ids['positive'],
                           self.label_token_ids['negative']]
            if mean:
                bi_prob = torch.softmax(logits[:, serial_list], dim=-1)
                reward = bi_prob[:,0].mean()
            else:
                bi_prob = torch.softmax(logits[:, serial_list], dim=-1)
                reward = bi_prob[:,0]
            # results = bi_prob.argmax(dim=-1)
            # reward = torch.where(results == 0, 1, -1)

        return reward


# class PTuneForLAMA(torch.nn.Module):
#
#     def __init__(self, args, template, label_token=None):
#         super().__init__()
#         self.args = args
#
#         self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#
#         # model setting
#         self.model = create_model(self.args)
#         # self.model.resize_token_embeddings(len(self.tokenizer))
#         self.model = self.model.to(self.args.device)
#         # for param in self.model.parameters():
#         #     param.requires_grad = self.args.use_lm_finetune
#
#         self.template = template
#
#         # get model's embeddings
#         self.embeddings = self.model.get_input_embeddings()
#         # label import torch
#         # from torch.nn.utils.rnn import pad_sequence
#         # from os.path import join
#         # from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
#         #
#         # import re
#         # import datetime
#         #
#         # from transformers import AutoTokenizer
#         #
#         # from models import get_embedding_layer, create_model, _create_model
#         # from prompt_encoder import PromptEncoder
#         #
#         # import torch
#         # import torch.nn as nn
#         # import torch.nn.functional as F
#         #
#         # SMALL_CONST = 1e-10
#         # BIG_CONST = -1e15
#         #
#         #
#         #
#         # class PTuneForLAMA(torch.nn.Module):
#         #
#         #     def __init__(self, args, template, label_token = None):
#         #         super().__init__()
#         #         self.args = args
#         #
#         #         self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
#         #         self.tokenizer.pad_token = self.tokenizer.eos_token
#         #
#         #         # model setting
#         #         self.model = create_model(self.args)
#         #         # self.model.resize_token_embeddings(len(self.tokenizer))
#         #         self.model = self.model.to(self.args.device)
#         #         for param in self.model.parameters():
#         #             param.requires_grad = self.args.use_lm_finetune
#         #
#         #         self.template = template
#         #
#         #         # get model's embeddings
#         #         self.embeddings = self.model.get_input_embeddings()
#         #
#         #         # label information
#         #
#         #         self.label_token = label_token
#         #         self.label_map = {}
#         #         self.label_token_ids ={}
#         #
#         #         for k, v in self.label_token.items():
#         #             print(k,v,self.tokenizer.convert_tokens_to_ids(v))
#         #             self.label_map[self.tokenizer.convert_tokens_to_ids(v)] = k
#         #
#         #             self.label_token_ids[k] = self.tokenizer.convert_tokens_to_ids(v)
#         #
#         #         # load prompt encoder
#         #         self.hidden_size = self.embeddings.embedding_dim
#         #         self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)
#         #
#         #         if self.args.disc_embedding_checkpoint == None:
#         #             self.spell_length_disc = sum(self.template)
#         #             self.prompt_encoder_disc = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
#         #             self.prompt_encoder_disc = self.prompt_encoder_disc.cuda()
#         #             self.prompt_encoder = self.prompt_encoder_disc
#         #             self.disc_embedding = self.embeddings
#         #         else:
#         #             self.disc_model = _create_model(self.args.disc_embedding_checkpoint[:-5]).to(self.args.device)
#         #             self.spell_length_disc = sum(self.args.template_disc)
#         #             self.disc_embedding = self.disc_model.get_input_embeddings()
#         #             self.prompt_encoder_disc = PromptEncoder(self.args.template_disc, self.disc_embedding.embedding_dim, self.tokenizer, args)
#         #             self.prompt_encoder_disc = self.prompt_encoder_disc.to(self.args.device)
#         #             self.prompt_encoder_disc.load_state_dict(self.load_prompt(self.args.disc_embedding_checkpoint))
#         #
#         #         self.fc_loss = CrossEntropyLoss()
#         #
#         #
#         #
#         #     def load_prompt(self, embedding_checkpoint):
#         #         checkpoint = torch.load(embedding_checkpoint)
#         #         prompt_embedding = checkpoint['embedding']
#         #         return prompt_embedding
#         #
#         #
#         #     def generate(self, prompts_ids, max_length, desired_att = None,  beta = 0.5):
#         #         """
#         #         generation forward based on given prompt tokens,
#         #         Args:
#         #             prompt_ids: the  prompt tokens
#         #             max_length: the max len of the generation
#         #         Returns:
#         #             generated_texts:[generated tokens]
#         #         """
#         #         cur_len = prompts_ids.shape[1]
#         #         logits = []
#         #         output_ids = prompts_ids
#         #         return_dict = {}
#         #         eos_flag = torch.ones([prompts_ids.shape[0]]).type(torch.uint8).to(self.args.device)
#         #
#         #
#         #         # start = datetime.datetime.now()
#         #         past = None
#         #         while cur_len <= max_length:
#         #             past_k_v =  past
#         #             future_logits, past = self.generate_soft_tokens(output_ids, past_k_v)
#         #             next_token_logits = future_logits.clone().detach().squeeze(1)
#         #             perturb_logits = self.feedback_from_discriminator(output_ids, future_logits.unsqueeze(1), desired_att)
#         #
#         #             next_token_logits_prob = torch.softmax(next_token_logits, dim=1)
#         #
#         #             perturb_logits_prob = torch.softmax(perturb_logits, dim=1)
#         #
#         #             next_token_logits_prob = perturb_logits_prob.mul(next_token_logits_prob)
#         #
#         #             next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
#         #             ## avoid eos token appeals continuely
#         #             eos_flag = eos_flag.mul((next_tokens != self.tokenizer.eos_token_id).type(torch.uint8))# if flag = 0, it means the generated is over
#         #             next_tokens = next_tokens.mul(eos_flag)
#         #             next_tokens[next_tokens == 0] = self.tokenizer.eos_token_id
#         #             output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)
#         #             print("cur_len is:",cur_len)
#         #             cur_len = cur_len + 1
#         #
#         #         # end = datetime.datetime.now()
#         #         # print("runing time is:",end-start)
#         #
#         #         return_dict = {"generated_tokens":output_ids}
#         #         return return_dict
#         #
#         #
#         #     def generate_soft_tokens(self, generated_tokens, past_key_values= None):
#         #
#         #         if past_key_values!= None:
#         #             last_embeds =self.embeddings(generated_tokens[:, -1]).unsqueeze(1)#get its embeddings
#         #             # print("last_embeds：", last_embeds.shape)
#         #             with torch.no_grad():
#         #                 outputs = self.model(inputs_embeds=last_embeds,
#         #                                                past_key_values = past_key_values,
#         #                                                return_dict=True)
#         #
#         #         else:
#         #             attention_mask = (generated_tokens!=self.tokenizer.eos_token_id).type(torch.uint8)
#         #             position_ids = attention_mask.long().cumsum(-1)- 1
#         #             position_ids.masked_fill_(attention_mask == 0, 0)
#         #             last_embeds =self.embeddings(generated_tokens) #get its embeddings
#         #
#         #             with torch.no_grad():
#         #                 outputs = self.model(inputs_embeds=last_embeds,
#         #                                                past_key_values = past_key_values,
#         #                                                attention_mask = attention_mask,
#         #                                                position_ids = position_ids,
#         #                                                return_dict=True)
#         #
#         #         next_token_logits = outputs.logits[:, -1, :]
#         #
#         #         next_token_logits = self.top_k_top_p_filtering(next_token_logits.squeeze(1), top_k=self.args.ranking_scope, top_p=self.args.top_p, filter_value=BIG_CONST)
#         #
#         #         return next_token_logits, outputs.past_key_values
#         #
#         #
#         #
#         #     def discriminator_predict(self, input_ids):
#         #
#         #         input_ids_left_pad, length_generated_tokens = self.pad_left_to_right(input_ids, self.tokenizer.eos_token_id)
#         #
#         #         musk =  (input_ids_left_pad != self.tokenizer.eos_token_id).type(torch.uint8)
#         #
#         #         pred_ids  = self.predict(input_ids_left_pad, musk)
#         #
#         #         return pred_ids
#         #
#         #
#         #     def scores_predict(self, input_ids):
#         #
#         #         input_ids_left_pad, length_generated_tokens = self.pad_left_to_right(input_ids, self.tokenizer.eos_token_id)
#         #
#         #         musk =  (input_ids_left_pad != self.tokenizer.eos_token_id).type(torch.uint8)
#         #
#         #         pred_scores  = self.predict_scores(input_ids_left_pad, musk)
#         #
#         #         return pred_scores
#         #
#         #
#         #     def top_k_top_p_filtering(self,
#         #         logits,
#         #         top_k = 0,
#         #         top_p = 1.0,
#         #         filter_value = BIG_CONST ,
#         #         min_tokens_to_keep = 1,
#         #     ):
#         #         """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
#         #         Args:
#         #             logits: logits distribution shape (batch size, vocabulary size)
#         #             if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
#         #             if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
#         #                 Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
#         #             Make sure we keep at least min_tokens_to_keep per batch example in the output
#         #         From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
#         #         """
#         #         if top_k > 0:
#         #             top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
#         #             # Remove all tokens with a probability less than the last token of the top-k
#         #             indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#         #             logits[indices_to_remove] = filter_value
#         #
#         #         if top_p < 1.0:
#         #             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         #             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#         #
#         #             # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
#         #             sorted_indices_to_remove = cumulative_probs > top_p
#         #             if min_tokens_to_keep > 1:
#         #                 # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
#         #                 sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
#         #             # Shift the indices to the right to keep also the first token above the threshold
#         #             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         #             sorted_indices_to_remove[..., 0] = 0
#         #
#         #             # scatter sorted tensors to original indexing
#         #             indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#         #             logits[indices_to_remove] = filter_value
#         #
#         #         return logits
#         #
#         #
#         #     def pad_left_to_right(self, inputs, pad_id):
#         #         trans_inputs = torch.empty_like(inputs)
#         #
#         #         input_remove_prompt = inputs[:, self.prompt_pad_length:]
#         #
#         #         index_musk =  (input_remove_prompt != pad_id).type(torch.uint8) # only calculte the token which is not eos
#         #
#         #         length_of_generated_text = torch.sum(index_musk, 1)
#         #
#         #         valid_number_length = length_of_generated_text + self.prompt_pad_length
#         #
#         #         count =0
#         #         for index, seq in zip(valid_number_length, inputs):
#         #
#         #             if index == 0 or index == inputs.shape[1]:
#         #                 trans_inputs[count] = seq
#         #             else:
#         #                 trans_inputs[count][-index:]   = seq[:index]
#         #                 trans_inputs[count][:inputs.shape[1]-index]  = seq[index:]
#         #             count +=1
#         #
#         #         return trans_inputs, length_of_generated_text
#         #
#         #
#         #
#         #     def  feedback_from_discriminator(self, input_ids, logits_seq, desired_att):
#         #             logits_seq = logits_seq.squeeze(1)
#         #             top_logits, top_indices = logits_seq.topk(self.args.ranking_scope, dim=1) # batch x topk
#         #
#         #             scores = []
#         #             candidates = []
#         #             for logit_id,  ids  in  zip(top_indices, input_ids):
#         #                 data = ids.expand(self.args.ranking_scope, -1)
#         #                 new_input_candidates = torch.cat([data, logit_id.unsqueeze(1)], dim=1) # batch x topk x seq+1
#         #                 candidates.append(new_input_candidates)
#         #             candidates = torch.cat(candidates, dim=0)
#         #
#         #             musk =  (candidates != self.tokenizer.eos_token_id).type(torch.uint8)
#         #             pred_scores  = self._predict_scores(candidates, musk)
#         #             pred_scores = pred_scores.reshape(input_ids.shape[0], -1)
#         #
#         #             logits_seq.scatter_(-1, top_indices, pred_scores)
#         #
#         #             indices_to_remove = logits_seq < torch.topk(logits_seq, 3)[0][..., -1, None]
#         #             logits_seq[indices_to_remove] = BIG_CONST
#         #
#         #             return logits_seq
#         #
#         #
#         #
#         #     def  gradient_feedback_from_discriminator(self, past, logits_seq, desired_att, lr):
#         #
#         #             musk_value = torch.empty_like(logits_seq).fill_(0.0).long().to(self.args.device)
#         #             indices_musk = (logits_seq== BIG_CONST)
#         #             musk_value[indices_musk] = BIG_CONST
#         #             musk_value.requires_grad =False
#         #
#         #             index = torch.nonzero(logits_seq!= BIG_CONST)
#         #
#         #             logits_seqs = torch.empty_like(logits_seq)
#         #
#         #             logits_seqs.fill_(0.0)
#         #
#         #             update_logit = logits_seqs
#         #             update_logit.requires_grad = True
#         #
#         #             optimizer = torch.optim.AdamW([{"params":update_logit}], lr = lr, amsgrad=True, weight_decay=0.1)
#         #
#         #             num_backward_iters = self.args.iter_num
#         #
#         #             for i in range(num_backward_iters):
#         #
#         #                 update_logit_ = update_logit.mul(~indices_musk) ##topk, value is orginal
#         #                 logits = update_logit_ + musk_value
#         #
#         #                 logit_softmax = torch.softmax(logits, dim=-1)
#         #
#         #                 soft_tokens = torch.matmul(logit_softmax, self.discrimirator_embedding.weight)
#         #
#         #                 loss_discrimlator = self.loss_for_desiredAtt(past, soft_tokens, desired_att)
#         #
#         #                 loss =  loss_discrimlator #+ 0.0*l1_loss
#         #
#         #                 print("the loss is:",loss)
#         #
#         #                 loss.backward()
#         #                 torch.cuda.empty_cache()
#         #                 optimizer.step()
#         #                 torch.cuda.empty_cache()
#         #                 optimizer.zero_grad()
#         #
#         #             update_logit_ = update_logit.mul(~indices_musk) ##topk, value is orginal
#         #             logits = update_logit_ + musk_value
#         #
#         #             indices_to_remove = logits < torch.topk(logits, self.args.top_k)[0][..., -1, None]
#         #             logits[indices_to_remove] = BIG_CONST
#         #
#         #             return logits.squeeze(1)
#         #
#         #
#         #     def get_query(self, x_h, prompt_tokens, x_t = None):
#         #
#         #         prompt_tensor = torch.tensor(prompt_tokens* (self.spell_length_disc)).to(x_h.device)
#         #         prompt_tensor = prompt_tensor.expand(x_h.shape[0],-1)
#         #         if x_t != None:
#         #             x_t = x_t.unsqueeze(1)
#         #             return  torch.cat([x_h, prompt_tensor, x_t], dim =1)
#         #         else:
#         #             return  torch.cat([x_h, prompt_tensor], dim =1)
#         #
#         #
#         #
#         #     def embed_input(self, queries):
#         #         bz = queries.shape[0]
#         #         queries_for_embedding = queries.clone()
#         #         raw_embeds = self.disc_embedding(queries_for_embedding)
#         #
#         #         replace_embeds = self.prompt_encoder_disc()
#         #
#         #         replace_embeds = replace_embeds.unsqueeze(0).expand(bz,-1, -1)
#         #
#         #         raw_embeds[:,-self.prompt_encoder_disc.spell_length:,: ] = replace_embeds
#         #
#         #         return raw_embeds
#         #
#         #
#         #
#         #
#         #     def _predict_scores(self, x_hs, att_mask):
#         #         bz = len(x_hs)
#         #         # construct query ids
#         #         prompt_tokens = [self.pseudo_token_id]
#         #
#         #         queries = self.get_query(x_hs, prompt_tokens)
#         #         # construct label ids
#         #         attention_mask = torch.cat([att_mask, torch.ones([att_mask.shape[0], self.prompt_encoder_disc.spell_length]).long().to(self.args.device)], dim=1)
#         #         # get embedded input
#         #
#         #         # print(queries.shape)
#         #         inputs_embeds = self.embed_input(queries)
#         #
#         #         position_ids = attention_mask.long().cumsum(-1)- 1
#         #         position_ids.masked_fill_(attention_mask == 0, 0)
#         #
#         #         # print(position_ids.shape, inputs_embeds.shape, attention_mask.shape)
#         #
#         #         with torch.no_grad():
#         #
#         #             output = self.disc_model(inputs_embeds = inputs_embeds,
#         #                             attention_mask = attention_mask,
#         #                             position_ids = position_ids,
#         #                             labels=None)
#         #
#         #         logits = output.logits[:,-1,:].squeeze(1)
#         #
#         #         binary_prob = torch.softmax(logits[:,[11274,14774]], dim=-1)
#         #
#         #         if self.args.target_type == "negative":
#         #             return binary_prob[:,1]
#         #         else:
#         #             return binary_prob[:,0]
#         #
#         #
#         #     def forward(self, x_hs, x_ts, att_mask):
#         #         bz = len(x_hs)
#         #         # construct query ids
#         #         prompt_tokens = [self.pseudo_token_id]
#         #
#         #         queries = self.get_query(x_hs, prompt_tokens)
#         #
#         #         # construct label ids
#         #         attention_mask = torch.cat([att_mask, torch.ones([att_mask.shape[0], self.prompt_encoder_disc.spell_length]).long().to(att_mask.device)], dim=1)
#         #
#         #         position_ids = attention_mask.long().cumsum(-1)- 1
#         #         position_ids.masked_fill_(attention_mask == 0, 0)
#         #
#         #         # get embedded input
#         #         inputs_embeds = self.embed_input(queries)
#         #
#         #         label_mask = att_mask
#         #
#         #         output = self.model(inputs_embeds=inputs_embeds,
#         #                             attention_mask=attention_mask,
#         #                             position_ids=position_ids,
#         #                             labels= None)
#         #
#         #         logits = output.logits[:,-1,:].squeeze(1)
#         #
#         #         loss = self.fc_loss(logits, x_ts.squeeze(1))
#         #
#         #
#         #         return loss
#         #
#         #
#         #
#         #
#         #
#         #
#         self.label_token = label_token
#         self.label_map = {}
#         self.label_token_ids = {}
#
#         # for k, v in self.label_token.items():
#         #     print(k, v, self.tokenizer.convert_tokens_to_ids(v))
#         #     self.label_map[self.tokenizer.convert_tokens_to_ids(v)] = k
#         #
#         #     self.label_token_ids[k] = self.tokenizer.convert_tokens_to_ids(v)
#         self.target_id = self.tokenizer.convert_tokens_to_ids(self.label_token['positive'])
#
#         # load prompt encoder
#         self.hidden_size = self.embeddings.embedding_dim
#         self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)
#
#         # if self.args.disc_embedding_checkpoint == None:
#         self.spell_length_disc = sum(self.template)
#         self.prompt_encoder_disc = PromptEncoder(self.template, self.hidden_size, self.tokenizer, args)
#         self.prompt_encoder_disc = self.prompt_encoder_disc.to(args.device)
#         self.prompt_encoder = self.prompt_encoder_disc
#         self.disc_embedding = self.embeddings
#         # else:
#         #     self.disc_model = _create_model(self.args.disc_embedding_checkpoint[:-5]).to(self.args.device)
#         #     self.spell_length_disc = sum(self.args.template_disc)
#         #     self.disc_embedding = self.disc_model.get_input_embeddings()
#         #     self.prompt_encoder_disc = PromptEncoder(self.args.template_disc, self.disc_embedding.embedding_dim,
#         #                                              self.tokenizer, args)
#         #     self.prompt_encoder_disc = self.prompt_encoder_disc.to(self.args.device)
#         #     self.prompt_encoder_disc.load_state_dict(self.load_prompt(self.args.disc_embedding_checkpoint))
#
#         self.fc_loss = CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss
#
#     def load_prompt(self, embedding_checkpoint):
#         checkpoint = torch.load(embedding_checkpoint)
#         prompt_embedding = checkpoint['embedding']
#         return prompt_embedding
#
#     def generate(self, prompts_ids, max_length, desired_att=None, beta=0.5):
#         """
#         generation forward based on given prompt tokens,
#         Args:
#             prompt_ids: the  prompt tokens
#             max_length: the max len of the generation
#         Returns:
#             generated_texts:[generated tokens]
#         """
#         cur_len = prompts_ids.shape[1]
#         logits = []
#         output_ids = prompts_ids
#         return_dict = {}
#         eos_flag = torch.ones([prompts_ids.shape[0]]).type(torch.uint8).to(self.args.device)
#
#         # start = datetime.datetime.now()
#         past = None
#         while cur_len <= max_length:
#             past_k_v = past
#             future_logits, past = self.generate_soft_tokens(output_ids, past_k_v)
#             next_token_logits = future_logits.clone().detach().squeeze(1)
#             perturb_logits = self.feedback_from_discriminator(output_ids, future_logits.unsqueeze(1), desired_att)
#
#             next_token_logits_prob = torch.softmax(next_token_logits, dim=1)
#
#             perturb_logits_prob = torch.softmax(perturb_logits, dim=1)
#
#             next_token_logits_prob = perturb_logits_prob.mul(next_token_logits_prob)
#
#             next_tokens = torch.multinomial(next_token_logits_prob, num_samples=1).squeeze(1)
#             ## avoid eos token appeals continuely
#             eos_flag = eos_flag.mul((next_tokens != self.tokenizer.eos_token_id).type(
#                 torch.uint8))  # if flag = 0, it means the generated is over
#             next_tokens = next_tokens.mul(eos_flag)
#             next_tokens[next_tokens == 0] = self.tokenizer.eos_token_id
#             output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)
#             print("cur_len is:", cur_len)
#             cur_len = cur_len + 1
#
#         # end = datetime.datetime.now()
#         # print("runing time is:",end-start)
#
#         return_dict = {"generated_tokens": output_ids}
#         return return_dict
#
#     def generate_soft_tokens(self, generated_tokens, past_key_values=None):
#
#         if past_key_values != None:
#             last_embeds = self.embeddings(generated_tokens[:, -1]).unsqueeze(1)  # get its embeddings
#             # print("last_embeds：", last_embeds.shape)
#             with torch.no_grad():
#                 outputs = self.model(inputs_embeds=last_embeds,
#                                      past_key_values=past_key_values,
#                                      return_dict=True)
#
#         else:
#             attention_mask = (generated_tokens != self.tokenizer.eos_token_id).type(torch.uint8)
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 0)
#             last_embeds = self.embeddings(generated_tokens)  # get its embeddings
#
#             with torch.no_grad():
#                 outputs = self.model(inputs_embeds=last_embeds,
#                                      past_key_values=past_key_values,
#                                      attention_mask=attention_mask,
#                                      position_ids=position_ids,
#                                      return_dict=True)
#
#         next_token_logits = outputs.logits[:, -1, :]
#
#         next_token_logits = self.top_k_top_p_filtering(next_token_logits.squeeze(1), top_k=self.args.ranking_scope,
#                                                        top_p=self.args.top_p, filter_value=BIG_CONST)
#
#         return next_token_logits, outputs.past_key_values
#
#     def discriminator_predict(self, input_ids):
#
#         input_ids_left_pad, length_generated_tokens = self.pad_left_to_right(input_ids, self.tokenizer.eos_token_id)
#
#         musk = (input_ids_left_pad != self.tokenizer.eos_token_id).type(torch.uint8)
#
#         pred_ids = self.predict(input_ids_left_pad, musk)
#
#         return pred_ids
#
#     def scores_predict(self, input_ids):
#
#         input_ids_left_pad, length_generated_tokens = self.pad_left_to_right(input_ids, self.tokenizer.eos_token_id)
#
#         musk = (input_ids_left_pad != self.tokenizer.eos_token_id).type(torch.uint8)
#
#         pred_scores = self.predict_scores(input_ids_left_pad, musk)
#
#         return pred_scores
#
#     def top_k_top_p_filtering(self,
#                               logits,
#                               top_k=0,
#                               top_p=1.0,
#                               filter_value=BIG_CONST,
#                               min_tokens_to_keep=1,
#                               ):
#         """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
#         Args:
#             logits: logits distribution shape (batch size, vocabulary size)
#             if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
#             if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
#                 Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
#             Make sure we keep at least min_tokens_to_keep per batch example in the output
#         From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
#         """
#         if top_k > 0:
#             top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
#             # Remove all tokens with a probability less than the last token of the top-k
#             indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#             logits[indices_to_remove] = filter_value
#
#         if top_p < 1.0:
#             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#
#             # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
#             sorted_indices_to_remove = cumulative_probs > top_p
#             if min_tokens_to_keep > 1:
#                 # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
#                 sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
#             # Shift the indices to the right to keep also the first token above the threshold
#             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#             sorted_indices_to_remove[..., 0] = 0
#
#             # scatter sorted tensors to original indexing
#             indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#             logits[indices_to_remove] = filter_value
#
#         return logits
#
#     def pad_left_to_right(self, inputs, pad_id):
#         trans_inputs = torch.empty_like(inputs)
#
#         input_remove_prompt = inputs[:, self.prompt_pad_length:]
#
#         index_musk = (input_remove_prompt != pad_id).type(torch.uint8)  # only calculte the token which is not eos
#
#         length_of_generated_text = torch.sum(index_musk, 1)
#
#         valid_number_length = length_of_generated_text + self.prompt_pad_length
#
#         count = 0
#         for index, seq in zip(valid_number_length, inputs):
#
#             if index == 0 or index == inputs.shape[1]:
#                 trans_inputs[count] = seq
#             else:
#                 trans_inputs[count][-index:] = seq[:index]
#                 trans_inputs[count][:inputs.shape[1] - index] = seq[index:]
#             count += 1
#
#         return trans_inputs, length_of_generated_text
#
#     def feedback_from_discriminator(self, input_ids, logits_seq, desired_att):
#         logits_seq = logits_seq.squeeze(1)
#         top_logits, top_indices = logits_seq.topk(self.args.ranking_scope, dim=1)  # batch x topk
#
#         scores = []
#         candidates = []
#         for logit_id, ids in zip(top_indices, input_ids):
#             data = ids.expand(self.args.ranking_scope, -1)
#             new_input_candidates = torch.cat([data, logit_id.unsqueeze(1)], dim=1)  # batch x topk x seq+1
#             candidates.append(new_input_candidates)
#         candidates = torch.cat(candidates, dim=0)
#
#         musk = (candidates != self.tokenizer.eos_token_id).type(torch.uint8)
#         pred_scores = self._predict_scores(candidates, musk)
#         pred_scores = pred_scores.reshape(input_ids.shape[0], -1)
#
#         logits_seq.scatter_(-1, top_indices, pred_scores)
#
#         indices_to_remove = logits_seq < torch.topk(logits_seq, 3)[0][..., -1, None]
#         logits_seq[indices_to_remove] = BIG_CONST
#
#         return logits_seq
#
#     def gradient_feedback_from_discriminator(self, past, logits_seq, desired_att, lr):
#
#         musk_value = torch.empty_like(logits_seq).fill_(0.0).long().to(self.args.device)
#         indices_musk = (logits_seq == BIG_CONST)
#         musk_value[indices_musk] = BIG_CONST
#         musk_value.requires_grad = False
#
#         index = torch.nonzero(logits_seq != BIG_CONST)
#
#         logits_seqs = torch.empty_like(logits_seq)
#
#         logits_seqs.fill_(0.0)
#
#         update_logit = logits_seqs
#         update_logit.requires_grad = True
#
#         optimizer = torch.optim.AdamW([{"params": update_logit}], lr=lr, amsgrad=True, weight_decay=0.1)
#
#         num_backward_iters = self.args.iter_num
#
#         for i in range(num_backward_iters):
#             update_logit_ = update_logit.mul(~indices_musk)  ##topk, value is orginal
#             logits = update_logit_ + musk_value
#
#             logit_softmax = torch.softmax(logits, dim=-1)
#
#             soft_tokens = torch.matmul(logit_softmax, self.discrimirator_embedding.weight)
#
#             loss_discrimlator = self.loss_for_desiredAtt(past, soft_tokens, desired_att)
#
#             loss = loss_discrimlator  # + 0.0*l1_loss
#
#             print("the loss is:", loss)
#
#             loss.backward()
#             torch.cuda.empty_cache()
#             optimizer.step()
#             torch.cuda.empty_cache()
#             optimizer.zero_grad()
#
#         update_logit_ = update_logit.mul(~indices_musk)  ##topk, value is orginal
#         logits = update_logit_ + musk_value
#
#         indices_to_remove = logits < torch.topk(logits, self.args.top_k)[0][..., -1, None]
#         logits[indices_to_remove] = BIG_CONST
#
#         return logits.squeeze(1)
#
#     def get_query(self, x_h, prompt_tokens, x_t=None):
#
#         prompt_tensor = torch.tensor(prompt_tokens * (self.spell_length_disc)).to(x_h.device)
#         prompt_tensor = prompt_tensor.expand(x_h.shape[0], -1)
#         if x_t != None:
#             x_t = x_t.unsqueeze(1)
#             return torch.cat([x_h, prompt_tensor, x_t], dim=1)
#         else:
#             return torch.cat([x_h, prompt_tensor], dim=1)
#
#     def embed_input(self, queries):
#         bz = queries.shape[0]
#         queries_for_embedding = queries.clone()
#         raw_embeds = self.disc_embedding(queries_for_embedding)
#
#         replace_embeds = self.prompt_encoder_disc()
#
#         replace_embeds = replace_embeds.unsqueeze(0).expand(bz, -1, -1)
#
#         raw_embeds[:, -self.prompt_encoder_disc.spell_length:, :] = replace_embeds
#
#         return raw_embeds
#
#     def _predict_scores(self, x_hs, att_mask, reward=False):
#         bz = len(x_hs)
#         # construct query ids
#         prompt_tokens = [self.pseudo_token_id]
#
#         queries = self.get_query(x_hs, prompt_tokens)
#         # construct label ids
#         attention_mask = torch.cat([att_mask,
#                                     torch.ones([att_mask.shape[0], self.prompt_encoder_disc.spell_length]).long().to(
#                                         self.args.device)], dim=1)
#         # get embedded input
#
#         # print(queries.shape)
#         inputs_embeds = self.embed_input(queries)
#
#         position_ids = attention_mask.long().cumsum(-1) - 1
#         position_ids.masked_fill_(attention_mask == 0, 0)
#
#         # print(position_ids.shape, inputs_embeds.shape, attention_mask.shape)
#
#         with torch.no_grad():
#
#             output = self.model(inputs_embeds=inputs_embeds,
#                                      attention_mask=attention_mask,
#                                      position_ids=position_ids,
#                                      labels=None)
#
#         logits = output.logits[:, -1, :].squeeze(1)
#
#         binary_prob = torch.softmax(logits[:, [11274, 14774]], dim=-1)
#
#         # if self.args.target_type == "negative":
#         #     return binary_prob[:, 1]
#         # else:
#         #     return binary_prob[:, 0]
#         if reward==False:
#             results = torch.where(binary_prob[:,0]>binary_prob[:,1],11274,14774)
#             return results.unsqueeze(-1).tolist()
#         else:
#             # binary_prob = torch.softmax(binary_prob,dim=-1)
#             return binary_prob[:,0]
#
#     def forward(self, x_hs, att_mask, label=None):
#         bz = len(x_hs)
#         # construct query ids
#         prompt_tokens = [self.pseudo_token_id]
#
#         queries = self.get_query(x_hs, prompt_tokens)
#
#         # construct label ids
#         attention_mask = torch.cat([att_mask,
#                                     torch.ones([att_mask.shape[0], self.prompt_encoder_disc.spell_length]).long().to(
#                                         att_mask.device)], dim=1)
#
#         position_ids = attention_mask.long().cumsum(-1) - 1
#         position_ids.masked_fill_(attention_mask == 0, 0)
#
#         # get embedded input
#         inputs_embeds = self.embed_input(queries)
#
#         label_mask = att_mask
#
#         output = self.model(inputs_embeds=inputs_embeds,
#                             attention_mask=attention_mask,
#                             position_ids=position_ids,
#                             labels=None)
#
#         logits = output.logits[:, self.spell_length_disc:, self.target_id].squeeze(1)
#
#         score = torch.sigmoid(logits)
#
#
#         if label==None:
#             ave = (score - label) * x_attn
#             print('ave_deprency:',ave.sum()/x_attn.sum())
#             return ave.sum(),x_attn.sum()
#         else:
#             print(score[0],score[1])
#             print(label[0],label[1])
#             loss = self.mse_loss(score, label)
#             # loss = self.fc_loss(logits, x_ts.squeeze(1))
#             return loss






