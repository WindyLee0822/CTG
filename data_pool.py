from typing import List
from copy import deepcopy
import torch
import random

class DataPool:
    def __init__(self):
        # self.tree_tokens = tree_tokens
        # self.n_extra_tokens = n_extra_tokens
        self.cat_mask = []
        self.ids_pool, self.mask_pool, self.score_pool, self.life = [], [], [], []
        self.sort_score=[]
        self.r_limit =0

    def add(self, input_ids: List[int], output_mask: List[int], scores: List[List],pos=True):

        ids_pool, mask_pool, score_pool, life = [], [], [], []
        for a,b,c,d in zip(self.ids_pool,self.mask_pool,self.score_pool,self.life):
            d-=1
            if d>0:
                ids_pool.append(a)
                mask_pool.append(b)
                score_pool.append(c)
                life.append(d)

        self.ids_pool,self.mask_pool,self.score_pool, self.life = ids_pool,mask_pool,score_pool,life

        self.ids_pool.extend(input_ids)
        self.mask_pool.extend(output_mask)
        self.score_pool.extend(scores)
        self.life.extend([8] * len(input_ids))

        sort_score=[]
        for score,mask in zip(self.score_pool,self.mask_pool):
            assert len(score)+1 == len(mask)
            sort_score.extend([s for s,m in zip(score,mask[1:]) if m!=0])
        # sorted_score = [y for x in self.score_pool for y in x]
        # score_tensor = torch.cat([torch.zeros(len(self.ids_pool),1),torch.tensor(self.score_pool)],dim=-1)
        # sort_score = (score_tensor * torch.tensor(mask_pool,dtype=torch.bool)).view(-1).tolist()
        sorted_score = sorted(sort_score,reverse=True)

        # r_limit = sorted_score[len(sorted_score)//5]  # for translation,sentiment task,k=5
        # self.r_limit = r_limit
        # p_limit = sorted_score[len(sorted_score)//5*4]
        #
        # # logging.info(f'score_distribution:{[sorted_score[0],r_limit,p_limit,sort_score[-1]]}')
        # r_limit = max(r_limit,0)
        # p_limit = min(p_limit,0)
        self.r_limit=[]
        self.r_score=[]
        quantile_num=5
        for i in range(quantile_num-1):
            self.r_limit.append(sorted_score[len(sorted_score)//quantile_num*(i+1)])
        self.r_limit.append(sorted_score[-1])
        self.r_limit.insert(0,sorted_score[0])
        self.r_interval = [self.r_limit[i-1]-self.r_limit[i] for i in range(1,len(self.r_limit))]
        self.r_limit = self.r_limit[1:]
        # ave = sum(sorted_score[len(sorted_score) // quantile_num * (quantile_num -1):])/(len(sorted_score)//quantile_num)
        # self.r_score.append(ave)
        # self.r_score = [i+abs(ave)/2 for i in self.r_score]
        # maltitude = 1/max(self.r_score)
        # self.r_score = [i * maltitude for i in self.r_score]
        # print(f'shrehold:{self.r_limit},ave_score:{self.r_score}')

        cur=[]
        for score in self.score_pool:
            cur_ele =[]
            for ele in score:
                # if ele>r_limit:
                #     cur_ele.append(1)
                # elif ele < p_limit:
                #     cur_ele.append(-0.2)  #translation sentiment -0.2
                # else:
                #     # if score[-1] > r_limit/2:
                #     #     cur_ele.append(random.uniform(0,0.01))
                #     # elif score[-1] < p_limit:
                #     #     cur_ele.append(-random.uniform(0,0.00001))
                #     # else:
                #     cur_ele.append(1e-3)
                flag = 1
                for i in range(len(self.r_limit)):
                    if ele >= self.r_limit[i]:
                        flag =0
                        # cur_ele.append(self.r_limit[i]+random.gauss(ele-self.r_limit[i],0.6))
                        cur_ele.append(self.r_limit[i] + max(min(random.gauss(ele-self.r_limit[i],0.1)* self.r_interval[i],self.r_interval[i]),0))
                        # cur_ele.append(self.r_limit[i] + random.random() * self.r_interval[i])
                        # cur_ele.append(ele)
                        break
                assert flag==0, f"element {ele}"
            cur.append(cur_ele)

        cur = ((torch.sigmoid((torch.tensor(cur) + abs(sorted_score[len(sorted_score)//10*(9)]))/sorted_score[0]) - 0.5 ) * 2).tolist()

        self.cat_mask = cur

    def get_data(self):
        return deepcopy(self.ids_pool), deepcopy(self.mask_pool), deepcopy(self.cat_mask)

