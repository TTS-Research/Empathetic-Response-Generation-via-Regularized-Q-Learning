from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizerFast, AutoModel, GPT2Tokenizer, BertTokenizer, BertModel, BertLMHeadModel
import json
from torch import nn
import torch
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import random
import numpy as np


class GPT2DataSet(Dataset):
    def __init__(self, tokenizer=None, max_len=256, root_path='./', train_path='single_emo_T_train.json', val_path='single_emo_T_valid.json', test_path='single_emo_T_test.json', status='train',
                 dataset_root_path='dataset', shuffle=True, length_lower_bound=30, length_upper_bound=180, seperate_word=['，', '。', '？', '！', '、', '[', ']', '（', '）', '～']) -> None:
        self.file_path = os.path.join(root_path, dataset_root_path, (
            train_path if 'train' in status else (val_path if 'val' in status else test_path)))
        with open(self.file_path) as f:
            # print(len(f.readlines()[0]))
            # print(self.file_path)
            self.data = json.load(f)
            # self.data = json.load()
        # f = open(self.file_path)
        # self.data = json.load(f)
        self.dataset = []
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-chinese') if tokenizer is None else tokenizer
        if status == 'test':
            self.data = [i[0] + i[1] for i in self.data]
        self.max_len = max_len
        temp_max_len = 0
        for idx, line in enumerate(self.data):
            if '[' not in line or ']' not in line or line.count(']') != 1 or line.count('[') != 1 or '哈' in line or '我也是' in line or '嗯' in line:
                continue
            else:
                ptr = 0
                while line[ptr] != ']':
                    ptr += 1
                counter = ptr + 1
                while counter < len(line) and line[counter] not in seperate_word:
                    counter += 1
                if counter - ptr <= 5:
                    continue
                temp_max_len = max(temp_max_len, len(line))
                if len(line) <= length_lower_bound or len(line) >= length_upper_bound:
                    continue
                if idx % 100000 == 0:
                    print("Dataset processing to...", idx)

                res = self.tokenizer.encode(line, return_tensors='pt')[0]
                self.dataset.append(res)
        if shuffle:
            random.shuffle(self.dataset)
        print(status, "Dataset Max Length : ", temp_max_len)
        print(status, "Dataset size : ", len(self.dataset))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        return self.dataset[index]

    def shuffle(self):
        random.shuffle(self.dataset)

    @staticmethod
    def get_toxic_ids_and_non_sense_response(tokenizer,
                                             dirty_words=['幹!', '賤貨', '米蟲', '王八', '王八蛋', '不要臉', '吃屎', '敗類', '智障', '白癡', '賤人', '下流',
                                                          '死肥豬', '人渣', '神經病', '賤', '尼瑪', '無恥', '婊', '娘炮', '魯蛇', '廢物', '腦殘'],
                                             non_sense_sentences=['嗯', '嗯嗯', '隨便啦', '隨便啊', '都可以', '呵呵', '哈哈', '喔', '笑死', '是喔', '好吧', '我不知道',
                                                                  '還好', '是啊', '對啊', '我也是', '嘿嘿']):
        # create toxic word list
        toxic_ids = []
        for i in range(len(dirty_words)):
            ids = tokenizer.encode(dirty_words[i])
            toxic_ids.append(ids[1:-1])
        # create non sense sentence list
        non_sense_ids = []
        for i in range(len(non_sense_sentences)):
            ids = tokenizer.encode(non_sense_sentences[i])
            non_sense_ids.append(ids[1:])
        return toxic_ids, non_sense_ids


class Reward(nn.Module):
    def __init__(self, gpt: GPT2LMHeadModel, question_mark_token, toxic_words: list, non_sense_response: list, eos_token=0, device="cpu", gpt_tokenizer=None, bos_token=101, root_path='./dataset',
                 length_word_weight=0.1, question_reward_weight=0.1, coherence_weight=0.1, toxicity_weight=0.1, ease_of_answering_weight=0.1, get_reward_semantic_coherence_weight=0.1) -> None:
        super(Reward, self).__init__()
        self.reward_coefficient = torch.tensor([length_word_weight, question_reward_weight, coherence_weight,
                                               toxicity_weight, ease_of_answering_weight, get_reward_semantic_coherence_weight], device=device)
        self.gpt = copy.deepcopy(gpt)
        self.gpt = self.gpt.to(device)
        self.gpt_tokenizer = BertTokenizer(vocab_file=os.path.join(
            root_path, 'GPT-2/vocab_small.txt')) if gpt_tokenizer is None else gpt_tokenizer
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.device = device
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.question_mark_token = question_mark_token
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-chinese')
        self.bert = AutoModel.from_pretrained(
            'ckiplab/bert-base-chinese').to(device)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.toxic_words = toxic_words
        self.non_sense_response = non_sense_response

    def update_model(self, gpt):
        self.gpt = copy.deepcopy(gpt)
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.gpt.eval()

    def to_device(self, device):
        self.device = device

    def update_reward_coefficient(self, gamma):
        self.reward_coefficient = copy.deepcopy(gamma)

    def forward(self, state: list):
        # state = [batch_size, {prev_utterance, response}] # elements of second dimension are dictionaries composed of previous utterance and subsequent response
        # prev_utterance = [prev_utterance_seq_length, ]
        # response = [response_seq_length, ]
        self.gpt.eval()
        for idx in range(len(state)):
            state[idx]['prev_utterance'] = state[idx]['prev_utterance'].to(
                self.device)
            state[idx]['response'] = state[idx]['response'].to(self.device)
        with torch.no_grad():
            # exp for avoiding +- offset each other, if you don't want it, go head delete .exp()
            reward = self.reward_coefficient[0] * \
                (self.get_length_reward(state).exp())
            reward += self.reward_coefficient[1] * \
                (self.get_question_reward(state).exp())
            reward += self.reward_coefficient[2] * \
                (self.get_coherence(state).exp())
            reward += self.reward_coefficient[3] * (self.get_toxicity(state))
            reward += self.reward_coefficient[4] * \
                (self.get_ease_of_answering(state).exp())
            reward += self.reward_coefficient[5] * \
                (self.get_reward_semantic_coherence(state).exp())

        return F.normalize(reward, dim=0)

    # reverse means prev_utterance = response, response = prev_utterance
    def get_response_prob(self, state, require_grad=False, reverse=False):
        state = copy.deepcopy(state)
        if reverse:
            state = [{"prev_utterance": state[idx]['response'],
                      'response': state[idx]['prev_utterance']} for idx in range(len(state))]

        def get_prob():
            probability = torch.ones((len(state)), device=self.device)
            for index, state_dict in enumerate(state):
                utterance, response = state_dict['prev_utterance'].clone(
                ).detach(), state_dict['response'].clone().detach()
                probability[index] *= self.p_seq2seq(utterance, response)
            return F.normalize(probability, dim=0)
        self.gpt.train(require_grad)
        if not require_grad:
            with torch.no_grad():
                return get_prob()
        else:
            return get_prob()

    def p_seq2seq(self, up, down):
        input_ids = up.clone().detach()
        probability = 1e20  # refrain from continuously multiplying number 0 < number < 1 such that probability would be too small, which in turn would cause precision problem
        for i in range(len(down)):
            logits = self.gpt(input_ids=input_ids)['logits']
            logits = F.softmax(logits, dim=-1)
            probability *= logits[-1, down[i]]
            input_ids = torch.cat((input_ids, torch.tensor(
                [down[i]], device=self.device)), dim=-1)
            if down[i] == self.eos_token:
                break
        return probability

    def get_length_reward(self, state):
        return F.normalize(torch.tensor([len(state[idx]['response']) for idx in range(len(state))], device=self.device, dtype=torch.float), dim=0)

    def get_question_reward(self, state):
        return F.normalize(torch.tensor([1 if self.question_mark_token in state[idx]['response'] else 0 for idx in range(len(state))], device=self.device, dtype=torch.float), dim=0)

    def transform_from_gpt_to_bert_tokens(self, input_id):
        original_string = self.gpt_tokenizer.decode(
            input_id, skip_special_tokens=True).replace(" ", "")
        return self.bert_tokenizer.encode(original_string, return_tensors='pt')[0].to(self.device)

    def get_coherence(self, state):
        state = [{"prev_utterance": self.transform_from_gpt_to_bert_tokens(
            state[idx]['prev_utterance']), 'response': self.transform_from_gpt_to_bert_tokens(state[idx]['response'])} for idx in range(len(state))]
        coherence = torch.zeros(
            (len(state)), device=self.device, dtype=torch.float)
        cos = nn.CosineSimilarity(dim=-1)
        for idx in range(len(state)):
            utterance = self.bert(input_ids=state[idx]['prev_utterance'].unsqueeze(0))[
                'last_hidden_state'][0][0]
            response = self.bert(input_ids=torch.cat((torch.tensor(
                [self.bos_token], device=self.device), state[idx]['response'])).unsqueeze(0))['last_hidden_state'][0][0]
            sim = cos(utterance, response)
            coherence[idx] = sim
        return F.normalize(coherence, dim=0)

    def get_toxicity(self, state):
        toxicity = []
        for idx, value in enumerate(state):
            counter = 0
            for word in self.toxic_words:
                if self.x_in_y(word, value['response']):
                    counter -= 1
            toxicity.append(counter)
        return torch.tensor(toxicity, device=self.device)

    def get_ease_of_answering(self, state):
        ease_of_answering = torch.zeros(
            (len(state)), device=self.device, dtype=torch.float)
        for idx in range(len(state)):
            temp = 0
            for sentence in self.non_sense_response:
                temp += self.p_seq2seq(state[idx]
                                       ['response'], sentence) / len(sentence)
            temp *= (- 1 / len(self.non_sense_response))
            ease_of_answering[idx] = temp
        return F.normalize(ease_of_answering, dim=0)

    def get_reward_semantic_coherence(self, state):
        forward = self.get_response_prob(state)
        backward = self.get_response_prob(state, reverse=True)
        return F.normalize(torch.tensor([forward[idx] / len(state[idx]['response']) + backward[idx] / len(state[idx]['prev_utterance']) for idx in range(len(state))], device=self.device, dtype=torch.float), dim=0)

    def x_in_y(self, query, base):
        try:
            l = len(query)
        except TypeError:
            l = 1
            query = type(base)((query,))

        for i in range(len(base)):
            if base[i: i+l] == query:
                return True
        return False

    @staticmethod
    def sentence2id(sentence, tokenizer, emotion_dict={'其它': 0, '喜歡': 1, '悲傷': 2, '噁心': 3, '憤怒': 4, '開心': 5}, max_len=None, pad_to_max_len=False):

        utterance = sentence[: sentence.index(']') + 1]
        response = sentence[sentence.index(']') + 1:]
        prev_utterance = tokenizer.encode(utterance, return_tensors='pt')[0]
        response = tokenizer.encode(response, return_tensors='pt')[0][1:]
        if max_len is not None and pad_to_max_len:
            padding = max_len - prev_utterance.shape[-1]
            if padding > 0:
                prev_utterance = torch.cat(
                    (prev_utterance, torch.zeros(padding, dtype=torch.int64) - 1), dim=-1)
            padding = max_len - response.shape[-1]
            if padding > 0:
                response = torch.cat((response, torch.zeros(
                    padding, dtype=torch.int64) - 1), dim=-1)

        return {'prev_utterance': prev_utterance, 'response': response}
