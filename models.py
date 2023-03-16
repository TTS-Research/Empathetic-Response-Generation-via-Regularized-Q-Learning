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
from typing import Iterable
from tqdm import tqdm
import csv
class BeamHypotheses(object):
    def __init__(self, num_beams, max_length = 200, length_penalty = 0.7):
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.num_beams = num_beams # beam size
        self.beams = [] # best sequences and corresponding scores
        self.worst_score = None

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, mask, sum_logprobs, cur_len, utter_length, prev_utterance, response, eos_token):
        score = sum_logprobs / (cur_len  ** self.length_penalty) # calculate penalized score
        if len(self) < self.num_beams or score > self.worst_score:
            if response[-1] != eos_token:
                response = torch.cat((response, response.new([eos_token])))
            self.beams.append((score, hyp, mask, utter_length, sum_logprobs, prev_utterance, response))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _0, _1, _2, _3, _4, _5) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score) if self.worst_score is not None else score

    def is_done(self, best_sum_logprobs, cur_len):
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / ((cur_len) ** self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret

class Q(nn.Module):
    def __init__(self, bert = None, bert_name = 'bert-base-chinese', bert_tokenizer = None, gpt_tokenizer = None, down_stream_features = 1024, only_down_stream = True, gamma = 0.9, device = 'cpu') -> None:
        super(Q, self).__init__()
        self.gamma = gamma
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_name) if bert_tokenizer is None else bert_tokenizer
        self.bert = AutoModel.from_pretrained(bert_name) if bert is None else copy.deepcopy(bert)
        self.gpt_tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.only_down_stream = only_down_stream
        if only_down_stream:
           self.bert.eval()
           for p in self.bert.parameters():
               p.require_grad = False
        self.down_stream = nn.Sequential( nn.Linear(in_features = self.bert.pooler.dense.out_features, out_features = down_stream_features // 2),
                                          nn.BatchNorm1d(down_stream_features // 2),
                                          nn.ReLU(),
                                          nn.Linear(in_features = down_stream_features // 2, out_features = down_stream_features),
                                          nn.ReLU(),
                                          nn.Linear(in_features = down_stream_features, out_features = down_stream_features),
                                          nn.ReLU(),
                                          nn.Dropout(0.1),
                                          nn.Linear(in_features = down_stream_features, out_features = down_stream_features // 4),
                                          nn.BatchNorm1d(down_stream_features // 4),
                                          nn.ReLU(),
                                          nn.Linear(in_features = down_stream_features // 4, out_features = 1)).to(device)
        self.bert.to(device)
        self.device = device
    def transform_from_gpt_to_bert_tokens(self, input_id):
        '''
            input_ids : [batch_size, sequences]
        '''
        original_string = [self.gpt_tokenizer.decode(input, skip_special_tokens = True).replace(" ", "") for input in input_id]
        return [self.bert_tokenizer.encode(string, return_tensors = 'pt')[0] for string in original_string]
    def get_processed(self, prev_utterance, bert_tokens = False, max_len = 256):
            if not bert_tokens:
                prev_utterance = self.transform_from_gpt_to_bert_tokens(prev_utterance)
            input_ids = torch.zeros((len(prev_utterance), max_len)) - 1
            for idx, utter in enumerate(prev_utterance):       
                input_ids[idx][:min(max_len, len(utter))] = utter[:min(max_len, len(utter))]
            mask = input_ids.ge(0)
            input_ids[~mask] = 0
            mask = mask.float()
            return input_ids, mask
    def forward(self, prev_utterance, response = None, mask = None, bert_tokens = False, max_len = 256, processed = False):
        '''
            prev_utterance : [batch_size, sequences]
            response : [batch_size, sequences]
            mask : [batch_size, sequences]
        '''
        prev_utterance = prev_utterance if response is None else [torch.cat((utt, res), dim = -1) for utt, res in zip(prev_utterance, response)]

        # creating mask
        if not processed or mask is None or not bert_tokens:
            prev_utterance, mask = self.get_processed(prev_utterance = prev_utterance, bert_tokens = bert_tokens, max_len = max_len)
        prev_utterance = prev_utterance.to(self.device)
        mask = mask.to(self.device)
        if self.only_down_stream:
            with torch.no_grad():
                up_stream = self.bert(input_ids = prev_utterance.to(torch.long), attention_mask = mask.to(torch.long))['last_hidden_state'][:, 0, :].view(len(prev_utterance), -1)
        else:
            up_stream = self.bert(input_ids = prev_utterance.to(torch.long), attention_mask = mask.to(torch.long))['last_hidden_state'][:, 0, :].view(len(prev_utterance), -1)
        return self.down_stream(up_stream).view(len(prev_utterance))

class GPT2Wrapper(nn.Module):
    def __init__(self, gpt = None, tokenizer = None, device = 'cpu'):
        super(GPT2Wrapper, self).__init__()
        self.gpt = gpt if gpt is not None else GPT2LMHeadModel.from_pretrained(os.path.join('.', './GPT-2/GPT2_finetune_1'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.vocab_size = self.tokenizer.vocab_size
        self.special_tokens = {}
        for idx, key in enumerate(self.tokenizer.all_special_tokens):
            self.special_tokens[key] = self.tokenizer.all_special_ids[idx]
        self.device = device
    def update_gpt(self, gpt):
        self.gpt = gpt
        self.gpt.to(self.device)
    def forward(self, prev_utterance, response = None, beam = 3, max_len = 100, require_grad = True, device = 'mps', beam_search = False, seqerate_sequence = [102, 101, 138, 0, 140, 103, 511, 8013, 8043, 8080, 8049, 510, 8024]):
        self.device = device
        self.gpt.to(device)
        if not require_grad:
            with torch.no_grad():
                    rslt, msk, utrlen, rsltprvrnce, rsltrspse = self.beam_search(prev_utterance, response = response, beam = beam, max_len = max_len, beam_search = beam_search, seqerate_sequence = seqerate_sequence)
        else:
                rslt, msk, utrlen, rsltprvrnce, rsltrspse = self.beam_search(prev_utterance, response = response, beam = beam, max_len = max_len, beam_search = beam_search, seqerate_sequence = seqerate_sequence)
        return rslt, msk, utrlen, rsltprvrnce, rsltrspse
    def get_prob(self, result, mask, prev_utterance, response):
        utter_len = [len(utt) for utt in prev_utterance]
        logits = torch.log(F.softmax(self.gpt(input_ids = result, attention_mask = mask)['logits'], dim = -1))
        probs = torch.zeros(len(result)).to(self.device)
        for idx, res in enumerate(response):
            res_len = len(res)
            # print(res_len, utter_len[idx])
            for idx_, digit in enumerate(res):
                if digit == 0:
                    break
                probs[idx] += logits[idx][idx_ + utter_len[idx] - 1][digit]
            probs[idx] /= res_len
        return probs
    def beam_search(self, prev_utterance, response = None, eos_word = '[SEP]', beam = 3, max_len = 200, length_penalty_for_hypothesis = 0.8, emotion = ['喜歡', '悲傷', '噁心', '憤怒', '開心', '其它'], beam_search = False, seqerate_sequence = [102, 101, 138, 0, 140, 103, 511, 8013, 8043, 8080, 8049, 510, 8024]):
        '''
            prev_utterance = [seq_length_prev_utterance]
            response = [seq_length_response]
        '''
        eos_token = self.special_tokens[eos_word]
        # prev_utterance = prev_utterance if response is None else [torch.cat((utt, res)) for utt, res in zip(prev_utterance, response)]
        batch_size = len(prev_utterance)
        utter_length = torch.zeros((batch_size), dtype = torch.long, device = self.device)
        ''' create mask '''
        input_ids = torch.zeros((batch_size, max_len * 2), dtype = torch.long, device = self.device) - 1 # * 2 is for latter we will concatenate the generated sequences to them
        with torch.no_grad():
            for idx, utt in enumerate(prev_utterance):
                    if response is not None: # means prev_utterance has not been processed
                        if 138 not in response[idx][-5: ] and 140 not in response[idx][-5: ]:
                            response[idx] = torch.cat((response[idx][:-1] if response[idx][-1] == eos_token else response[idx], self.tokenizer.encode('[' + emotion[random.randint(0, len(emotion) - 1)] + ']', return_tensors = 'pt')[0][1:].to(self.device)))
                        elif response[idx][-1] != eos_token:
                            response[idx] = torch.cat((response[idx], torch.tensor([eos_token], device = self.device)))    
                        if 138 not in utt[-5: ] and 140 not in utt[-5: ]: # if emotion is not included in the sentence, randomly select a emotion type
                            prev_utterance[idx] = torch.cat((utt[:-1] if utt[-1] == eos_token else utt, self.tokenizer.encode('[' + emotion[random.randint(0, len(emotion) - 1)] + ']', return_tensors = 'pt')[0][1:].to(self.device)))
                        elif utt[-1] != eos_token:
                            prev_utterance[idx] = torch.cat((prev_utterance[idx], torch.tensor([eos_token], device = self.device)))   
                        prev_utterance[idx] = torch.cat((prev_utterance[idx], response[idx]))
                        utter_length[idx] = min(len(prev_utterance[idx]), max_len)
                        input_ids[idx][: utter_length[idx]] = prev_utterance[idx][len(prev_utterance[idx]) - utter_length[idx]:]

                    else:
                        length = len(utt[utt > 0])
                        if length == len(utt):
                            prev_utterance[idx] = torch.cat((prev_utterance[idx], torch.zeros(10, device = self.device)))
                        if 138 not in utt[length - 5:] and 140 not in utt[length - 5:]:
                            start = length - 1 if utt[length - 1] == eos_token else length
                            prev_utterance[idx][start: start + 5] = self.tokenizer.encode('[' + emotion[random.randint(0, len(emotion) - 1)] + ']', return_tensors = 'pt')[0][1:].to(self.device)
                        elif utt[length - 1] != eos_token:
                            prev_utterance[idx][length] = eos_token
                        utter_length[idx] = len(prev_utterance[idx][prev_utterance[idx] > 0])
                        input_ids[idx][: utter_length[idx]] = prev_utterance[idx][: utter_length[idx]]

        mask = input_ids.ge(0)
        input_ids[~mask] = 0
        mask = mask.float()
        prev_utterance = input_ids
        
        mask = mask.unsqueeze(1).expand((-1, beam, -1))
        mask = mask.contiguous().view(-1, max_len * 2) # [batch_size * beam, max_len]
        mask = mask.to(self.device)
        input_ids = prev_utterance.unsqueeze(1).expand((-1, beam, -1))
        input_ids = input_ids.contiguous().view((batch_size * beam, max_len * 2)) # [batch_size * beam, max_len]
        input_ids = input_ids.to(self.device)
        
        '''
            prev_utterance = [batch_size, max_len * 2]
            input_ids = [batch_size * beam, max_len * 2]
            mask = [batch_size * beam, max_len * 2] 
        '''
        if beam_search:
            utter_length = utter_length.unsqueeze(-1).expand((-1, beam)).contiguous().view(-1).to(self.device)
            original_utter_length = utter_length.detach().clone()
            beams_score = torch.zeros((batch_size, beam))
            beams_score[:, 1:] = -1e9
            beams_score = beams_score.view(-1) # [batch_size * beam]
            beams_score = beams_score.to(self.device)
            done = [False for _ in range(batch_size)]
            hyps = [BeamHypotheses(num_beams = beam, max_length = max_len, length_penalty = length_penalty_for_hypothesis) for _ in range(self.vocab_size)]
            for cur_len in range(max_len):
                # print(input_ids[3], mask[3])
                # global a, b
                # if cur_len == max_len + 1:
                #     a = input_ids
                #     b = mask
                #     return
                out = torch.log(F.softmax(self.gpt(input_ids = input_ids, attention_mask = mask)['logits'].gather(index = (utter_length - 1).unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.vocab_size)), dim = 1), dim = -1))
                # print(out.argmax(-1))
                out = out.contiguous().view((batch_size, -1)) # [batch_size, beam * vocab_size]
                beams_score_next = beams_score.unsqueeze(-1).expand((-1, self.vocab_size)).contiguous().view(batch_size, -1) + out
                next_scores, next_tokens = beams_score_next.topk(beam * 2, dim = -1)
                # print(next_tokens)
                next_beams = [] 
                for batch in range(batch_size):
                    next_beams_batch = []
                    batch_is_done = True
                    for score, token in zip(next_scores[batch], next_tokens[batch]):
                        if len(next_beams_batch) >= beam:
                            break
                        beam_index = token // self.vocab_size
                        real_token = token % self.vocab_size
                        beam_index_for_input_ids = batch * beam + beam_index
                        # print(score / ((cur_len) ** length_penalty_for_hypothesis), hyps[batch].worst_score, beam_index_for_input_ids)
                        if (real_token == eos_token or cur_len == max_len - 1 or hyps[batch].worst_score is None or \
                        score / ((cur_len) ** length_penalty_for_hypothesis) > hyps[batch].worst_score or len(hyps[batch].beams) < beam) and cur_len >= 5:
                            
                            mask_beam = mask[beam_index_for_input_ids].detach().clone()
                            input_id_beam = input_ids[beam_index_for_input_ids].detach().clone()
                            mask_beam[utter_length[beam_index_for_input_ids]] = 1
                            input_id_beam[utter_length[beam_index_for_input_ids]] = real_token

                            hyps[batch].add(hyp = input_id_beam, mask = mask_beam, sum_logprobs = score, 
                                            cur_len = cur_len, utter_length = utter_length[beam_index_for_input_ids] + 1, 
                                            prev_utterance = input_id_beam[:original_utter_length[beam_index_for_input_ids]].detach().clone(),
                                            response = input_id_beam[original_utter_length[beam_index_for_input_ids]: utter_length[beam_index_for_input_ids] + 1].detach().clone(),
                                            eos_token = eos_token)
                            if real_token != eos_token:
                                next_beams_batch.append((score, real_token, beam_index_for_input_ids))
                        else:
                            next_beams_batch.append((score, real_token, beam_index_for_input_ids))
                    if batch_is_done:
                        batch_is_done = hyps[batch].is_done(score, cur_len)
                    next_beams.extend(next_beams_batch)
                    if cur_len >= 20:
                        done[batch] = batch_is_done if not done[batch] else True

                beams_score = beams_score.new([x[0] for x in next_beams])
                beam_token = input_ids.new([x[1] for x in next_beams])
                beam_idx = input_ids.new([x[2] for x in next_beams])
                utter_length = utter_length[beam_idx] + 1
                original_utter_length = original_utter_length[beam_idx]
                input_ids = input_ids[beam_idx, :]
                mask = mask[beam_idx, :]
                with torch.no_grad():
                    input_ids = input_ids.scatter(dim = 1, index = utter_length.unsqueeze(-1) - 1, src = beam_token.unsqueeze(-1).expand((-1, 2 * max_len)))
                    mask = mask.scatter(dim = 1, index = utter_length.unsqueeze(-1) - 1, src = torch.zeros((batch_size * beam, 2 * max_len), device = self.device) + 1)
                if all(done):
                    break
            results = []
            scores = []
            masks = []
            utter_len = []
            sum_logprobs = []
            result_prev_utterance = []
            result_response = []
            for batch in range(batch_size):
                # (score, hyp, mask, utter_length, sum_logprobs)
                results.append([])
                scores.append([])
                masks.append([])
                utter_len.append([])
                sum_logprobs.append([])
                result_prev_utterance.append([])
                result_response.append([])
                for x in hyps[batch].beams: 
                    result_prev_utterance[-1].append(x[5])
                    result_response[-1].append(x[6])     
                    sum_logprobs[-1].append(x[4])
                    utter_len[-1].append(x[3])
                    masks[-1].append(x[2])
                    results[-1].append(x[1])
                    scores[-1].append(x[0])
                results[-1] = torch.stack(results[-1])
                scores[-1] = torch.stack(scores[-1])
                masks[-1] = torch.stack(masks[-1])
                utter_len[-1] = torch.stack(utter_len[-1])
                sum_logprobs[-1] = torch.stack(sum_logprobs[-1])
            results = torch.stack(results)
            masks = torch.stack(masks)
            scores = torch.stack(scores)
            utter_len = torch.stack(utter_len)
            sum_logprobs = torch.stack(sum_logprobs)
        else:
            logits_temp = self.gpt(input_ids = prev_utterance)['logits']
            logits_temp[:, :, 5:100] = -100000000
            begin = F.softmax(logits_temp.gather(index = (utter_length - 1).unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.vocab_size)), dim = 1), dim = -1)
            utter_length = utter_length.unsqueeze(-1).expand((-1, beam)).contiguous().view(-1).to(self.device)
            original_utter_length = utter_length.detach().clone()
            value, indices = begin.topk(beam, dim = -1)
            input_ids = input_ids.scatter(dim = 1, index = utter_length.unsqueeze(-1), src = indices.contiguous().view(-1).unsqueeze(-1))
            mask = mask.scatter(dim = 1, index = utter_length.unsqueeze(-1), src = torch.zeros((len(mask), 1), device = self.device) + 1)
            utter_length += 1
            # print(input_ids[0], mask[0])
            not_done = utter_length > 0
            for curlen in range(max_len - 2):
                logits_temp = self.gpt(input_ids = input_ids)['logits']
                logits_temp[:, :, 5:100] = -100000000
                generated = logits_temp.gather(index = (utter_length - 1).unsqueeze(-1).unsqueeze(-1).expand((-1, -1, self.vocab_size)), dim = 1)
                generated = F.softmax(generated, dim = -1).squeeze(1)
                values, indices = generated.topk(beam, dim = -1)
                next_tokens = indices.gather(dim = -1, index = torch.multinomial(torch.nan_to_num(torch.clamp(values, 0, 1), nan = 0.5), 1)).view(-1)
                current_done = (next_tokens == seqerate_sequence[0])
                for token in seqerate_sequence[1:]:
                    current_done = torch.logical_or(current_done, next_tokens == token)
                current_done = torch.logical_or(current_done, next_tokens == input_ids.gather(dim = -1, index = (utter_length - 1).unsqueeze(-1)).view(-1))
                next_tokens[current_done] = eos_token
                input_ids[not_done] = input_ids[not_done].scatter(dim = -1, index = (utter_length[not_done]).unsqueeze(-1), src = next_tokens[not_done].unsqueeze(-1))
                mask[not_done] = mask[not_done].scatter(dim = -1, index = (utter_length[not_done]).unsqueeze(-1), src = torch.zeros((len(mask)), device = self.device)[not_done].unsqueeze(-1) + 1)
                utter_length[not_done] += 1
                not_done = torch.logical_and(not_done, ~current_done)
                if torch.all(~not_done):
                    break
            if not torch.all(~not_done):
                input_ids[not_done] = input_ids[not_done].scatter(dim = -1, index = (utter_length[not_done]).unsqueeze(-1), src = torch.zeros((len(mask)), device = self.device, dtype = input_ids.dtype)[not_done].unsqueeze(-1) + eos_token)
                mask[not_done] = mask[not_done].scatter(dim = -1, index = (utter_length[not_done]).unsqueeze(-1), src = torch.zeros((len(mask)), device = self.device, dtype = mask.dtype)[not_done].unsqueeze(-1) + 1)
                utter_length[not_done] += 1
            result_prev_utterance = [utt[:original_utter_length[idx]] for idx, utt in enumerate(input_ids)]
            rslt_prev_utt_reshaped = []
            for reshape_prev_utt_id in range(batch_size):
                rslt_prev_utt_reshaped.append([])
                for beam_id in range(beam):
                    rslt_prev_utt_reshaped[-1].append(result_prev_utterance[beam * reshape_prev_utt_id + beam_id])
                    
            result_prev_response = [utt[original_utter_length[idx]: utter_length[idx]] for idx, utt in enumerate(input_ids)]
            rslt_prev_res_reshaped = []
            for reshape_prev_res_id in range(batch_size):
                rslt_prev_res_reshaped.append([])
                for beam_id in range(beam):
                    rslt_prev_res_reshaped[-1].append(result_prev_response[beam * reshape_prev_res_id + beam_id])
            return input_ids.view(batch_size, beam, -1), mask.view(batch_size, beam, -1), utter_length.view(batch_size, beam), rslt_prev_utt_reshaped, rslt_prev_res_reshaped 
        return results, masks, utter_len, result_prev_utterance, result_response