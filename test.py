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
import argparse
from models import *
from Data_Reward import *
import time
from torcheval.metrics.text import Perplexity
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for testing GPT by RL', add_help = False)
    parser.add_argument('--root_path', default = os.getcwd())
    parser.add_argument('--model_path', default = "./train/runs/exp16/models/latest.pth")
    parser.add_argument('--gpu', default = 'cuda')
    parser.add_argument('--beam', default = 3, type = int)
    parser.add_argument('--pretrained_gpt', default = os.path.join('.', 'GPT-2/gpt2_latest'))
    parser.add_argument('--dataset_folder', default = 'dataset')
    parser.add_argument('--loading_mode', default = 0, help = '0 means loading from checkpoint completely, 1 means only loadingQ models,\
                        2 means not loading from checkpont, 3 means only loading GPT from checkpoint', type = int)
    parser.add_argument('--max_len', default = 100)
    return parser
def conversation(input_string, gpt_wrapper, Q_A, Q_B, gpt_tokenizer, max_len = 256, device = 'cpu', beam = 3, emotion = ['喜歡', '悲傷', '噁心', '憤怒', '開心', '其它']):
    if input_string[-1] != ']':
        input_string += f"[{emotion[random.randint(0, 5)]}]"
    input_ids = gpt_tokenizer.encode(input_string, return_tensors = 'pt')
    rslt, msk, utrlen, rsltprvrnce, rsltrspse = gpt_wrapper(prev_utterance = [input_ids[0].to(device)], max_len = max_len, require_grad = False, 
                                                            device = device, beam = beam, seqerate_sequence = [102, 101, 138, 0, 140, 103, 511, 8013, 8043, 8080, 8049, 510]) # , 8024
    scores = (Q_A(prev_utterance = rslt[0]) + Q_B(prev_utterance = rslt[0])).exp()
    select = int(torch.multinomial(scores / sum(scores), 1))
    return rsltrspse[0], select, rslt[0][select][rslt[0][select] > 0], scores
def perplexity(test_set, gpt_wrapper, device = 'cpu', max_len = 200):
    prev_utterance = []
    response = []
    metric = 0
    for idx, sentence in tqdm(enumerate(test_set), desc = 'Calculating Perplexity'):
        counter = 0
        lower_bracket =  gpt_wrapper.tokenizer.encode(']')[1]
        while sentence[counter] != lower_bracket:
            counter += 1
        prev_utterance.append(torch.cat((sentence[: (counter)], sentence.new(gpt_wrapper.tokenizer.encode(']')[1:]))).to(device))
        response.append(sentence[counter + 1: ].to(device))
        logits = F.softmax(gpt_wrapper.gpt(input_ids = torch.cat((prev_utterance[-1], response[-1])))['logits'][len(prev_utterance[-1]) - 1:-1, :], dim = -1).unsqueeze(1)
        
        c, t = logits, response[-1].unsqueeze(-1)
        temp = 0
        for i in range(len(c)):
            temp -= torch.log(c[i, 0, t[i][0]])
        metric += temp / len(c)
    return metric / len(test_set)
def bleu_score(candidates, targets, n_gram = 4):
    total = 0
    offset = 0
    for idx_i, candidate in enumerate(candidates):
        if len(candidate) < n_gram:
            offset += 1
            continue
        length = len(candidate) - n_gram + 1
        counter = 0
        for idx in range(length):
            fined = False
            for idx_t in range(length):
                for target in targets[idx_i]:
                    if len(target) < n_gram:
                        continue
                    if target[idx_t: idx_t + n_gram] == candidate[idx: idx + n_gram]:
                        counter += 1
                        fined = True
                        break
                if fined:
                    break
        total += counter / length
    return total / (len(candidates) - offset) if (len(candidates) - offset) > 0 else 1
def bleu_accuracy(test_set, gpt_wrapper, device = 'cpu', max_len = 100):
    candidates = []
    prev_utterance = []
    response = []
    targets = []
    candidates_bleu = []
    targets_bleu = []
    for idx, sentence in tqdm(enumerate(test_set), desc = 'Calculating BLEU'):
        counter = 0
        lower_bracket =  gpt_wrapper.tokenizer.encode(']')[1]
        while sentence[counter] != lower_bracket:
            counter += 1
        prev_utterance.append(torch.cat((sentence[: (counter)], sentence.new(gpt_wrapper.tokenizer.encode(']')[1:]))).to(device))
        response.append([sentence[counter + 1: ].to(device)])
        results, masks, utter_len, result_prev_utterance, result_response = gpt_wrapper(prev_utterance = [prev_utterance[-1]], max_len = max_len,
                                                                                            require_grad = False, device = device, beam_search = False, beam = 2, seqerate_sequence = [102, 101, 138, 0, 140, 103, 511, 8013, 8043, 8080, 8049, 510])
        candidates.append(gpt_wrapper.tokenizer.decode(response[-1][0], skip_special_tokens = True).replace(" ", ""))
        targets.append([gpt_wrapper.tokenizer.decode(res, skip_special_tokens = True).replace(" ", "") for res in result_response[0]])
        candidates_bleu.append(gpt_wrapper.tokenizer.decode(result_response[0][0], skip_special_tokens = True).replace(" ", ""))
        targets_bleu.append([gpt_wrapper.tokenizer.decode(response[-1][0], skip_special_tokens = True).replace(" ", "")])
    accuracy = 0
    for idx, (candidate, target) in tqdm(enumerate(zip(candidates, targets))):
        processed_length = min(len(candidate), len(target[0]), len(target[1]))
        counter = 0
        for index in range(processed_length):
            if candidate[index] == target[0][index] or candidate[index] == target[1][index]:
                counter += 1
        counter /= processed_length
        accuracy += counter
    accuracy /= len(candidates)
    return bleu_score(candidates, targets, n_gram = 2), bleu_score(candidates, targets, n_gram = 4), accuracy, candidates, targets
@torch.no_grad()
def main(args, emotion = ['喜歡', '悲傷', '噁心', '憤怒', '開心', '其它']):
    device = args.gpu
    beam = args.beam
    pwd = args.root_path
    gpt2 = GPT2LMHeadModel.from_pretrained(args.pretrained_gpt)
    gpt2_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    gpt_wrapper = GPT2Wrapper(gpt2, tokenizer = gpt2_tokenizer, device = device)
    Q_A = Q(gpt_tokenizer = gpt2_tokenizer, bert_name = 'bert-base-chinese', device = device)
    Q_B = Q(gpt_tokenizer = gpt2_tokenizer, bert_name = 'bert-base-chinese', device = device)
    test_set = GPT2DataSet(tokenizer = gpt2_tokenizer, length_upper_bound = 70, length_lower_bound = 60, status = 'test', dataset_root_path = args.dataset_folder, root_path = args.root_path)
    bleu2, bleu4, accuracy, candidates, targets = bleu_accuracy(test_set, gpt_wrapper, device = device, max_len = args.max_len)
    perplexity_score = perplexity(test_set, gpt_wrapper, device = device, max_len = 40)
    print(f"BLEU2: {bleu2} BLEU4: {bleu4} Acc: {accuracy} Perplexity: {perplexity_score} ")
    if os.path.exists(args.model_path) and args.loading_mode != 2:
        ckpt = torch.load(args.model_path)
        if args.loading_mode != 1:
            gpt_wrapper.load_state_dict(ckpt['GPT'])
        if args.loading_mode != 3:
            Q_A.load_state_dict(ckpt['Q_A'])
            Q_B.load_state_dict(ckpt['Q_B'])
    # else:
    #     print("model file doesn't exist")
    input_string = input("Enter String To Converse, Enter quit to quit : ")
    prev_utterance = ""
    while input_string != "quit":
        print("inputed string :", input_string)
        prev_utterance += input_string
        prev_utterance = input_string
        responses, selected_index, prev_utterance, scores = conversation(prev_utterance, gpt_wrapper, Q_A, Q_B, gpt2_tokenizer, beam = beam, emotion = emotion, device = device, max_len = args.max_len)
        print("Candidate Sentences")
        for i in range(beam):
            print(gpt2_tokenizer.decode(responses[i], skip_special_tokens = True).replace(" ", ""), float(scores[i]))
        print(f"Selected index : {selected_index}\nSentence : {gpt2_tokenizer.decode(responses[selected_index], skip_special_tokens = True).replace(' ', '')}")
        prev_utterance = gpt2_tokenizer.decode(prev_utterance, skip_special_tokens = True).replace(" ", "")
        if prev_utterance[-1] != ']':
            prev_utterance += f"[{emotion[random.randint(0, 5)]}]"
        input_string = input("Enter String To Converse, Enter quit to quit : ")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPT testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)