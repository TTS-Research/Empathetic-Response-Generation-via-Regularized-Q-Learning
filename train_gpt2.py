from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizerFast, AutoModel, GPT2Tokenizer, BertTokenizer, BertModel, BertLMHeadModel, AdamW, get_linear_schedule_with_warmup
import json
from torch import nn
import torch
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import seaborn as sn
import random
from typing import Iterable
import numpy as np
from tqdm import tqdm
import csv
import subprocess
import argparse
from models import *
from Data_Reward import *
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training GPT', add_help = False)
    parser.add_argument('--lr', default = 1e-4, type = float)
    parser.add_argument('--batch_size', default = 32, type=int)
    parser.add_argument('--weight_decay', default = 1e-4, type=float)
    parser.add_argument('--epochs', default = 32, type = int)
    parser.add_argument('--pretrained_gpt', default = os.path.join('.', 'GPT-2/GPT2_finetune_4'))
    # dataset parameters
    parser.add_argument('--dataset_folder', default = 'dataset')
    parser.add_argument('--root_path', default = './')
    parser.add_argument('--model_save_dir', default = './GPT-2')
    parser.add_argument('--gpu', default = 'cuda')
    parser.add_argument('--saving_name', default = 'gpt2_latest')
    return parser
def main(args):
    device = args.gpu
    if args.pretrained_gpt == 'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese':
        model = GPT2LMHeadModel.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
        tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
    else:   
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_gpt)
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    wrapper = GPT2Wrapper(gpt = model, tokenizer = tokenizer, device = device)
    dataset = GPT2DataSet(tokenizer, root_path = args.root_path,  
                 dataset_root_path = args.dataset_folder, shuffle = True, length_lower_bound = 30, length_upper_bound = 500)
    print(torch.cuda.is_available())
    print('sampling', len(dataset), dataset[11])
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    model.to(device)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    WARMUP_STEPS = 5000
    MAX_SEQ_LEN = 500
    model.train()
    optimizer = AdamW(model.parameters(), lr = LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP_STEPS, num_training_steps = -1)
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    data_loader = DataLoader(dataset, batch_size = 1, shuffle = True)
    tmp_datas_tens = None

    for epoch in range(EPOCHS):
        
        print(f"EPOCH {epoch} started" + '=' * 30)
        
        for idx, data in tqdm(enumerate(data_loader)):
            
            #################### "Fit as many data sequences into MAX_SEQ_LEN sequence as possible" logic start ####
            data_tens = data.to(device)
            #Skip sample from dataset if it is longer than MAX_SEQ_LEN
            # if data_tens.size()[1] > 256:
            #     continue
            
            #The first data sequence in the sequence
            if not torch.is_tensor(tmp_datas_tens):
                tmp_datas_tens = data_tens
                continue
            else:
                #The next data does not fit in so we process the sequence and leave the last data 
                #as the start for next sequence 
                if tmp_datas_tens.size()[1] + data_tens.size()[1] > 1: # MAX_SEQ_LEN
                    work_datas_tens = tmp_datas_tens
                    tmp_datas_tens = data_tens
                else:
                    #Add the data to sequence, continue and try to add more
                    tmp_datas_tens = torch.cat([tmp_datas_tens, data_tens[:,1:]], dim=1)
                    continue
            ################## Sequence ready, process it trough the model ##################
            counter = 0
            while counter < work_datas_tens.shape[1] and work_datas_tens[0][counter] != 140:
                counter += 1
            if work_datas_tens.shape[1] - counter < 10:
                continue
            work_datas_tens = torch.cat((work_datas_tens[0][:counter + 1], work_datas_tens.new([102]), work_datas_tens[0][counter + 1:])).unsqueeze(0)
            labels = work_datas_tens.clone().detach()
            # labels[0][:counter + 2] = -100
            
            
            outputs = model(work_datas_tens, labels = labels)
            loss, logits = outputs[:2]                        
            loss.backward()
            sum_loss = sum_loss + loss.detach().data
                        
            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == BATCH_SIZE:
                proc_seq_count = 0    
                batch_count += 1
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                wrapper.update_gpt(model)
                results, masks, utter_len, result_prev_utterance, result_response = wrapper(prev_utterance = [work_datas_tens.new([101, 3209, 1921, 6206, 1139, 1343, 4381, 8024, 5646, 1956, 1765, 4717, 679, 5865, 6221, 138, 7274, 2552, 140, 102]
    ), work_datas_tens.new([101, 2769, 947, 3209, 1921, 671, 6629, 1139, 1343, 4381, 1962, 679, 1962, 102])], require_grad = False, device = device, max_len = MAX_SEQ_LEN, seqerate_sequence = [102, 101, 138, 0, 140, 103, 511, 8013, 8043, 8080, 8049, 510])
                model.save_pretrained(os.path.join(args.model_save_dir, args.saving_name))
                for sentence in results:
                    for sen in sentence:
                        print(tokenizer.decode(sen, skip_special_tokens = True))
                print(f"sum loss {sum_loss}")
                print(work_datas_tens, labels)
                batch_count = 0
                sum_loss = 0.0
        
        # Store the model after each epoch to compare the performance of them
        model.save_pretrained(os.path.join(args.model_save_dir, f"gpt2_epoch_{epoch}"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPT training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)