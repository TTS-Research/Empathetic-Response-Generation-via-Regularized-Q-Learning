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


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set parameters for training GPT by RL', add_help=False)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=32, type=int)
    parser.add_argument('--lr_drop', default=3, type=int)
    parser.add_argument('--eval_freq', default=3, type=int)

    # * Loss coefficients
    parser.add_argument('--Q_discounter_factor', default=0.9, type=float)

    parser.add_argument('--kl_control', default=0, type=float,
                        help="coefficent of KL loss")
    parser.add_argument('--gpt_loss_coefficient', default=0.01, type=float,
                        help="coefficent of GPT loss")

    # dataset parameters
    parser.add_argument('--dataset_folder', default='dataset')
    parser.add_argument('--root_path', default=os.getcwd())
    parser.add_argument('--gpu', default='cuda')
    parser.add_argument('--len_lower_bound', default=50, type=int)
    parser.add_argument('--len_upper_bound', default=80, type=int)
    parser.add_argument(
        '--checkpoint', default='./runs/train/exp20/models/latest.pth')
    parser.add_argument('--pretrained_gpt',
                        default=os.path.join('.', 'GPT-2/gpt2_latest'))
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--length_word_weight', default=0.1)
    parser.add_argument('--question_reward_weight', default=0.1)
    parser.add_argument('--coherence_weight', default=0.1)
    parser.add_argument('--toxicity_weight', default=0.1)
    parser.add_argument('--ease_of_answering_weight', default=1)
    parser.add_argument('--get_reward_semantic_coherence_weight', default=0.1)
    parser.add_argument('--loading_mode', default=0, help='0 means loading from checkpoint completely, 1 means only loading Q models,\
                        2 means not loading from checkpont, 3 means only loading GPT from checkpoint', type=int)
    return parser


def get_output_dir(pwd):
    runs_dir = os.path.join(pwd, 'runs')
    train_dir = os.path.join(runs_dir, 'train')
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    counter = 1
    exp = f"exp{counter}"
    while exp in os.listdir(train_dir):
        counter += 1
        exp = f"exp{counter}"
    output_dir = os.path.join(train_dir, exp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'models')):
        os.mkdir(os.path.join(output_dir, 'models'))
    with open(os.path.join(output_dir, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'KLLoss', 'GPTLoss', 'QLoss', 'TOTALLoss'])
    return output_dir


def main(args):
    print(args)
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    lr_drop = args.lr_drop
    Q_discounter_factor = args.Q_discounter_factor
    kl_control = args.kl_control
    gpt_loss_coefficient = args.gpt_loss_coefficient
    pwd = args.root_path
    data_root = args.dataset_folder
    output_dir = get_output_dir(pwd)
    device = args.gpu
    gpt2 = GPT2LMHeadModel.from_pretrained(args.pretrained_gpt)
    gpt2_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    gpt_wrapper = GPT2Wrapper(gpt2, tokenizer=gpt2_tokenizer, device=device)
    Q_A = Q(gpt_tokenizer=gpt2_tokenizer, gamma=Q_discounter_factor,
            bert_name='bert-base-chinese', device=device)
    Q_B = Q(gpt_tokenizer=gpt2_tokenizer, gamma=Q_discounter_factor,
            bert_name='bert-base-chinese', device=device)
    if not args.checkpoint is None and args.loading_mode != 2:
        checkpoint = torch.load(args.checkpoint)
        if args.loading_mode != 1:
            gpt_wrapper.load_state_dict(checkpoint['GPT'])
        if args.loading_mode != 3:
            Q_A.load_state_dict(checkpoint['Q_A'])
            Q_B.load_state_dict(checkpoint['Q_B'])
    toxic_words, non_sense_response = GPT2DataSet.get_toxic_ids_and_non_sense_response(
        gpt2_tokenizer)
    R = Reward(gpt=gpt2, question_mark_token=136, toxic_words=toxic_words, gpt_tokenizer=gpt2_tokenizer,
               non_sense_response=non_sense_response, eos_token=102, device=device, bos_token=101,
               length_word_weight=args.length_word_weight, question_reward_weight=args.question_reward_weight,
               coherence_weight=args.coherence_weight, toxicity_weight=args.toxicity_weight,
               ease_of_answering_weight=args.ease_of_answering_weight, get_reward_semantic_coherence_weight=args.get_reward_semantic_coherence_weight)
    criterion = nn.MSELoss()
    gpt_wrapper.to(device)
    Q_A.to(device)
    Q_B.to(device)
    optimizer = torch.optim.Adam([{'params': [p for p in gpt_wrapper.parameters() if p.requires_grad]},
                                  {'params': [
                                      p for p in Q_A.parameters() if p.requires_grad]},
                                  {'params': [
                                      p for p in Q_B.parameters() if p.requires_grad]},
                                  ], lr=lr, weight_decay=weight_decay)
    print('total parameters: ', sum([p.numel() for p in gpt_wrapper.parameters() if p.requires_grad]) +
          sum([p.numel() for p in Q_A.parameters() if p.requires_grad]) +
          sum([p.numel() for p in Q_B.parameters() if p.requires_grad]))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)
    train_dataset = GPT2DataSet(tokenizer=gpt2_tokenizer, max_len=args.max_len, root_path=pwd, status='train', dataset_root_path=data_root,
                                length_lower_bound=args.len_lower_bound, length_upper_bound=args.len_upper_bound)
    test_dataset = GPT2DataSet(tokenizer=gpt2_tokenizer, max_len=args.max_len,
                               root_path=pwd, status='test', dataset_root_path=data_root)
    q_losses = []
    gpt_losses = []
    kl_losses = []
    total_losses = []
    print('Start training')
    for epoch in range(epochs):
        if True:
            # try:
            q_loss, kl_loss, gpt_loss, total_loss = train_one_epoch(epoch=epoch, gpt=gpt_wrapper, Q_A=Q_A, Q_B=Q_B,
                                                                    optimizer=optimizer, R=R, dataset=train_dataset, device=device, batch_size=batch_size,
                                                                    max_len=args.max_len, beam=2, update_time_per_episode=1, criterion=criterion,
                                                                    kl_control=kl_control, gpt_loss_coefficient=gpt_loss_coefficient, output_dir=output_dir)
            q_losses.append(float(q_loss))
            kl_losses.append(float(kl_loss))
            gpt_losses.append(float(gpt_loss))
            total_losses.append(float(total_loss))
            torch.save({
                'Q_A': Q_A.state_dict(),
                'Q_B': Q_B.state_dict(),
                'GPT': gpt_wrapper.state_dict()
            }, os.path.join(output_dir, 'models/latest.pth'))
            if min(q_losses) == q_loss:
                time.sleep(0.1)
                torch.save({
                    'Q_A': Q_A.state_dict(),
                    'Q_B': Q_B.state_dict(),
                    'GPT': gpt_wrapper.state_dict()
                }, os.path.join(output_dir, 'models/best_q_loss.pth'))
            if min(kl_losses) == kl_loss:
                time.sleep(0.1)
                torch.save({
                    'Q_A': Q_A.state_dict(),
                    'Q_B': Q_B.state_dict(),
                    'GPT': gpt_wrapper.state_dict()
                }, os.path.join(output_dir, 'models/best_kl_loss.pth'))
            if min(gpt_losses) == gpt_loss:
                time.sleep(0.1)
                torch.save({
                    'Q_A': Q_A.state_dict(),
                    'Q_B': Q_B.state_dict(),
                    'GPT': gpt_wrapper.state_dict()
                }, os.path.join(output_dir, 'models/best_gpt_loss.pth'))
            if min(total_losses) == total_loss:
                time.sleep(0.1)
                torch.save({
                    'Q_A': Q_A.state_dict(),
                    'Q_B': Q_B.state_dict(),
                    'GPT': gpt_wrapper.state_dict()
                }, os.path.join(output_dir, 'models/best_total_loss.pth'))
            with open(os.path.join(output_dir, 'log.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, kl_loss, gpt_loss, q_loss, total_loss])
            if epoch % args.eval_freq == 0:
                q_loss, kl_loss, gpt_loss, total_loss = evaluate(gpt=gpt_wrapper, Q_A=Q_A, Q_B=Q_B, R=R, dataset=test_dataset, device=device, batch_size=batch_size,
                                                                 max_len=args.max_len, beam=2, criterion=criterion,
                                                                 kl_control=kl_control, gpt_loss_coefficient=gpt_loss_coefficient)

                # print the evaluation results
                print(
                    '=======================================test=======================================')
                print(
                    f"QLoss : {q_loss} KLLoss {kl_loss}: GPTLoss : {gpt_loss} TotalLoss : {total_loss}")
                print(
                    '=======================================test=======================================')

            R.update_model(gpt_wrapper.gpt)
        # except Exception as e:
        #     print(e)
        #     continue


def train_one_epoch(epoch: int, gpt: GPT2Wrapper, Q_A: Q, Q_B: Q,  optimizer: torch.optim.Optimizer = None, R: Reward = None,
                    dataset: Iterable = None, device: torch.device = 'cpu', batch_size=16, max_len=100, beam=3, update_time_per_episode=10, criterion=nn.MSELoss(), kl_control=0,
                    gpt_loss_coefficient=0.1, output_dir='./'):
    gpt.train()
    # two Qs for Double DQN
    Q_A.train()
    Q_B.train()
    gpt.to(device)
    Q_A.to(device)
    Q_B.to(device)
    kl_losses = []
    gpt_losses = []
    q_losses = []
    total_losses = []

    dataset.shuffle()
    with tqdm(total=len(dataset) // batch_size) as t:
        for step in range(len(dataset) // batch_size):
            t.set_description(f"Epoch {epoch}")
            if True:
                prev_utterance = []
                response = []
                for mini_step in range(step * batch_size, (step + 1) * batch_size):
                    pair = dataset[mini_step]
                    counter = 0
                    while pair[counter] != gpt.tokenizer.encode(']')[1]:
                        counter += 1
                    prev_utterance.append(torch.cat(
                        (pair[: (counter)], pair.new(gpt.tokenizer.encode(']')[1:]))).to(device))
                    response.append(pair[counter + 1:].to(device))
                utter_length = None
                generate_time = 0
                with torch.no_grad():  # generate episode

                    for iterate in range(3):
                        generate_time += 1
                        if utter_length is None:
                            rslt, msk, utrlen, rsltprvrnce, rsltrspse = gpt(prev_utterance=prev_utterance, response=response, beam=beam, max_len=max_len,
                                                                            device=device)
                            # soft sampling
                            previous_Q_distribution = F.softmax((Q_A(prev_utterance=rslt.view(batch_size * beam, -1), response=None, mask=None, bert_tokens=False, max_len=max_len * 2, processed=False) +
                                                                Q_B(prev_utterance=rslt.view(batch_size * beam, -1), response=None, mask=None, bert_tokens=False, max_len=max_len * 2, processed=False)).view(batch_size, -1), dim=-1)
                            select = torch.multinomial(torch.nan_to_num(
                                torch.clamp(previous_Q_distribution, 0, 1), 0.5), 1)
                            previous_result = rslt.gather(
                                index=select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim=1).squeeze(1)
                            results = previous_result.detach().clone()
                            masks = msk.gather(
                                index=select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim=1).squeeze(1)
                            utter_length = utrlen.gather(
                                index=select, dim=1).squeeze(1)
                            results_prev_utterance = [
                                rsltprvrnce[i][int(select[i])] for i in range(batch_size)]
                            results_response = [
                                rsltrspse[i][int(select[i])] for i in range(batch_size)]
                        else:
                            rslt, msk, utrlen, rsltprvrnce, rsltrspse = gpt(prev_utterance=previous_result, beam=beam, max_len=max_len,
                                                                            device=device)

                            previous_Q_distribution = F.softmax((Q_A(prev_utterance=rslt.view(batch_size * beam, -1), response=None, mask=None, bert_tokens=False, max_len=max_len * 2, processed=False) +
                                                                 Q_B(prev_utterance=rslt.view(batch_size * beam, -1), response=None, mask=None, bert_tokens=False, max_len=max_len * 2, processed=False)).view(batch_size, -1), dim=-1)
                            select = torch.multinomial(torch.nan_to_num(
                                torch.clamp(previous_Q_distribution, 0, 1), 0.5), 1)
                            previous_result = rslt.gather(
                                index=select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim=1).squeeze(1)

                            results = torch.cat(
                                (results, previous_result.detach().clone()), dim=0)
                            masks = torch.cat((masks, msk.gather(
                                index=select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim=1).squeeze(1)), dim=0)
                            utter_length = torch.cat(
                                (utter_length, utrlen.gather(index=select, dim=1).squeeze(1)), dim=0)
                            results_prev_utterance.extend(
                                [rsltprvrnce[i][int(select[i])] for i in range(batch_size)])
                            results_response.extend(
                                [rsltrspse[i][int(select[i])] for i in range(batch_size)])
                    results_Q, mask_Q = Q_A.get_processed(
                        prev_utterance=results, bert_tokens=False, max_len=max_len * 2)
                    reward = R([{'prev_utterance': utt, 'response': res} for utt, res in zip(
                        results_prev_utterance[: len(results_Q) - batch_size], results_response[: len(results_Q) - batch_size])])

                for update_time in range(update_time_per_episode):
                    if random.random() >= 0.5:  # update Q_A
                        q_estimate = Q_A.forward(prev_utterance=results_Q[: len(results_Q) - batch_size], response=None, mask=mask_Q[: len(
                            results_Q) - batch_size], bert_tokens=True, max_len=max_len * 2, processed=True)
                        q_target = reward + Q_A.gamma * \
                            Q_B.forward(prev_utterance=results_Q[batch_size:], response=None,
                                        mask=mask_Q[batch_size:], bert_tokens=True, max_len=max_len * 2, processed=True)
                    else:  # update Q_B
                        q_estimate = Q_B.forward(prev_utterance=results_Q[: len(results_Q) - batch_size], response=None, mask=mask_Q[: len(
                            results_Q) - batch_size], bert_tokens=True, max_len=max_len * 2, processed=True)
                        q_target = reward + Q_B.gamma * \
                            Q_A.forward(prev_utterance=results_Q[batch_size:], response=None,
                                        mask=mask_Q[batch_size:], bert_tokens=True, max_len=max_len * 2, processed=True)
                    q_loss = criterion(q_estimate, q_target)
                    kl_loss = - kl_control * \
                        torch.log(torch.mean(q_estimate.exp()))
                    probs = torch.nan_to_num(gpt.get_prob(results[: len(results_Q) - batch_size], masks[: len(results_Q) - batch_size], results_prev_utterance[: len(
                        results_Q) - batch_size], results_response[: len(results_Q) - batch_size]), torch.log(torch.tensor(0.5)))
                    gpt_loss = - \
                        torch.mean(probs.detach().clone().exp() *
                                   probs * q_target.detach().clone().exp())
                    total_loss = q_loss + kl_control * kl_loss + gpt_loss_coefficient * gpt_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    kl_losses.append(float(kl_loss.detach().clone()))
                    gpt_losses.append(float(gpt_loss.detach().clone()))
                    q_losses.append(float(q_loss.detach().clone()))
                    total_losses.append(float(total_loss.detach().clone()))
                    t.set_postfix(klloss=float(kl_losses[-1]), gpt_loss=float(gpt_losses[-1]),
                                  q_loss=float(q_losses[-1]), total_loss=float(total_losses[-1]))
                if step % 10 == 0:
                    torch.save({
                        'Q_A': Q_A.state_dict(),
                        'Q_B': Q_B.state_dict(),
                        'GPT': gpt.state_dict()
                    }, os.path.join(output_dir, 'models/latest.pth'))
                t.update(1)
    return np.mean(q_losses), np.mean(kl_losses), np.mean(gpt_losses), np.mean(total_losses)


@torch.no_grad()
def evaluate(gpt, Q_A, Q_B, R, dataset, device, batch_size=16, max_len=100, beam=3, criterion=nn.MSELoss(), kl_control=0, gpt_loss_coefficient=0.1):
    gpt.eval()
    # two Qs for Double DQN
    Q_A.eval()
    Q_B.eval()
    gpt.to(device)
    Q_A.to(device)
    Q_B.to(device)
    kl_losses = []
    gpt_losses = []
    q_losses = []
    total_losses = []

    with tqdm(total=len(dataset) // batch_size) as t:
        for step in range(len(dataset) // batch_size):
            # for step in range(2):
            t.set_description(f"Validation")

            prev_utterance = []
            response = []
            for mini_step in range(step * batch_size, (step + 1) * batch_size):
                pair = dataset[mini_step]
                counter = 0
                while pair[counter] != gpt.tokenizer.encode(']')[1]:
                    counter += 1
                prev_utterance.append(torch.cat(
                    (pair[: (counter)], pair.new(gpt.tokenizer.encode(']')[1:]))).to(device))
                response.append(pair[counter + 1:].to(device))
            utter_length = None
            generate_time = 0
            with torch.no_grad():  # generate episode
                for iterate in range(3):
                    generate_time += 1
                    if utter_length is None:
                        rslt, msk, utrlen, rsltprvrnce, rsltrspse = gpt(prev_utterance=prev_utterance, response=response, beam=beam, max_len=max_len,
                                                                        device=device)
                        # soft sampling
                        previous_Q_distribution = F.softmax((Q_A(prev_utterance=rslt.view(batch_size * beam, -1), response=None, mask=None, bert_tokens=False, max_len=max_len * 2, processed=False) +
                                                            Q_B(prev_utterance=rslt.view(batch_size * beam, -1), response=None, mask=None, bert_tokens=False, max_len=max_len * 2, processed=False)).view(batch_size, -1), dim=-1)
                        select = torch.multinomial(torch.nan_to_num(
                            torch.clamp(previous_Q_distribution, 0, 1), 0.5), 1)
                        previous_result = rslt.gather(
                            index=select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim=1).squeeze(1)
                        results = previous_result.detach().clone()
                        masks = msk.gather(
                            index=select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim=1).squeeze(1)
                        utter_length = utrlen.gather(
                            index=select, dim=1).squeeze(1)
                        results_prev_utterance = [
                            rsltprvrnce[i][int(select[i])] for i in range(batch_size)]
                        results_response = [
                            rsltrspse[i][int(select[i])] for i in range(batch_size)]
                    else:
                        rslt, msk, utrlen, rsltprvrnce, rsltrspse = gpt(prev_utterance=previous_result, beam=beam, max_len=max_len,
                                                                        device=device)

                        previous_Q_distribution = F.softmax((Q_A(prev_utterance=rslt.view(batch_size * beam, -1), response=None, mask=None, bert_tokens=False, max_len=max_len * 2, processed=False) +
                                                             Q_B(prev_utterance=rslt.view(batch_size * beam, -1), response=None, mask=None, bert_tokens=False, max_len=max_len * 2, processed=False)).view(batch_size, -1), dim=-1)
                        select = torch.multinomial(torch.nan_to_num(
                            torch.clamp(previous_Q_distribution, 0, 1), 0.5), 1)
                        previous_result = rslt.gather(
                            index=select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim=1).squeeze(1)

                        results = torch.cat(
                            (results, previous_result.detach().clone()), dim=0)
                        masks = torch.cat((masks, msk.gather(
                            index=select.unsqueeze(-1).expand(-1, -1, max_len * 2), dim=1).squeeze(1)), dim=0)
                        utter_length = torch.cat(
                            (utter_length, utrlen.gather(index=select, dim=1).squeeze(1)), dim=0)
                        results_prev_utterance.extend(
                            [rsltprvrnce[i][int(select[i])] for i in range(batch_size)])
                        results_response.extend(
                            [rsltrspse[i][int(select[i])] for i in range(batch_size)])
                results_Q, mask_Q = Q_A.get_processed(
                    prev_utterance=results, bert_tokens=False, max_len=max_len * 2)
                reward = R([{'prev_utterance': utt, 'response': res} for utt, res in zip(
                    results_prev_utterance[: len(results_Q) - batch_size], results_response[: len(results_Q) - batch_size])])

            if random.random() >= 0.5:
                q_estimate = Q_A.forward(prev_utterance=results_Q[: len(results_Q) - batch_size], response=None, mask=mask_Q[: len(
                    results_Q) - batch_size], bert_tokens=True, max_len=max_len * 2, processed=True)
                q_target = reward + Q_A.gamma * \
                    Q_B.forward(prev_utterance=results_Q[batch_size:], response=None,
                                mask=mask_Q[batch_size:], bert_tokens=True, max_len=max_len * 2, processed=True)
            else:  # update Q_B
                q_estimate = Q_B.forward(prev_utterance=results_Q[: len(results_Q) - batch_size], response=None, mask=mask_Q[: len(
                    results_Q) - batch_size], bert_tokens=True, max_len=max_len * 2, processed=True)
                q_target = reward + Q_B.gamma * \
                    Q_A.forward(prev_utterance=results_Q[batch_size:], response=None,
                                mask=mask_Q[batch_size:], bert_tokens=True, max_len=max_len * 2, processed=True)
            q_loss = criterion(q_estimate, q_target)
            kl_loss = - kl_control * torch.log(torch.mean(q_estimate.exp()))
            probs = gpt.get_prob(results[: len(results_Q) - batch_size], masks[: len(results_Q) - batch_size],
                                 results_prev_utterance[: len(results_Q) - batch_size], results_response[: len(results_Q) - batch_size])
            gpt_loss = -torch.sum(
                probs.detach().clone().exp() * probs * q_target)
            total_loss = q_loss + kl_control * kl_loss + gpt_loss_coefficient * gpt_loss

            kl_losses.append(float(kl_loss.detach().clone()))
            gpt_losses.append(float(gpt_loss.detach().clone()))
            q_losses.append(float(q_loss.detach().clone()))
            total_losses.append(float(total_loss.detach().clone()))
            t.set_postfix(klloss=float(kl_losses[-1]), gpt_loss=float(gpt_losses[-1]),
                          q_loss=float(q_losses[-1]), total_loss=float(total_losses[-1]))
            t.update(1)
    return np.mean(q_losses), np.mean(kl_losses), np.mean(gpt_losses), np.mean(total_losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'GPT training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
