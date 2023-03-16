#載入LineBot所需要的模組
from flask import Flask, request, abort
 
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import *
from cnsenti import Emotion
from opencc import OpenCC
import testApp
import argparse
import os

app = Flask(__name__)
 
# 必須放上自己的Channel Access Token
line_bot_api = LineBotApi('bRX4fZdNlwOGSL9OWIMIBLPP0VcaRZ29XQKOx+PiixbxP4PwHaDDXTGNrjdKTniF+2/3f9gcD3WEF7Hwq5/sHMhWCKFRu6Ip51QmUh4nAXBzPxD6bLQaYek8YcqTwaBYzGsC1vfo26vv4ilqOQJpogdB04t89/1O/w1cDnyilFU=')
 
# 必須放上自己的Channel Secret
handler = WebhookHandler('90e33b636a9d35c6b86c4410234fa3e1')
line_bot_api.push_message('Ub3ac847377f02cc07ff4dc238290338c', TextSendMessage(text='你可以開始了'))

#####################################################################################

# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
 
  
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
 
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
 
    return 'OK'
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for testing GPT by RL', add_help = False)
    parser.add_argument('--root_path', default = os.getcwd())
    parser.add_argument('--model_path', default = "./train/runs/exp24/models/latest.pth")
    parser.add_argument('--gpu', default = 'cuda')
    parser.add_argument('--beam', default = 5, type = int)
    parser.add_argument('--pretrained_gpt', default = os.path.join('.', 'GPT-2/gpt2_latest'))
    parser.add_argument('--dataset_folder', default = 'dataset')
    parser.add_argument('--loading_mode', default = 0, help = '0 means loading from checkpoint completely, 1 means only loadingQ models,\
                        2 means not loading from checkpont, 3 means only loading GPT from checkpoint', type = int)
    parser.add_argument('--max_len', default = 100)
    return parser

def main():
    parser = argparse.ArgumentParser('GPT testing script', parents=[get_args_parser()])
    args = parser.parse_args()

def top_n_scores(n, score_dict):
  ''' returns the n scores from a name:score dict'''
  lot = [(k,v) for k, v in score_dict.items()] #make list of tuple from scores dict
  nl = []
  while len(lot)> 0:
    nl.append(max(lot, key=lambda x: x[1]))
    lot.remove(nl[-1])
  return nl[0:n]

#訊息傳遞區塊
##### 基本上程式編輯都在這個function #####
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    #message = TextSendMessage(text=event.message.text)
    #line_bot_api.reply_message(event.reply_token,message)
    # parser = argparse.ArgumentParser('GPT testing script', parents=[get_args_parser()])
    # args = parser.parse_args()

    if event.message.text[-1] == ']' :
        event.message.text = event.message.text
    else:
        print('traditional one =', event.message.text)
        event_message_text_s = OpenCC('t2s').convert(event.message.text)
        print('simplified one = ', event_message_text_s)
        emotion = Emotion()
        result = emotion.emotion_count(event_message_text_s)
        del result['words'] 
        del result['sentences']
        print(result)
        label = top_n_scores(1, result)
        InputEmotion = label[0][0]
        if InputEmotion == '好' :
            event.message.text = event.message.text + '[喜歡]'
        elif InputEmotion == '乐' :
            event.message.text = event.message.text + '[開心]'
        elif InputEmotion == '哀' :
            event.message.text = event.message.text + '[悲傷]'
        elif InputEmotion == '怒' :
            event.message.text = event.message.text + '[憤怒]'
        elif InputEmotion == '恶' :
            event.message.text = event.message.text + '[噁心]'
        elif InputEmotion == '惧' :
            event.message.text = event.message.text + '[其它]'
        else:
            event.message.text = event.message.text + '[其它]'
    #Candidates, Final_Response = testApp.handle(args,text = event.message.text)
    Final_Response = get_response(emotion = ['喜歡', '悲傷', '噁心', '憤怒', '開心', '其它'],text = event.message.text)

    #print("Candidates: ", Candidates)
    print("Final_Response: ",Final_Response)
    line_bot_api.reply_message(event.reply_token,TextSendMessage(text=Final_Response))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPT testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    get_response = testApp.handle(args=args)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)