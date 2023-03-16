This is the PyTorch implementation for "Empathetic Response Generation via Regularized Q-Learning".

<!-- This study presents  -->

## Envionment Setup
- Install the required packaged
```bash
pip install -r requirements.txt
```

- [Download link](https://drive.google.com/drive/folders/1rR7IvAyH0rtE1hlRiFOYDy8NiZzHpC6j?usp=sharing) for preprocessed LCCC-based dataset (put in the `dataset` file).

## Run Training

```bash
python3 train.py
```

## Run Testing

```bash
python3 test.py --model_path ./runs/train/exp31/models/latest.pth --beam 5

```
- arguments:
     - `--model_path`: the path of the testing model you store
     - `--beam`: the number of candidate responses
     
 - for quick start, you can download trained model from [here](https://drive.google.com/drive/folders/1fBHqwh5pVsFoE1XMnUfgusRnt06ZzHeQ?usp=share_link), and put it in the `./runs/train` file.

## LINE Bot Setup
**Step 1.** Create a new project.

**Step 2.** Select `Create a Messaging API channel` and fill in the needed info.

**Step 3.** In `Messaging API`, `Bot basic ID` and`QR code` are 2 ways to add users on LINE.

**Step 4.** Disable the function of `Auto-reply messages` and enable the `Webhook` function.

![](https://i.imgur.com/XhnkAEK.png)

**Step 5.** Get the `Channel secret` and the `user ID` in `Basic settings` part, and the `Channel access token` in `Messaging API` part. Then paste them into the `app.py` file.

![](https://i.imgur.com/impSFsA.png)


## Ngrok Setup
Use `Ngrok` to help local sever communicate with LINE Bot platform.
- Install and setup the Ngrok. Details please refer to this [website](https://linuxhint.com/set-up-use-ngrok/).


## Run the LINE Bot application
**Step 1.** Run the command to setup Bot.
```bash
python3 app.py
```
**Step 2.** Open a new terminal window to execute the Ngrok.
```bash
ngrok http 5000
```
**Step 3.** Copy the link.

![](https://i.imgur.com/4bl0Zv8.png)

**Step 4.** Paste the copied link at `Webhook URL` in `Messaging API` part, and attach `/callback` behind then update and verify if it is successful or not.

![](https://i.imgur.com/yrGVfrg.png)

**Step 5.** Beging to use the LINE Bot.

![](https://i.imgur.com/spM9bAI.png)

