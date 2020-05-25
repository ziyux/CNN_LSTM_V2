#!/bin/bash
#Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
GIT_REPO_URL="https://github.com/ziyux/CNN_LSTM_V2"
REPO="CNN_LSTM_V2"
VIDEO="121"
UIN_JSON="627008251.json"
UIN_JPG="627008251.jpg"
git clone $GIT_REPO_URL
cd $REPO
#Replace this line with commands for running your test python file.
echo $VIDEO
python test.py --video_name $VIDEO
#If your test file is ipython file, uncomment the following lines and
#replace IPYTHON_NAME with your test ipython file.
#IPYTHON_NAME="test.ipynb"
#echo $IPYTHON_NAME
#jupyter notebook
#rename the generated timeLabel.json and figure with your UIN.
cp timeLable.json $UIN_JSON
cp timeLable.jpg $UIN_JPG