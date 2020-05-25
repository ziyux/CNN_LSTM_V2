# CNN_LSTM_V2

Project Description
-------------
This project aims to design a deep learning model for rolling action recognition. The rolling action includes forward roll, backward roll, shoulder roll and dive roll. But side roll (pencil roll) is not included.

The dataset is collected from Youtube, the detailed information is included in the file /src/rolling.csv

The project uses Openpose from CMU (https://github.com/CMU-Perceptual-Computing-Lab/openpose) to implement feature extraction and obtain 25 keypoints of body as input features, which includes: 

Nose, Neck, Right Shoulder, Right Elbow, Right Wrist, Left Shoulder, Left Elbow, Left wrist, Mid Hip, Right Hip, Right Knee, Right Ankle, Left Hip, Left knee, Left Ankle, Right Eye, Left Eye, Right Ear, Left Ear, Left Big Toe, Left Small Toe, Left Heel, Right Big Toe, Right Samll Toe, Right Heel, Background.  

The project uses CNN_LSTM model, which is composed of Conv2D(64,(240,1,25,2)), Maxpooling(1,2), Conv2D(32), Biderectional LSTM(50), Biderectional LSTM(50), Biderectional LSTM(50), Dropout(0.25), Dense(240).

Running Code
-------------
1. Download the dataset folder through [Google Drive](https://drive.google.com/open?id=1D2bJ9HLYt08NxpOj2HgoqJ2RpJXN_DZ_). Only dataset folder is required)
2. Download src/ folder and add dataset folder into it.
3. Run main.py

File Description
-------------
colab_src.txt (not required) : Colab code to implement openpose

dl_dataset.py : generate dataset and extract features and labels.

dl_model.py  : build the Deep Learning Model.

main.py : main function to run the program

model_cnnlstm.h5  : trained model for the project

rolling.csv : the dataset info file

Note: Put the above files and dataset/ folder into the same folder to run
