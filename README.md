# CNN_LSTM_V2

Project Description
-------------
This project aims to design a deep learning model for rolling action recognition. The rolling action includes forward roll, backward roll, shoulder roll and dive roll. But side roll (pencil roll) is not included.

The project use Openpose from CMU (https://github.com/CMU-Perceptual-Computing-Lab/openpose) to implement feature extraction and obtain 25 keypoints of body as input features, which includes: 

Nose, Neck, Right Shoulder, Right Elbow, Right Wrist, Left Shoulder, Left Elbow, Left wrist, Mid Hip, Right Hip, Right Knee, Right Ankle, Left Hip, Left knee, Left Ankle, Right Eye, Left Eye, Right Ear, Left Ear, Left Big Toe, Left Small Toe, Left Heel, Right Big Toe, Right Samll Toe, Right Heel, Background.  

Running Code
-------------
1. Download the dataset folder through (https://drive.google.com/open?id=1D2bJ9HLYt08NxpOj2HgoqJ2RpJXN_DZ_). Only dataset folder is required)
2. Download src/ folder and add dataset folder into it.
3. Run main.py

File Description
-------------
colab_src.txt (not required) : Colab code to implement openpose

dl_dataset.py : generate dataset and extract features and labels.

dl_model.py  : build the Deep Learning Model.

main.py : main function to run the program

model_cnnlstm.h5  : Trained model for the project

rolling.csv : the dataset info file

Note: Put the above files and dataset/ folder into the same folder to run
