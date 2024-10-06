import os
import time
from datetime import timedelta 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
from glob import glob
from collections import OrderedDict
import librosa as lb
import librosa.display
import torch
from torch import nn
from torchvision import transforms, datasets, models
from efficientnet_pytorch import EfficientNet

# Get current working directory
Work_dir_NN = os.getcwd()

# Create function to create directory if it doesn't exist
def Create_Dir_Fuc(Dir_):
    if not os.path.exists(Dir_):
        os.makedirs(Dir_)

# Define directories
Main_Records_dir = os.path.join(Work_dir_NN, "Recordings")
MODEL_WEIGHT_PATH = os.path.join(Work_dir_NN, "Models_Weight")

# Mel spectrogram parameters
win_length = 2048
hop_length = win_length // 4
n_fft = win_length
window = 'hann'
n_mels = 128
fmax_value = 4000

# Models to use
use_model_list = ['resnet50', 'efficientnet-b2', 'mobilenet_v2']

# Create models based on names and weights
def model_create(model_name:str, weight:str, fc_num:int):
    # Load weights safely with weights_only=True
    weight_path = os.path.join(MODEL_WEIGHT_PATH, f'{weight}')
    weight = torch.load(weight_path, map_location={'cuda:0':'cpu'}, weights_only=True)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in weight.items():
        name = k[8:] # remove `feature.` <- feature. is the name set during training. 
        new_state_dict[name] = v
    if model_name == 'resnet50' and 'resnet50' in use_model_list:
        _model = models.resnet50(weights = None)
        _model.fc = nn.Linear(fc_num, 2)
    elif model_name == 'mobilenet_v2' and 'mobilenet_v2' in use_model_list:
        _model = models.mobilenet_v2(weights=None)
        # _model.fc = nn.Linear(fc_num, 2)
        _model.classifier[1] = nn.Linear(fc_num, 2)
    elif model_name == 'efficientnet-b2' and 'efficientnet-b2' in use_model_list:
        _model = EfficientNet.from_name('efficientnet-b2') # https://nonbiri-tereka.hatenablog.com/entry/2020/03/26/083557
        _model._fc = nn.Linear(fc_num, 2)
    _model.load_state_dict(new_state_dict, strict=False)
    return _model

# Read models' information and create model list
model_ls = []
df_model = pd.read_csv(f'{MODEL_WEIGHT_PATH}/deer_project_model_weights.csv')

for _, content in df_model.iterrows():
    model_name, model_weight, fc_num, _ = content
    model = model_create(model_name, model_weight, fc_num)
    model_ls.append(model)

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare result columns and folder list
columns = ['resnet50', 'efficientnet-b2', 'mobilenet_v2']
List_of_Folders = sorted(next(os.walk(Main_Records_dir))[1])

# Iterate through each hourly folder
for Hourly_Folder_ in List_of_Folders:
    Hourly_dir = os.path.join(Main_Records_dir, Hourly_Folder_)
    
    # Prepare results dictionary
    results = {"Hourly_Folder": [], "clip_name": [], 'resnet50': [], 'efficientnet-b2': [], 'mobilenet_v2': []}

    # Check if CSV file exists
    fn_dir = os.path.join(Hourly_dir, f'{Hourly_Folder_}_DNN.csv')
    Clips__dir = os.path.join(Hourly_dir, "Potential_Deer_Clips")

    # Get list of clips and filter for .wav files only
    Clips_List = [file for file in next(os.walk(Clips__dir))[2] if file.endswith('.wav')]
    Clips_List = sorted(Clips_List, key=lambda x: int("".join([i for i in x if i.isdigit()])))

    for Clip_ in Clips_List:
        plt.close('all')
        Clip_Path = os.path.join(Clips__dir, Clip_)

        # Load audio file
        SignalAry, sampling_rate = lb.load(Clip_Path, sr=None)
        mel_power = lb.feature.melspectrogram(y=SignalAry, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length,window=window, center=True, n_mels=n_mels, fmax=fmax_value)# こっちのfmaxばデータの周波数                                                                                                                                                                                                                                      
        mel_power_in_db = lb.power_to_db(mel_power, ref=np.max)

        # Plotting spectrogram and converting to image array
        fig, axs = plt.subplots(figsize=(6.3, 4))
        Image_ = lb.display.specshow(mel_power_in_db, sr=sampling_rate, x_axis="time", y_axis="mel", fmax=fmax_value)
        plt.colorbar(Image_, label='Amplitude (dB)')
        fig.tight_layout()
        fig.canvas.draw()

        # Convert image to numpy array
        data = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        c = len(data) // (w * h)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c)

        # Transform image and prepare for model input
        t_sound = transform(img).unsqueeze(0)

        # Record metadata
        results['Hourly_Folder'].append(Hourly_Folder_)
        results['clip_name'].append(Clip_)


        # Model inference and recording results
        for model, column_name in zip(model_ls, columns):
            model.eval()
            y_pred = model(t_sound)
            y_preds = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
            results[column_name] += list(y_preds)

    # Create DataFrame and save results
    df_result = pd.DataFrame(results)

    # Combine models' predictions
    df_result['3models'] = df_result[['resnet50', 'efficientnet-b2', 'mobilenet_v2']].sum(axis=1).apply(lambda x: 1 if x >= 2 else 0)
    df_result.to_csv(fn_dir, index=False)
