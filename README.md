# DeerAudioClassifier

**DeerAudioClassifier** is a tool designed for the automatic detection and classification of deer sounds using synchronized multi-microphone recordings. The project implements anomaly detection and deep learning classification models to identify potential deer calls from large sets of audio recordings, significantly reducing the manual investigation required.

This project is associated with the research paper ***Ensemble Deep Learning and Anomaly Detection Framework for Automatic Audio Classification: Insights into Deer Vocalizations*** published in ***Ecological Informatics*** with the DOI: ***[https://doi.org/10.1016/j.ecoinf.2024.102883](https://doi.org/10.1016/j.ecoinf.2024.102883)***. 

The tool utilizes anomaly detection to filter out potential deer sound clips, allowing researchers to focus on a smaller subset of relevant audio data. Subsequently, the tool classifies these clips using three deep learning models: **ResNet50**, **MobileNetV2**, and **EfficientNet-B2**, as well as an **Ensemble approach** for improved accuracy.

The Ensemble approach operates through majority voting. For each audio clip, the three top-performing models provide their classifications, and the final label is determined by the class receiving the most votes. For example, if two models classify an audio clip as `deer` and one model classifies it as `non-deer`, the final label will be `deer`. 


## Key Features:
- **Anomaly Detection:** Automatically identifies potential deer sounds from at least three syncoronized hourly recordings, minimizing manual processing.
- **Deep Learning Classification:** Utilizes ResNet50, MobileNetV2, and EfficientNet-B2 models to classify audio clips as "deer" or `non-deer`.
- **Ensemble Approach:** Combines predictions from all three models using a majority voting scheme to determine the final label.


## Installation

### Environment Setup
Create and activate a new Conda environment named `deer_classifier` with the required dependencies.

```bash
# Create a new conda environment named 'deer_classifier'
conda create --name deer_classifier python=3.8
```

```bash
# Activate the new environment
conda activate deer_classifier
```

```bash
# Update the conda environment with the required dependencies
conda env update --file environment.yml
```

## Running the Code
### Anomaly Detection
The anomaly detection script extracts potential deer sound clips from hourly synchronized recordings placed in folders like `221011_10PM` within the Recordings directory.

```bash
python DeerClips_AnomalyDetection.py
```
This will create a **`Potential_Deer_Clips`** folder containing all potential deer sound clips extracted from the recordings.

### Classification of Potential Deer Sound Clips
To classify the potential deer sound clips into "deer" (1) or "non-deer" (0), run the following command:

```bash
python DeerClips_ClassifierDNN.py
```
A `CSV` file containing the classification results of the three models and the Ensemble approach will be created in each hourly directory.

Sample Classification Results
| Hourly_Folder | clip_name                                                      | resnet50 | efficientnet-b2 | mobilenet_v2 | 3models |
|---------------|----------------------------------------------------------------|----------|-----------------|--------------|---------|
| 221011_10pm   | DeerSoundNo3_POI1_20221011_22_S304.2_E310.7_D6.5.wav            | 0        | 0               | 0            | 0       |
| 221011_10pm   | DeerSoundNo3_POI5_20221011_22_S304.2_E310.7_D6.5.wav            | 1        | 1               | 1            | 1       |
| 221011_10pm   | DeerSoundNo3_POI8_20221011_22_S304.2_E310.7_D6.5.wav            | 0        | 0               | 0            | 0       |
| 221011_10pm   | DeerSoundNo8_POI1_20221011_22_S409.9_E416.4_D6.5.wav            | 0        | 0               | 0            | 0       |
| 221011_10pm   | DeerSoundNo8_POI5_20221011_22_S409.9_E416.4_D6.5.wav            | 1        | 1               | 1            | 1       |

## Training Dataset
We added 3,842 sound clips of deer (1,151 clips) or non-deer (2,691 clips, including birdsongs, dogs barking, and moving vehicles) that used to train the model. 

## Folder Structure
The directory structure is as follows:

```bash
DeerAudioClassifier/
│
├── DeerClips_AnomalyDetection.py   # Anomaly detection script
├── DeerClips_ClassifierDNN.py      # Deep learning classification script
├── environment.yml                 # Conda environment configuration file
├── Models_Weight/                  # Pre-trained model weights
├── Recordings/                     # Directory for input audio recordings
│   ├── 221011_10pm/                # Full processed Hourly data
│   │   ├── POI1_221011_10pm.mp3
│   │   ├── POI2_221011_10pm.mp3
│   │   ├── POI3_221011_10pm.mp3
│   │   ├── Potential_Deer_Clips/   # Output directory for potential deer clips
│   │   ├── 221011_10pm_DNN.csv/    # Output CSV with deer clips classification
│   └── 221011_04am/                # Only Hourly data for practice 
│   │   ├── POI1_221011_01am.mp3
│   │   ├── POI2_221011_01am.mp3
│   │   ├── POI3_221011_01am.mp3
└── README.md                       # Project documentation
└── Training_Dataset                # Training dataset of 3,842 clips for "deer" and "non-deer" classes
│   ├── Deer/                       # Deer Dataset of 1,151 clips
│   │   ├── Deer_0001_20190926.wav
│   │   ├── ......
│   │   ├── ......
│   ├── NonDeer/                    # Non-deer Dataset of 2,691 clips
│   │   ├── NonDeer_0001_20190901.wav
│   │   ├── ......
│   │   ├── ......

```

## Citation
If you use this code in your research, please cite the following paper:

- **Title:** Ensemble Deep Learning and Anomaly Detection Framework for Automatic Audio Classification: Insights into Deer Vocalizations.
- **Journal:** Ecological Informatics.
- **DOI:** XXXXXX
