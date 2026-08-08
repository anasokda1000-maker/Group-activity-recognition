# [Hierarchical Deep Temporal Models for Group Activity Recognition](https://arxiv.org/pdf/1607.02643)

> **Authors:** Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, Greg Mori  
> **Framework:** PyTorch
## Contents

0. [Motivation](#motivation)
1. [Data](#data)
   - [Description](#Description)
   - [Distributions](#Distributions)
2. [Hierarchical Model Architecture](#hierarchical-model-architecture)
   - [Overview](#overview)
   - [Detailed Architecture](#detailed-architecture)
3. [Ablation](#Ablation)
4. [Results](#results)

## Motivation
There are two different classification problems the group activity classification and the person action classification. The person action classification depends on the spatial features and the temporal dynamics of the person himself whilst the group activity classification depends on each person in the group and its classification and the relation between the different people in the group and utilizing that information gives the model a greater learning capacity with higher performance. And this is where the idea of hierarchical models came from which was proposed first by "Hierarchical Deep Temporal Models for Group Activity Recognition" (Ibrahim et al., 2016). The hierarchical model classifies the group activity based on each person in the group and the relation between them to end up with a single classification on the group activity. In this project, we implement a hierarchical model on a dataset to tackle the group activity classification problem.

## Data

### Description
In this project we got a data consisting of volleyball olympic videos each has clips where there are sequential frames in each clip.
The frame is a photo of the court, the players and some other Classification-irrelevant factors like crowd. each clip has one label for the group activity and each player in each frame has a label for the player action in the frame as shown in **Figure 1**.
<img width="1024" height="576" alt="image" src="https://github.com/user-attachments/assets/6e2c6149-bcc1-43b0-a460-cd53b944822d" />
<p align="center">Figure 1: A frame with the group activity label "r-pass" and player action labels.</p>
Notice that the group activity label, the player action labels, and the bounding boxes around the players, which indicate their positions, are already annotated in the dataset. Additionally, the group activity label remains the same across all frames belonging to the same clip, as the group activity is assumed not to change within a single clip.

### Distributions
The total number of videos is 55 with 4830 total number of clips. Each clip has 41 frames from which we used nine frames per clip 
We used 3493 frames for training, and the remaining 1337 frames for testing. The train-test split is performed at video level, rather than at frame level so that it makes the evaluation of models more convincing. The list of action and activity labels and related statistics are tabulated in following tables:
| Group Activity Class | No. of Instances |
|---|---|
| Right set | 644 |
| Right spike | 623 |
| Right pass | 801 |
| Right winpoint | 295 |
| Left winpoint | 367 |
| Left pass | 826 |
| Left spike | 642 |
| Left set | 633 |

| Action Classes | No. of Instances |
|---|---|
| Waiting | 3601 |
| Setting | 1332 |
| Digging | 2333 |
| Falling | 1241 |
| Spiking | 1216 |
| Blocking | 2458 |
| Jumping | 341 |
| Moving | 5121 |
| Standing | 38696 |
 

## Hierarchical model architecture 

### Overview
Our model is a group activity recognition model that classifies each clip into one of eight group activity classes, utilizing both the temporal dynamics of frames in the clip and the person actions of individual players. 

### Detailed Architecture
The model starts with a ResNet50 CNN that was fine-tuned on person actions as a frozen backbone to represent the spatial features for each player in each frame in the clip as shown in **Figure 2**. Then each player spatial features along the sequential frames in the clip is fed to an LSTM for a temporal dynamic representation. And now each player has features that represents him both spatially and temporally. These features are pooled with all other players' features in the same frame and then the output represents all the frame players temporal and spatial features and their relations. And since every frame has its own features we feed all frames features in the clip to an LSTM for a single output as shown in **Figure 3**. The output now is features that deemd to be representing the temporal dynamic of a group based on individuals in it and their dynamics and the relations between them. Finally we deem these features to be most representing for the group and hence we feed them to a classifier to classify them to one of the eight group activity classes.
<img width="1024" height="576" alt="#333333" src="https://github.com/user-attachments/assets/c838f906-af73-41e9-b8b6-5b31b6428084" />
<p align="center">Figure 2: the frozen fine-tuned ResNet50 CNN extract players features and feed them to lstm ((t) stands for temporal player features).</p>
<img width="1536" height="1024" alt="ChatGPT Image Aug 8, 2026, 03_03_47 PM" src="https://github.com/user-attachments/assets/f8b652d1-7eff-49e0-8eba-740642d8b044" />
<p align="center">Figure 3: T : stands for temporal group features (combined_team frame9(t) is deemed to be representing all the clip hence we took it as an output).</p>

## Ablation
Since the hierarchical model is complicated with a lot of extensions each one has its own effect on the performance we start with a simple baseline showing its result and then extending the architecture step by step making another baselines to demonstrate each extension effect and to have a better insight of the performance difference between the proposed model and the baselines as it was proposed in the paper.

### B3) Fine-tuned Person Classification:
ResNet50 model on each player is fine-tuned to recognize
person-level actions. Then, fc7 is pooled over all players
to recognize group activities in a scene without any finetuning of the resnet50 model.
The rationale behind this baseline is to examine a scenario
where person-level action annotations as well as group
activity annotations are used in a deep learning model that
does not model the temporal aspect of group activities.
This is very similar to our two-stage model without the
temporal modeling.

### B4) Temporal Model with Image Features:
It examines the idea of feeding image level features directly to a LSTM
model to recognize group activities. In this baseline, the
ResNet50 model is deployed on the whole image and
resulting fc7 features are fed to a LSTM model.

### B5) Temporal Model with Person Features:  
fc7 features pooled over all people are fed to a LSTM model to
recognize group activities.

### B6) Two-stage Model without LSTM 1: 
This baseline is a variant of our model, omitting the person-level temporal
model (LSTM 1). Instead, the person-level classification
is done only with the fine-tuned person CNN.

### B7) Two-stage Model without LSTM 2: 
This baseline is a variant of our model, omitting the group-level temporal
model (LSTM 2). In other words, we do the final classification based on the outputs of the temporal models for
individual person action labels, but without an additional group-level LSTM

## Results
| Method | Accuracy |
|---|---|
| B3-Fine-tuned Person Classification | 68.1 |
| B4-Temporal Model with Image Features | 63.1 |
| B5-Temporal Model with Person Features | 67.6 |
| B6-Two-stage Model without LSTM 1 | 74.7 |
| B7-Two-stage Model without LSTM 2 | 80.2 |
| **Our Two-stage Hierarchical Model** | **81.9** |
