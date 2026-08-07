# Hierarchical-Deep-Temporal-Models-for-Group-Activity-Recognition-implementation-

## Motivation
There are two different classification problems the group activity classification and the person action classification. The person action classification depends on the spatial features and the temporal dynamics of the person himself whilst the group activity classification depends on each person in the group and its classification and the relation between the different people in the group and utilizing that information gives the model a greater learning capacity with higher performance. And this is where the idea of hierarchical models came from which was proposed first by "Hierarchical Deep Temporal Models for Group Activity Recognition" (Ibrahim et al., 2016). The hierarchical model classifies the group activity based on each person in the group and the relation between them to end up with a single classification on the group activity. In this project, we implement a hierarchical model on a dataset to tackle the group activity classification problem.

## Data
### explaination
In this project we got a data consisting of volleyball olympic videos each has clips where there are sequential frames in each clip.
The frame is a photo of the court, the players and some other Classification-irrelevant factors like crowd. each clip has one label for the group activity and each player in each frame has a label for the player action in the frame as shown in **Figure 1**.
<img width="1024" height="576" alt="image" src="https://github.com/user-attachments/assets/6e2c6149-bcc1-43b0-a460-cd53b944822d" />
<p align="center">Figure 1: A frame with the group activity label "r-pass" and player action labels.</p>
Notice that the group activity label, the player action labels, and the bounding boxes around the players, which indicate their positions, are already annotated in the dataset. Additionally, the group activity label remains the same across all frames belonging to the same clip, as the group activity is assumed not to change within a single clip.

### distributions
The total number of videos is 55 with 4830 total number of clips. Each clip has 41 frame from which we used nine frames per clip 
We used 3493 frames for training, and the remaining 1337 frames for testing. The train-test split of is performed at video level, rather than at frame level so that it makes the evaluation of models more convincing. The list of action and activity labels and related statistics are tabulated in following tables:
| Group Activity Class | No. of Instances |
|-----------------------|-------------------|
| Right set              | 644               |
| Right spike            | 623               |
| Right pass              | 801               |
| Right winpoint          | 295               |
| Left winpoint            | 367               |
| Left pass                | 826               |
| Left spike                | 642               |
| Left set                   | 633               |


 

## Hierarchical model architecture 
Our model is supposed to be a group activity recognition model that classify each clip into one of eight group activity classes utilizing both the temporal dynamic of frames in the clip and the person actions in the clip. The model starts with a Resnet50 CNN that was fine-tuned on person actions as a frozen backbone to represent the spatial features for each player in each frame in the clip. Then each player spatial features along the sequential frames in the clip is fed to an LSTM for a temporal dynamic representation. And now each player has a features that represents him both spatially and temporaly. These features are pooled with all other 'players features in the same frame and then the output represents all the frame players temporal and spatial features and their relations. And since every frame has its own features we can feed all frames features in the clip to an LSTM for a single output .The output now is a features that deemd to be representing the temporal dynamic of a group based on individuals in it and their dynamics and the relations between them. Finally we deem these features to be most representing for the group and hence we feed them to a classifier to classify them to one of the eight group activity classes 

## Baselines
Since the hierarchical model is complicated wtih a lot of extentions each one has its own effect on the performance we start with a simple baseline showing its result and then extending the architecture step by step making another baselines to demonstarte each extention effect and to have a better insight of the perforamnce difference between the proposed model and the baselines as it was proposed in the paper.

### B1) Image Classification: This baseline is the basic resnet50
model fine-tuned for group activity recognition in a single
frame.

### B3) Fine-tuned Person Classification: This baseline is similar to the previous baseline with one distinction. The
resnet50 model on each player is fine-tuned to recognize
person-level actions. Then, fc7 is pooled over all players
to recognize group activities in a scene without any finetuning of the resnet50 model.
The rationale behind this baseline is to examine a scenario
where person-level action annotations as well as group
activity annotations are used in a deep learning model that
does not model the temporal aspect of group activities.
This is very similar to our two-stage model without the
temporal modeling.
### B4) Temporal Model with Image Features: This baseline is
a temporal extension of the first baseline. It examines the
idea of feeding image level features directly to a LSTM
model to recognize group activities. In this baseline, the
AlexNet model is deployed on the whole image and
resulting fc7 features are fed to a LSTM model. This
baseline can be considered as a reimplementation of
Donahue et al. [11].
### B5) Temporal Model with Person Features: This baseline is
a temporal extension of the second baseline: fc7 features
pooled over all people are fed to a LSTM model to
recognize group activities.
### B6) Two-stage Model without LSTM 1: This baseline is a
variant of our model, omitting the person-level temporal
model (LSTM 1). Instead, the person-level classification
is done only with the fine-tuned person CNN.
### B7) Two-stage Model without LSTM 2: This baseline is a
variant of our model, omitting the group-level temporal
model (LSTM 2). In other words, we do the final classification based on the outputs of the temporal models for
individual person action labels, but without an additional group-level LSTM

## Results
