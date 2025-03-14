## AI-Toolbox - Motion Duet - RNN

![MocapPlayer](./data/media/rnn_screenshot.JPG)

Figure 1. Screenshot of the Motion Duet tool after training concluded. The image on the left depicts the learning progress as graph plot. The image in the middle depicts a still image from a GIF animation that shows the original motion by dancer 1. The image on the right depicts a still image of a GIF animation that shows synthetically generated motion for dance 2.

### Summary

This Python-based tool implements a motion translation model that can be trained to generate synthetic motion data for an artificial dancer that could act as a partner with a human dancer. The dataset used for training consists of motion capture recordings of two dancers that were performing a duet. During training, the model learns to predict the motion of the second dancer based on the motion of the first dancer. The tool described here doesn't operate in real-time and is used exclusively for training the model. Once a model is trained, it can be used in combination with other tools that generate synthetic motions in real-time and can be interactively controlled. 

### Installation

The software runs within the *premiere* anaconda environment. For this reason, this environment has to be setup beforehand.  Instructions how to setup the *premiere* environment are available as part of the [installation documentation ](https://github.com/bisnad/AIToolbox/tree/main/Installers) in the [AI Toolbox github repository](https://github.com/bisnad/AIToolbox). 

The software can be downloaded by cloning the [MotionDuet Github repository](https://github.com/bisnad/MotionDuet). After cloning, the software is located in the MotionDuet / rnn directory.

### Directory Structure

- rnn
  - common (contains python scripts for handling mocap data)
  - data 
    - configs (contains lists of loss weights for joints in skeletons that include hand joints)
    - media (contains media used in this Readme)
    - mocap (contains an example mocap recording)
  - results
    - anims (after training, contains synthetic motion data exported as Gif animations and FBX/BVH files)
    - histories (after training, contains logs of the training process as csv file and graph plot)
    - weights (after training, contains the weights of the trained model)

### Usage

#### Start

The tool can be used to train the motion translation model on motion capture recordings of Duets that are stored in BVH or FBX format.  The tool can be started by double clicking the rnn.bat (Windows) or rnn.sh (MacOS) shell scripts or by typing the following commands into the Anaconda terminal:

```
conda activate premiere
cd MotionDuet/rnn
python rnn.py
```

#### Functionality

##### Motion Data Import

This tool imports motion data from one or several pairs of motion files that are stored either in FBX or BVH format. In each pair of motion files, the first file stores the motion data of dancer 1 in the duet, and the second file stores the motion data of the dancer 2 in the duet. These motion files are then used to create the training set. By default, the tool loads the motion files `Jason_Take4.fbx` and `Sherise_Take4.fbx` in the `data/mocap` folder. To read different motion files, the following source code in the file rnn.py has to be modified:

```
mocap_file_path = "data/mocap"
mocap_files = [ [ "Jason_Take4.fbx", "Sherise_Take4.fbx" ] ]
mocap_valid_frame_ranges = [ [ [ 490, 30679] ] ]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_loss_weights_file = None
```

The string value assigned to the variable `mocap_file_path` specifies the path to the folder that contains motion data files. The nested list of string values assigned to the variable `mocap_files` specifies the names of motion data files that will be loaded in pairs (one file for the first dancer and the other file for the second dancer in a duet). The nested list of integer values that is assigned to the variable `mocap_valid_frame_ranges` specifies for each motion data file the frame ranges that should be used for training. Each frame range is defined by a start and end frame. It is possible to specify multiple frame ranges per motion data file. Any frames outside of these frame ranges will be excluded from training. The float value assigned to the variable `mocap_pos_scale` specifies a scaling value that is applied to joint positions. The purpose of the scaling value is to bring the position values to cm units. The integer value assigned to the variable `mocap_fps` specifies the number of frames per second with which the motion data is stored. This value has no influence on training but affects the synthetic motion data that is exported at the end of a training run. The string value assigned to the variable `mocap_loss_weights_file` specifies the path to a configuration file that contains loss scales for skeleton joints. Loading such a configuration file is recommended for motion  data that contains multiple joints per hand in order to reduce the influence of the hand joints on the overall loss calculation during training. If the motion data doesn't contain multiple hand joints, then this variable can be set to `None`. 

#### Functionality

The tool extracts the required motion data from the imported motion files into motion sequences, splits these sequences into short segments to create the dataset, then constructs and initialises machine learning model, trains this model using on the dataset, and finally stores the trained models weights, the training history, and examples of original and generated motion sequences. 

##### Model Settings

The model consists of one or several [Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-Term_Memory) (LSTM) layers. By default, there are 2 layers and 512 units per layer. When constructing the model, its weights can either be initialised with random values or with values that have been determined in a previous training run. To use different model settings, the following source code in the file `rnn.py` has to be modified:

```
rnn_layer_dim = 512
rnn_layer_count = 2

load_weights = False
rnn_weights_file = "results/weights/rnn_weights_epoch_100"
```

The integer value assigned to the variable `rnn_layer_dim` specifies the number of units per LSTM layer. The integer value assigned to the variable `rnn_layer_count` specifies the number of LSTM layers. The boolean value assigned to the variable `load_weights` specifies if the model should be initialised with previously stored weights or not. The string value assigned to the variable `rnn_weights_file` specifies the path to a previously exported weights file. 

##### Dataset Settings

The dataset consists of pairs of motion sequences that have been extracted from the loaded motion files. The first sequence in the pair represents the motion sequence of dancer 1 that is passed to the model as input. The second sequence in the pair represents the motion sequence of dancer 2 that the model should learn to output. The motion sequences consist of a time series of values that represent at each timestep a full skeleton pose. A pose is represented by joint orientations using quaternions.

##### Training Settings

During training, a short motion sequence of dancer 1 is passed as input to the model and the model predicts the sequence for dancer 2. 

In `rnn.py`, the loss is calculated as a combination of several losses. These are: 

- norm_loss: loss based on the deviation of the predicted quaternions from unit length
- pos_loss: loss based on the deviation of the predicted joint positions from the correct joint positions. The joint positions are derived from joint rotations using forward kinematics. 
- quat_loss: loss based on the deviation of the predicted joint rotations from the correct joint rotations.

When running the tool, it employs default training settings. To change these settings, the follows source code in the file `rnn.py` has to be changed:

```
batch_size = 32
test_percentage = 0.1
seq_input_length = 64
learning_rate = 1e-4
norm_loss_scale = 0.1
pos_loss_scale = 0.1
quat_loss_scale = 0.9
model_save_interval = 10
epochs = 200
```

The integer value assigned to the variable `batch_size` specifies the number of motion examples in a training batch. The float value assigned to the variable `test_percentage` specifies the percentage of training data used for testing the model. The integer value assigned to the variable `seq_input_length` specifies the length (in number of frames) that is used a input to the model. The float value assigned to the variable learning_rate specifies the initial learning rate. The float value assigned to the variable `norm_loss_scale` specifies the weighted contribution of the quaternion normalisation loss to the overall training loss. The float value assigned to the variable `pos_loss_scale` specifies its weighted contribution of the predicted joint positions loss to the overall training loss. The float value assigned to the variable `quat_loss_scale` specifies its weighted contribution to the predicted joint rotations loss to the overall training loss. The integer value assigned to the variable `model_save_interval` specifies the interval (in number of epochs) at which model weights are stored. The integer value assigned to the variable epochs specifies the number of `epochs` used for training.

##### Training 

Once the dataset has been created and the model initialised, training begins and runs for the number of epochs specified by the user. During training, the tool prints for each epoch a log message to the console that provide information about the training progress. 

An example log message looks like this:

`epoch 1 : train: 2.6130 test: 2.2668 norm 0.0869 pos 23.7277 quat 0.2572 time 35.74`

The information specifies, from left to right: the epoch number, the loss on the train set, the loss on the test set, the quaternion normalisation loss, the joint position loss, the joint rotation loss, and the time elapsed.

At the end of training, the tool displays the training history as graph plot, and stores the training history both as image and `.csv` file, the last model weights, and original and predicted motion sequences for dancer 1 and dancer 2 exported either as BVH or FBX file and GIF animation.

### Limitations and Bugs

- The tool only supports motion capture recordings that contain a single person.
- `rnn.py` reads only motion capture recordings in FBX format in which each skeleton pose has its own keyframe and in which the number of keyframes is the same for all skeleton joints.



