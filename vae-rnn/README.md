## AI-Toolbox - Motion Duet - VAE RNN

![MocapPlayer](./data/media/vae-rnn_deepfake_screenshot.JPG)

Figure 1. Screenshot of the Motion Deep Fake tool after training concluded. The image on the left depicts the learning progress as graph plot. The images on the right depict stills from GIF animations. From top to bottom and left to right, these stills show the original motion of dancer 1, the original motion of dancer 2, the motion of dancer 1 transformed into the motion of dancer 2, and the motion of dancer 2 transformed into the motion of dancer 1. 

### Summary

This Python-based tool implements a motion translation model that can be trained to generate synthetic motion data for an artificial dancer that could act as a partner with a human dancer. The dataset used for training consists of motion capture recordings of two dancers that were performing a duet. There exist two versions of the tool, both of which use a Variational Autoencoder that learns to generate synthetic motion data by encoding and decoding short motion excerpts into and from compressed representations. The first version of the tool learns to directly translate the motion of dancer 1 into the corresponding motion of dancer 2 based on their simultaneous occurrence in the motion capture recordings. The second version of the tool employs a deep fake approach. This version learns to transform the motions of one dancer so that these motions become more similar to the motions of the other dancer in the duet. The tools described here don't operate in real-time and are used exclusively for training their respective machine learning models. Once the models are trained, they can be used in combination with other tools that generate synthetic motions in real-time and can be interactively controlled. 

### Installation

The software runs within the *premiere* anaconda environment. For this reason, this environment has to be setup beforehand.  Instructions how to setup the *premiere* environment are available as part of the [installation documentation ](https://github.com/bisnad/AIToolbox/tree/main/Installers) in the [AI Toolbox github repository](https://github.com/bisnad/AIToolbox). 

The software can be downloaded by cloning the [MotionDuet repository](https://github.com/bisnad/MotionDuet). After cloning, the software is located in the MotionDuet / vae-rnn directory.

### Directory Structure

- vae-rnn
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

The tool exists in two versions. The first version named vae_rnn is used to train a motion transformation model that employs a standard Variational Autoencoder to directly translate the motion of dancer 1 into the corresponding motion of dancer 2 based on their simultaneous occurrence in the motion capture recordings. The second version named vae_rnn_deepfake.py employs a deep fake approach. This version learns to transform the motions of one dancer so that these motions become more similar to the motions of the other dancer in the duet.

The first version of the tool can be started by double clicking the vae_rnn.bat (Windows) or vae_rnn.sh (MacOS) shell scripts or by typing the following commands into the Anaconda terminal:

```
conda activate premiere
cd MotionDuet/vae_rnn
python vae_rnn.py
```

The second version of the tool can be started by double clicking vae_rnn_deepfake.bat (Windows) or vae_rnn_deepfake.sh (MacOS) shell scripts or by typing the following commands into the Anaconda terminal:

```
conda activate premiere
cd MotionDuet/vae_rnn
python vae_rnn_deepfake.py
```

#### Functionality

##### Motion Data Import

Both versions of the tool imports motion data from one or several pairs of motion files that are stored either in FBX or BVH format. In each pair of motion files, the first file stores the motion data of dancer 1 in the duet, and the second file stores the motion data of the dancer 2 in the duet. These motion files are then used to create the training set. By default, the tool loads the motion files `Jason_Take4.fbx` and `Sherise_Take4.fbx` in the `data/mocap` folder. To read different motion files, the following source code in the file `vae_rnn.py` or v`ae_rnn_deepfake.py` has to be modified:

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

##### VAE-RNN Model Settings

The first version of the tool employs a standard Variational Autoencoder to directly translate the motion of dancer 1 into the corresponding motion of dancer 2 based on their simultaneous occurrence in the motion capture recordings. The Variational Autoencoder consists of a total of two models: the encoder and the decoder. The encoder takes as input a short motion excerpt of dancer 1 and outputs the mean and standard deviation of a normal distribution. A latent vector can be obtained by sampling from this distribution. The decoder takes as input a latent vector and decompresses it into a short motion excerpt for dancer 2. Both the encoder and decoder consist of one or several [Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-Term_Memory) (LSTM) layers and fully connected layers (Dense) layers.  The number of units for the first and last Dense layer is automatically calculated. The number of units for the middle Dense layer(s) is by default 512. Other than that, the dimension of the latent vector representation of motion excerpts has to be specified. The default value for this dimension is 32. When constructing the models, their weights can either be initialised with random values or with values that have been determined in a previous training run. To use different models settings, the the following source code in the file`vae_rnn.py` has to be modified:

```
latent_dim = 32
sequence_length = 64
ae_rnn_layer_count = 2
ae_rnn_layer_size = 256
ae_rnn_bidirectional = True
ae_dense_layer_sizes = [ 512 ]

save_weights = True
load_weights = False
    
encoder_weights_file = "results_vae_jason_sherise/weights/encoder_weights_epoch_600"
decoder_weights_file = "results_vae_jason_sherise/weights/decoder_weights_epoch_600"
```

The integer value assigned to the variable `latent_dim` specifies the dimension of the latent vector representation. The integer value assigned to the variable `sequence_length` specifies the length (in number of frames) of the motion excerpt the Autoencoder operates on. The integer value assigned to the variable `ae_rnn_layer_count` specifies the number of LSTM layers in the encoder and decoder models. The integer value assigned to the variable `ae_rnn_layer_size` specifies the number of units per LSTM layer in the encoder and decoder models. The boolean value assigned to the variable ae_rnn_bidirectional specifies if the LSTM layers process sequential data in both forward and backward directions. A value of True corresponds to both forward and backwards direction and a value of False corresponds to forward direction only. The list of integer values assigned to the variable `ae_dense_layer_sizes` specifies the number of units in the Dense layers in the encoder and decoder with the exception of the first and last Dense layer (for which the number of units is determined automatically). The boolean value assigned to the variable `load_weights` specifies if the models should be initialised with previously stored weights or not. The string values assigned to the variables `encoder_weights_file` and`decoder_weights_file` specify the paths to the previously exported weights file for the encoder and decoder, respectively. 

##### VAE-RNN Deepfake Model Settings

The second version of the tool follows a deep fake approach. Accordingly, instead of a direct translation of motions from dancer 1 to dancer 2 based on their simulatenous occurence, the tool transform the motions of one dancer so that these motions become more similar to the motions of the other dancer in the duet. In addition, contrary to the direct translation approach, this version of the tool allows to transform motions in both directions, from dancer 1 to dancer 2 and from dancer 2 to dancer 1.  The tool employs an extended version of a Variational Autoencoder that consists of one encoder and two decoders.  The encoder takes as input a short motion excerpt that can either be from dancer 1 or dancer 2. It then outputs the mean and standard deviation of a normal distribution. A latent vector can be obtained by sampling from this distribution. The two decoders are specific for each dancer. Decoder 1 takes as input a latent vector and decompresses it into a short motion excerpt for dancer 1. Decoder 2 takes as input a latent vector and decompresses it into a short motion excerpt for dancer 2. Both the encoder and decoders consist of one or several [Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-Term_Memory) (LSTM) layers and fully connected layers (Dense) layers.  The number of units for the first and last Dense layer is automatically calculated. The number of units for the middle Dense layer(s) is by default 512. Other than that, the dimension of the latent vector representation of motion excerpts has to be specified. The default value for this dimension is 32. When constructing the models, their weights can either be initialised with random values or with values that have been determined in a previous training run. To use different models settings, the the following source code in the file`vae_rnn_deepfake.py` has to be modified:

```
latent_dim = 32
sequence_length = 64
ae_rnn_layer_count = 2
ae_rnn_layer_size = 256
ae_rnn_bidirectional = True
ae_dense_layer_sizes = [ 512 ]

save_weights = True
load_weights = False

encoder_weights_file = "results_deepfake_jason_sherise/weights/encoder_weights_epoch_600"
decoder1_weights_file = "results_deepfake_jason_sherise/weights/decoder1_weights_epoch_600"
decoder2_weights_file = "results_deepfake_jason_sherise/weights/decoder2_weights_epoch_600"
```

The purpose of these variables is the same as in the first version of the tool, with the exception of three instead of two weight files.  The string values assigned to the variables `encoder_weights_file`,`decoder1_weights_file`, and `decoder2_weights_file` specify the paths to the previously exported weights file for the encoder, the decoder for dancer 1, and the decoder for dancer 2, respectively. 

##### Dataset Settings

The dataset consists of pairs of motion sequences that have been extracted from the loaded motion files. The first sequence in the pair represents the motion sequence of dancer 1 that is passed to the model as input. The second sequence in the pair represents the motion sequence of dancer 2 that the model should learn to output. The motion sequences consist of a time series of values that represent at each timestep a full skeleton pose. A pose is represented by joint orientations using quaternions.

##### Training Settings

During training, two short motion sequences, one from dancer 1 and one from dancer 2 are passed as inputs to the encoder. The encoder compresses each motion sequence and outputs a normal distribution from which a latent vector representation can be sampled.  The latent vector representation of the motion sequence from dancer 1 is then passed to decoder1 that decompresses it back into a motion sequence for dancer 1.  Similarily, the latent vector representation of the motion sequence from dancer 2 is passed to decoder 2 that decompresses it back into a motion sequence for dancer 2.  At each training step, the loss of the Autoencoder is calculated based on a combination of the reconstruction of both motion sequences and a similarity measure (KL-Divergence) between the outputs of the encoder and a unit distribution. The influence of the KL-Divergence loss on the overall training loss cyclically varies over time. 

##### Training Settings

In `vae_rnn.py` and `vae_rnn_deepfake.py,`  the loss of the autoencoder is calculated as a combination of several losses. These are: 

- variational_loss: loss based on the KL-Divergence between mean and standard deviation output by encoder and a unit distribution.
- ae_norm_loss: loss based on the deviation of the reconstructed joint rotations from unit quaternions
- ae_pos_loss: loss based on the deviation of the reconstructed joint positions from the correct joint positions. The joint positions are derived from joint rotations using forward kinematics. 
- ae_quat_loss: loss based on the deviation of the reconstructed joint rotations from the correct joint rotations.

When running the tool, it employs default training settings. To change these settings, the follows source code in the file `vae_rnn.py` has to be changed:

```
sequence_offset = 2
batch_size = 16
test_percentage  = 0.2
ae_learning_rate = 1e-4
ae_norm_loss_scale = 0.1
ae_pos_loss_scale = 0.1
ae_quat_loss_scale = 1.0
ae_kld_loss_scale = 0.0
kld_scale_cycle_duration = 100
kld_scale_min_const_duration = 20
kld_scale_max_const_duration = 20
min_kld_scale = 0.0
max_kld_scale = 0.1

epochs = 600
model_save_interval = 50
```

The integer value assigned to the variable `sequence_offset` specifies the offset (in number of frames) used when extracting motion excerpts from the loaded motion files. The integer value assigned to the variable `batch_size` specifies the number of motion examples in a training batch. The float value assigned to the variable `test_percentage` specifies the percentage of training data used for testing the model. The float value assigned to the variable `ae_learning_rate` specifies the initial learning rate for the encoder and decoders. The float value assigned to the variable `norm_loss_scale` specifies the weighted contribution of the quaternion normalisation loss to the overall autoencoder training loss. The float value assigned to the variable `pos_loss_scale` specifies its weighted contribution of the joint position reconstruction loss to the overall autoencoder training loss. The float value assigned to the variable `quat_loss_scale` specifies its weighted contribution to the joint rotation reconstruction loss to the overall autoencoder training loss. The float value assigned to the variable `ae_kld_loss_scale` specifies the weighted contribution of the KL-Divergence loss to the overall autoencoder training loss. This value will is automatically calculated during training. The integer value assigned to the variable `kld_scale_cycle_duration` specifies the duration (in number of epochs) of a cycle during which the KL Divergence scale increases from a minim to a maximum value. The minimum value is specified by the float value assigned to the variable  `min_kld_scale`. The maximum value is specified by the float value assigned to the variable  `max_kld_scale`.  The integer value assigned to the variable `kld_scale_min_const_duration` specifies the duration (in number of epochs) at the beginning of a cycle during which the KL Divergence scale is constant and has a minimum value. The integer value assigned to the variable `kld_scale_max_const_duration` specifies the duration (in number of epochs) at the end of a cycle during which the KL Divergence scale is constant and has a maximum value. The integer value assigned to the variable epochs specifies the number of `epochs` used for training. The integer value assigned to the variable `model_save_interval` specifies the interval (in number of epochs) at which model weights are stored. 

##### Training 

Once the dataset has been created and the model initialised, training begins and runs for the number of epochs specified by the user. During training, both tools prints for each epoch the same type of log message to the console that provide information about the training progress. 

An example log message produced by `vae_rnn.py` and  `vae_rnn_deepfake.py` looks like this:

`epoch 1 : ae train: 2.7454 ae test: 2.5641 norm 0.0836 pos 24.7882 quat 0.2582 kld 2.8813 time 71.69`

The information specifies, from left to right: the epoch number, the overall loss of the autoencoder on the train set, the overall loss of the autoencoder on the test set, the quaternion normalisation loss, the joint position loss, the joint rotation loss, the KL-Divergence loss ,and the time elapsed. 

At the end of training, the tool displays the training history as graph plot, and stores the training history both as image and `.csv` file, the last model weights, and an original and predicted motion sequences for dancer 1 and dancer 2 exported either as BVH or FBX file and GIF animation.

### Limitations and Bugs

- The tool only supports motion capture recordings that contain a single person.
- `aae_rnn.py` reads only motion capture recordings in FBX format in which each skeleton pose has its own keyframe and in which the number of keyframes is the same for all skeleton joints.



