# AI-Toolbox - Motion Duet

The "Motion Duet" category of the AI-Toolbox contains a collection of python-based generative machine learning models that can generate synthetic motions for an artificial dancer that could act as a partner with a human dancer. These models are trained on motion capture recordings of two dancers who were performing a duet. From these recordings, the models learn to predict the motions of one dancer from the motions of the other dancer. After training, the models mimic at least to some degree the movement based interaction between two dancers in a duet. 

The following tools are available:

- [rnn](rnn)

  A Python-based tool for training a motion-translation model. This model takes as input a short motion of one dancer and generates the motions of a second dancer. 

- [rnn_interactive](rnn_interactive)

  A Python-based tool that employ a previously trained motion translation model in real-time. 

- [vae-rnn](vae-rnn)

  A Python-based tool for training a motion deep fake model. This model that takes as input a short motion of one dancer and makes them resemble the motions of a second dancer.


- [vae-rnn_interactive](vae-rnn_interactive)

  A Python-based tool that employs a previously trained motion deep fake model in real-time.
