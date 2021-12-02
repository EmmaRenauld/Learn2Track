"""
Additional parameters for Learn2track as compared to other projects in dwi_ml:

model:
    previous_dirs:
        nb_previous_dirs: int
             Concatenate X previous streamline directions to the input vector.
             Null is equivalent to 0.
        embedding: str
             One of 'no_embedding', 'nn_embedding', 'cnn_embedding' or
             None (nb_previous_dirs must then be 0).
        embedding_output_size: int
           Size of the output after passing the previous dirs through the
           embedding.
    input:
        embedding: str
             One of 'no_embedding', 'nn_embedding' or 'cnn_embedding'.
        output_size_ratio: float
             Output size for the input embedding will be intput_size * ratio
             Ex: 1 means the output_size = input_size.
                 2 means input is divided into 2.
    neighborhood:
        sphere_radius: float
             If not null, a neighborhood will be added to the input
             information. This neighborhood definition lies on a sphere. It
             will be a list of 6 positions (up, down, left, right, behind, in
             front) at exactly given radius around each point of the
             streamlines, in voxel space. **Can't be used together with
             grid_radius.
        grid_radius: int
             If not null, a neighborhood will be added to the input
             information. This neighborhood definition uses a list of points
             similar to the original voxel grid around each point of the
             streamlines. Ex: with radius 1, that's 27 points. With radius 2,
             that's 125 points. Radius is in voxel space. **Can't be used
             together with sphere_radius.
    rnn:
        key: str
             One of 'lstm' or 'gru'
        layer_sizes: list[int]
             The output size after each layer (the real output size depends on
             skip connections).
        dropout: float
             If >0: add a dropout layer between RNN layers.
        use_skip_connections: bool
             Set to true to add skip connections. The pattern for skip
             connections is as seen here:
             https://arxiv.org/pdf/1308.0850v5.pdf
        use_layer_normalization: bool
             set to true to add layer normalization. Explained here:
             https://arxiv.org/pdf/1607.06450.pdf
    direction_getter:
        key: str
             One of 'cosine-regression', 'l2-regression',
             'sphere-classification', 'gaussian', 'gaussian-mixture',
             'fisher-von-mises', 'fisher-von-mises-mixture'
"""
