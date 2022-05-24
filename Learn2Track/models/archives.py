

# Forward method of the stacked_rnn when it was created to deal with either
# torch or packed. Changed to only accept packed.
def forward(self, inputs: Union[Tensor, PackedSequence],
            hidden_states: Tuple[Tensor, ...] = None):
    """
    Parameters
    ----------
    inputs : torch.Tensor or PackedSequence
        Batch of input sequences. Size (seq, features).
    hidden_states : tuple of torch.Tensor
        The current hidden states of the model ((h_(t-1), C_(t-1) for LSTM)

    Returns
    -------
    last_output : Tensor or PackedSequence, depending on the input type.
        Output of the last RNN layer.
    out_hidden_states : tuple of Tensor
        The last step hidden states (h_(t-1), C_(t-1) for LSTM) for each
        layer.
    """
    if isinstance(inputs, Tensor):
        was_packed = False
        inputs_tensor = inputs
    elif isinstance(inputs, list):
        raise TypeError("Unexpected input type! Data should not be a list."
                        "You could try using PackedSequences.")
    elif isinstance(inputs, PackedSequence):
        was_packed = True
        inputs_tensor = inputs.data
    else:
        raise TypeError("Unexpected input type!")

    # Arranging states
    if hidden_states is None:
        hidden_states = (None,) * len(self.rnn_layers)

    # Initializing variables that we will want to return
    out_hidden_states = []
    outputs = []

    # Running forward on each layer:
    # linear --> layer norm --> dropout --> skip connection
    last_output = inputs
    for i, (layer_i, states_i) in enumerate(zip(self.rnn_layers,
                                                hidden_states)):
        logger.debug(
            'Applying StackedRnn layer #{}\n'
            '    Layer is: {}\n'
            '    Received input size: {}.'
                .format(i, layer_i,
                        [last_output.data.shape if was_packed else
                         last_output.shape]))

        # Apply main sub-layer: either as 3D tensor or as packedSequence
        last_output, new_state_i = layer_i(last_output, states_i)

        # Forward functions for layer_norm, dropout and skip take tensors
        # Does not matter if order of datapoints is not kept, applied on
        # each data point separately
        if was_packed:
            last_output = last_output.data

        logger.debug('   Output size after main sub-layer: {}'
                     .format(last_output.shape))

        # Apply layer normalization
        if self.use_layer_normalization:
            last_output = self.layer_norm_layers[i](last_output)

        logger.debug('   Output size after normalization: {}'
                     .format(last_output.shape))

        if i < len(self.rnn_layers) - 1:
            # Apply dropout except on last layer
            if self.dropout > 0:
                last_output = self.dropout_module(last_output)
                logger.debug('   Output size after dropout: {}'
                             .format(last_output.shape))

            # Apply ReLu activation except on last layer
            last_output = self.relu_sublayer(last_output)
            logger.debug('   Output size after reLu: {}'
                         .format(last_output.shape))

        # Saving layer's last_output and states for later
        outputs.append(last_output)
        out_hidden_states.append(new_state_i)

        if self.use_skip_connections and i < len(self.rnn_layers) - 1:
            # Adding skip connection, i.e. initial input.
            # See here: https://arxiv.org/pdf/1308.0850v5.pdf
            # Skip connection for last layer is different and will be done
            # outside the loop.
            last_output = torch.cat((last_output, inputs_tensor), dim=-1)
            logger.debug('   Output size after skip connection: {}'
                         .format(last_output.shape))

        # Packing. Either for use on next layer or for returning a packed
        # sequence.
        if was_packed:
            last_output = PackedSequence(last_output, inputs.batch_sizes,
                                         inputs.sorted_indices,
                                         inputs.unsorted_indices)

    # Final last_output
    if self.use_skip_connections:
        if was_packed:
            # Can't just replace last_output.data, throws an attribute
            # error. Solutions: create the model as inplace (ex,
            # torch.nn.Dropout(p=p, inplace=True)), or reconstruct a
            # packedSequence
            last_output = PackedSequence(torch.cat(outputs, dim=-1),
                                         inputs.batch_sizes,
                                         inputs.sorted_indices,
                                         inputs.unsorted_indices)
        else:
            last_output = torch.cat(outputs, dim=-1)

        logger.debug(
            'Final skip connection: concatenating all outputs '
            'but not input: {} = {}'
                .format([outputs[i].shape for i in
                         range(len(outputs))],
                        [last_output.data.shape if was_packed else
                         last_output.shape]))
    return last_output, tuple(out_hidden_states)