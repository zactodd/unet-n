from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from typing import Tuple, Optional, Union


def unet_n(n: int = 5, input_size: Tuple[int] = (256, 256, 1), pretrained_weights: Optional[str] = None,
           activation='relu', padding='same', kernel_initializer='he_normal',
           learning_rate: float = 1e-4, loss='binary_crossentropy', metrics=['accuracy'],
           output_activation="sigmoid") -> Model:
    """
    Unet-n model.
    :param n: The the number of down layers.
    :param input_size: The input size.
    :param pretrained_weights: If not None, the path to the pretrained weights.
    :param activation: Activation function of internal layers, default is 'relu'.
    :param padding: Padding strategy, default 'same'.
    :param kernel_initializer: kernel initializer, 'default he_normal'.
    :param learning_rate: learning rate for the model, default 1e-4.
    :param loss: loss function, default 'binary_crossentropy'.
    :param metrics: evaluation metrics, default ['accuracy'].
    :param output_activation: The activate function for the output layer, default 'sigmoid'.
    :return: The model.
    :raise AssertionError: Raises error if n < 1.
    """
    assert n >= 1, f'n need to be greater or equal to 1, not {n}.'

    inputs = Input(input_size)
    prev_layers = []

    # Up and down filters (without n-1 or n+1)
    filters = [64 * 2 ** i for i in range(n - 2)]
    print(filters)
    # Bridge filters (n-1, n+1 and n)
    filter_prev_n = 64 * 2 ** (n - 2)
    filter_n = 2 * filter_prev_n

    outputs = inputs

    # Down layers (1 to n-2)
    for f in filters:
        cov = Conv2D(f, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(outputs)
        cov = Conv2D(f, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(cov)

        prev_layers.append(cov)

        outputs = MaxPooling2D(pool_size=(2, 2))(cov)

    # Bridge n-1
    cov_prev_n = Conv2D(filter_prev_n, 3, activation=activation, padding=padding, 
                        kernel_initializer=kernel_initializer)(outputs)
    cov_prev_n = Conv2D(filter_prev_n, 3, activation=activation, padding=padding, 
                        kernel_initializer=kernel_initializer)(cov_prev_n)
    drop_prev_n = Dropout(0.5)(cov_prev_n)
    pool_prev_n = MaxPooling2D(pool_size=(2, 2))(drop_prev_n)

    # Bridge n
    cov_n = Conv2D(filter_n, 3, activation=activation, padding=padding, 
                   kernel_initializer=kernel_initializer)(pool_prev_n)
    cov_n = Conv2D(filter_n, 3, activation=activation, padding=padding, 
                   kernel_initializer=kernel_initializer)(cov_n)
    drop_n = Dropout(0.5)(cov_n)

    # Bridge n+1
    up_post_n = Conv2D(filter_prev_n, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)\
            (UpSampling2D(size=(2, 2))(drop_n))
    merge_post_n = concatenate([drop_prev_n, up_post_n], axis=3)
    cov_post_n = Conv2D(filter_prev_n, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)\
        (merge_post_n)
    outputs = Conv2D(filter_prev_n, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)\
        (cov_post_n)

    # Up layers (n+2 to 2n-1)
    for f, l in zip(reversed(filters), reversed(prev_layers)):
        up = Conv2D(f, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)\
                (UpSampling2D(size=(2, 2))(outputs))
        merge = concatenate([l, up], axis=3)
        cov = Conv2D(f, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge)
        outputs = Conv2D(f, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(cov)

    # Result
    outputs = Conv2D(2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(outputs)
    outputs = Conv2D(1, 1, activation=output_activation)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    return model
