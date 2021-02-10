from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from typing import Tuple, Optional


def _down_layers(input, filter, activation, padding, kernel_initializer):
    cov = Conv2D(filter, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(input)
    return Conv2D(filter, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(cov)


def _up_layers(input, skip, filter, activation, padding, kernel_initializer):
    up = Conv2D(filter, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer) \
        (UpSampling2D(size=(2, 2))(input))
    merge = concatenate([skip, up], axis=3)
    return _down_layers(merge, filter, activation, padding, kernel_initializer)


def unet_n(n: int = 5, input_size: Tuple[int, int, int] = (256, 256, 1), pretrained_weights: Optional[str] = None,
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

    # Bridge filters (n-1, n+1 and n)
    filter_prev_n = 64 * 2 ** (n - 2)
    filter_n = 2 * filter_prev_n

    outputs = inputs

    # Down layers (1 to n-2)
    for f in filters:
        cov = _down_layers(outputs, f, activation, padding, kernel_initializer)
        prev_layers.append(cov)
        outputs = MaxPooling2D(pool_size=(2, 2))(cov)

    # Bridge n-1
    cov_prev_n = _down_layers(outputs, filter_prev_n, activation, padding, kernel_initializer)
    drop_prev_n = Dropout(0.5)(cov_prev_n)
    pool_prev_n = MaxPooling2D(pool_size=(2, 2))(drop_prev_n)

    # Bridge n
    cov_n = _down_layers(pool_prev_n, filter_n, activation, padding, kernel_initializer)
    drop_n = Dropout(0.5)(cov_n)

    # Bridge n+1
    outputs = _up_layers(drop_n, drop_prev_n, filter_prev_n, activation, padding, kernel_initializer)

    # Up layers (n+2 to 2n-1)
    for l, f in zip(reversed(prev_layers), reversed(filters)):
        outputs = _up_layers(outputs, l, f, activation, padding, kernel_initializer)

    # Result
    outputs = Conv2D(2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(outputs)
    outputs = Conv2D(1, 1, activation=output_activation)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    return model
