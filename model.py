from keras.models import *
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam


def unet_n(pretrained_weights=None, input_size=(256, 256, 1), depth=5):

    inputs = Input(input_size)
    prev_layers = []

    # Up and down filters
    filters = [64 * 2 ** i for i in range(depth - 2)]

    # Bridge filters
    filter_prev_n = 64 * 2 ** (depth - 1)
    filter_n = 2 * filter_prev_n

    outputs = inputs.copy()

    # Down layers
    for f in filters:
        cov = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(outputs)
        cov = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cov)

        prev_layers.append(cov)

        outputs = MaxPooling2D(pool_size=(2, 2))(cov)

    # Bridge n-1
    cov_prev_n = Conv2D(filter_prev_n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(outputs)
    cov_prev_n = Conv2D(filter_prev_n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cov_prev_n)
    drop_prev_n = Dropout(0.5)(cov_prev_n)
    pool_prev_n = MaxPooling2D(pool_size=(2, 2))(drop_prev_n)

    # Bridge n
    cov_n = Conv2D(filter_n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_prev_n)
    cov_n = Conv2D(filter_n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cov_n)
    drop_n = Dropout(0.5)(cov_n)

    # Bridge n+1
    up_post_n = Conv2D(filter_prev_n, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
            (UpSampling2D(size=(2, 2))(drop_n))
    merge_post_n = concatenate([drop_prev_n, up_post_n], axis=3)
    cov_post_n = Conv2D(filter_prev_n, 3, activation='relu', padding='same', kernel_initializer='he_normal')\
        (merge_post_n)
    outputs = Conv2D(filter_prev_n, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cov_post_n)

    # Up layers
    for f, l in zip(reversed(filters), reversed(prev_layers)):
        up = Conv2D(f, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(outputs))
        merge = concatenate([l, up], axis=3)
        cov = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
        outputs = Conv2D(f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cov)

    # Exit
    outputs = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(outputs)
    outputs = Conv2D(1, 1, activation='sigmoid')(outputs)

    model = Model(input=inputs, output=outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    return model
