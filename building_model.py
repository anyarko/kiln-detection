import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (Concatenate, Conv2DTranspose, MaxPooling2D, SeparableConv2D, UpSampling2D)
from tensorflow.keras.optimizers import Adam

from utils import *





def build_model(model_name, learning_rate, activation_function, loss_function, upsampling_interpolation, weight_regularizer):
    image_dim = (300,300,3)
    input_layer = Input(shape=image_dim, name='original_image')

    # Downsampling layers
    conv1 = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=activation_function, name='first_conv')(input_layer)
    conv2 = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=activation_function, name='second_conv')(conv1)
    pool = MaxPooling2D(pool_size=(2,2), strides=2, name='first_pool')(conv2)

    conv3 = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', activation=activation_function, name='third_conv')(pool)
    conv4 = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', activation=activation_function, name='fourth_conv')(conv3)
    pool = MaxPooling2D(pool_size=(2,2), strides=2, name='second_pool')(conv4)

    conv5 = SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', activation=activation_function, name='fifth_conv')(pool)
    conv6 = SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', activation=activation_function, name='sixth_conv')(conv5)
    pool = MaxPooling2D(pool_size=(2,2), strides=2, name='third_pool')(conv6)

    conv7 = SeparableConv2D(filters=512, kernel_size=(3,3), padding='same', activation=activation_function, name='seventh_conv')(pool)
    conv8 = SeparableConv2D(filters=512, kernel_size=(3,3), padding='same', activation=activation_function, name='eighth_conv')(conv7)
    pool = MaxPooling2D(pool_size=(2,2), strides=2, name='fourth_pool')(conv8)

    conv = SeparableConv2D(filters=1024, kernel_size=(3,3), padding='same', activation=activation_function, name='ninth_conv')(pool)
    conv = SeparableConv2D(filters=1024, kernel_size=(3,3), padding='same', activation=activation_function, name='tenth_conv')(conv)

    # Upsampling Layers with skip connections
    upsample = UpSampling2D(name='first_upsample', interpolation=upsampling_interpolation)(conv)
    convt = Conv2DTranspose(filters=1024, kernel_size=(2,2), activation=activation_function, name='first_conv_transpose', kernel_regularizer=weight_regularizer)(upsample)
    concat1 = Concatenate(name='first_concatenation')([conv8, convt])
    conv9 = SeparableConv2D(filters=512, kernel_size=(3,3), padding='same', activation=activation_function, name='first_up_conv')(concat1)
    conv10 = SeparableConv2D(filters=512, kernel_size=(3,3), padding='same', activation=activation_function, name='second_up_conv')(conv9)

    upsample = UpSampling2D(name='second_upsample', interpolation=upsampling_interpolation)(conv10)
    convt = Conv2DTranspose(filters=512, kernel_size=(2,2), activation=activation_function, name='second_conv_transpose', kernel_regularizer=weight_regularizer)(upsample)
    concat = Concatenate(name='second_concatenation')([conv6, convt])
    conv11 = SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', activation=activation_function, name='third_up_conv')(concat)
    conv12 = SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', activation=activation_function, name='fourth_up_conv')(conv11)

    upsample = UpSampling2D(name='third_upsample', interpolation=upsampling_interpolation)(conv12)
    concat = Concatenate(name='third_concatenation')([conv4, upsample])
    conv13 = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', activation=activation_function, name='fifth_up_conv')(concat)
    conv14 = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', activation=activation_function, name='sixth_up_conv')(conv13)

    upsample = UpSampling2D(name='fourth_upsample', interpolation=upsampling_interpolation)(conv14)
    concat = Concatenate(name='fourth_concatenation')([conv2, upsample])
    conv15 = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=activation_function, name='seventh_up_conv')(concat)
    conv16 = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=activation_function, name='eighth_up_conv')(conv15)

    final = SeparableConv2D(filters=3, kernel_size=(3,3), padding='same', activation='softmax', name='final_conv_masker')(conv16)

    model =  Model(inputs=input_layer, outputs=final, name=model_name)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function)

    return model


# def build_model(model_name, learning_rate, activation_function, loss_function, upsampling_interpolation, weight_initializer):
#     image_dim = (300,300,3)
#     input_layer = Input(shape=image_dim, name='orginal_image')

#     # Downsampling layers
#     conv1 = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=activation_function, name='first_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(input_layer)
#     conv2 = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=activation_function, name='second_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(conv1)
#     pool = MaxPooling2D(pool_size=(2,2), strides=2, name='first_pool')(conv2)

#     conv3 = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', activation=activation_function, name='third_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(pool)
#     conv4 = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', activation=activation_function, name='fourth_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(conv3)
#     pool = MaxPooling2D(pool_size=(2,2), strides=2, name='second_pool')(conv4)

#     conv5 = SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', activation=activation_function, name='fifth_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(pool)
#     conv6 = SeparableConv2D(filters=256, kernel_size=(3,3), padding='same', activation=activation_function, name='sixth_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(conv5)

#     # Upsampling Layers with skip connections
#     upsample = UpSampling2D(name='first_upsample', interpolation=upsampling_interpolation)(conv6)
#     concat1 = Concatenate(name='first_concatenation')([conv4, upsample])
#     conv7 = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', activation=activation_function, name='first_up_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(concat1)
#     conv8 = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', activation=activation_function, name='second_up_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(conv7)

#     upsample = UpSampling2D(name='second_upsample', interpolation=upsampling_interpolation)(conv8)
#     concat = Concatenate(name='second_concatenation')([conv2, upsample])
#     conv9 = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=activation_function, name='third_up_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(concat)
#     conv10 = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', activation=activation_function, name='fourth_up_conv', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer)(conv9)

#     final = SeparableConv2D(filters=1, kernel_size=(3,3), padding='same', activation=activation_function, name='final_conv_masker', depthwise_initializer=weight_initializer, pointwise_initializer=weight_initializer, use_bias=False)(conv10)

#     model =  Model(inputs=input_layer, outputs=final, name=model_name)

#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss=loss_function)

#     return model


def run_experiment(model_name, learning_rate, activation, loss_function, \
        upsampling_interpolation, weight_regularizer, epochs):
    train_gen, validation_gen = setup_training_variables()
    with tf.device('/GPU:0'):
        model = build_model(model_name, learning_rate, activation, loss_function, \
            upsampling_interpolation, weight_regularizer)
        model.summary()
        history = model.fit(train_gen,
                            steps_per_epoch=4,
                            epochs=epochs,
                            validation_data=validation_gen,
                            validation_steps=1,
                            verbose=1)

    save_histories = 'histories//' + model_name + '.pkl'
    with open(save_histories, 'wb') as file:
        pickle.dump(history.history, file)


def run_grid_search():
    activation_functions = ['sigmoid', 'selu', 'elu', 'relu','tanh']
    learning_rates = [0.01, 0.001, 0.0001]
    weight_regularizers = [None, 'l1', 'l2']
    upsampling_interpolations = ['nearest', 'bicubic', 'bilinear']
    loss_name = 'psnr'
    loss_function = lambda pred, true: -1*tf.image.psnr(true, pred, max_val=1)

    for activation_f in activation_functions:
        for lr in learning_rates:
            for wr in weight_regularizers:
                for interpolation_m in upsampling_interpolations:
                    model_name = f'unet_large_all_india_2020_{wr}_{activation_f}_{loss_name}_{lr}_{interpolation_m}'
                    run_experiment(model_name, lr, activation_f, loss_function, interpolation_m, wr, epochs=25)


def setup_training_variables():
    all_india_2020 = load_tiff_paths('tiffs//all_india_2020')
    area_of_interest = load_shapefile('shapefiles//india_kilns_classified_v3//india_kiln_types_v3.shp')
    weights = get_tiff_selection_probs(tiff_paths=all_india_2020,validation_tiff_num=13,shapefile=area_of_interest)
    train_gen = image_data_generator(batch_size=4, is_training=True, validation_tiff_num=13, \
                    weights=weights, tiff_paths=all_india_2020, shapefile=area_of_interest, trio=True)
    validation_gen = image_data_generator(batch_size=2, is_training=False, validation_tiff_num=13, \
                    weights=weights, tiff_paths=all_india_2020, shapefile=area_of_interest, trio=True)
    return train_gen, validation_gen


class CustomCallback(Callback):
    def __init__(self, validation_gen):
        super(CustomCallback, self).__init__()
        self.validation_gen = validation_gen

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            inputs, masks = next(self.validation_gen)
            show_predictions(self.model, inputs, masks, save=True, epoch=epoch, backend='PDF')

        if epoch % 200 == 0:
            os.makedirs(f'networks//{self.model.name[:10]}//{self.model.name}//', exist_ok=True)
            self.model.save_weights(f'networks//{self.model.name[:10]}//{self.model.name}//{self.model.name}_epoch_{epoch}.h5')
        return logs



def main():
    tf.random.set_seed(123)

    train_gen, validation_gen = setup_training_variables()
    weight_regularizer = 'l1'
    activation_function = 'elu'
    loss_name = 'cce'
    cce = tf.keras.losses.CategoricalCrossentropy()
    epsilon = tf.constant(0.000001)
    loss_function = lambda y_true, y_pred: cce(y_true+epsilon, y_pred+epsilon)
    # loss_function = lambda y_true, y_pred: -1*tf.image.psnr(y_true, y_pred, max_val=1)
    # loss_function = lambda y_true, y_pred: -1*tf.image.ssim(y_true, y_pred, max_val=1.0)
    # tf.random.set_seed(269)
    # weight_initializer = tf.keras.initializers.GlorotUniform(seed=269)
    learning_rate = 0.0001
    upsampling_interpolation = 'bilinear'

    model_name = f'unet_large_3_mask_all_india_2020_{weight_regularizer}_{activation_function}_{loss_name}_{learning_rate}_{upsampling_interpolation}'

    custom_callback = CustomCallback(validation_gen)

    with tf.device('/GPU:0'):
        model = build_model(model_name, learning_rate, activation_function, loss_function, \
            upsampling_interpolation, weight_regularizer)

        model.load_weights(f'networks/{model_name[:10]}/{model_name}/{model_name}_epoch_57400.h5')
        model.summary()
        history = model.fit(train_gen,
                            steps_per_epoch=4,
                            epochs=80000,
                            initial_epoch=57402,
                            verbose=1,
                            callbacks=[custom_callback])
    save_history = model_name + '.pkl'
    with open(save_history, 'wb') as file:
        pickle.dump(history.history, file)

if __name__ == '__main__':
    main()
    