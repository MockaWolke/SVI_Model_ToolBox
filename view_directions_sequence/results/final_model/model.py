import tensorflow as tf
import os
import tensorflow.keras.layers as layers


def load_ensemble_part(weights):
    inputs = tf.keras.Input([260,260,3])

    img_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.RandomRotation(factor=0.15),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )

    x = tf.keras.layers.Resizing(224,224) (inputs)

    x = img_augmentation(x)

    base = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_tensor= x)

    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base.output)

    cnn =  tf.keras.Model(inputs,x)

    inputs = tf.keras.Input([5,260,260,3])
    x = layers.TimeDistributed(cnn)(inputs)

    x = layers.TimeDistributed( layers.Dropout(0.16892078783969278), name = "drop1") (x)
    
    x = layers.LSTM(192, name = "lstm")(x)

    x = layers.Dropout(0.2907156272060155,name ="drop2")(x)

    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred",kernel_regularizer=tf.keras.regularizers.L2())(x)


    model = tf.keras.Model(inputs,outputs)
    model.load_weights(weights)
    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model():
    inputs = tf.keras.Input([5,260,260,3])
    weight_dirs = [i+"/cp.cpkt" for i in os.listdir() if [i-2:]!='py']
    x = [load_ensemble_part(i)(inputs) for i in weight_dirs]
    x = tf.math.reduce_mean(x,axis = 0)
    ensemble = tf.keras.Model(inputs,x[:,1])
    ensemble.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return ensemble
