"""
Models for training
"""

from platform import architecture
import tensorflow as tf
import tensorflow.keras.layers as layers

# some img_augmentation
img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def transfer_model(number_classes,loss,architecture = tf.keras.applications.efficientnet.EfficientNetB2,name=None):

    inputs = tf.keras.Input([260,260,3])

    x = img_augmentation(inputs) # apply image augmentaion

    model = architecture(
        include_top=False,
        weights='imagenet',
        input_tensor= x)

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(number_classes, activation="softmax", name="pred",kernel_regularizer=tf.keras.regularizers.L2())(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name=name)

    return model


def unfreeze_model(model,n_layers = 20):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-n_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True


    return model