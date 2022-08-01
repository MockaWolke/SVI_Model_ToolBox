# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pano_generator
import os

# %%
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

def transfer_model(number_classes,architecture = tf.keras.applications.efficientnet.EfficientNetB2,name=None):

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

# %%
rescale = False
inputs = tf.keras.Input([260,260,3])

x = img_augmentation(inputs) # apply image augmentaion

if rescale:
    x = tf.keras.layers.Resizing(224,224) (x)

model =  tf.keras.applications.efficientnet.EfficientNetB2(
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
outputs = layers.Dense(2, activation="softmax", name="pred",kernel_regularizer=tf.keras.regularizers.L2())(x)

# Compile
model = tf.keras.Model(inputs, outputs, name="b2_freez")



# %%
model.compile(tf.keras.optimizers.Adam(1e-4),loss = tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])

# %%
log_dir =  "training_results/logs/" + model.name + "/"
checkpoint_path =  "training_results/weights/" + model.name + "/"
last_model = "training_results/last_model/" + model.name + "/"

try:
    os.makedirs(log_dir)
    os.makedirs(checkpoint_path)
    os.makedirs(last_model)
except:
    pass


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor='val_accuracy',
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    mode='auto') 

# %%
train_ds , val_ds = pano_generator.get_train_and_val()

if rescale:
    train_ds = train_ds.map(lambda x,y : (x/255-1,y))
    val_ds = val_ds.map(lambda x,y : (x/255-1,y))

# %%
with tf.device('/device:GPU:0'):
    model.fit(train_ds,epochs=20,validation_data=val_ds,callbacks=[tensorboard_callback,cp_callback,early_stopping_callback])

# %%
# save last model
model.save(last_model+"last_model")
# load best weights
model.load_weights(checkpoint_path)

# %%
model = unfreeze_model(model,n_layers = 20)
model.compile(tf.keras.optimizers.Adam(1e-4),loss = tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])

# %%
model._name = "B2_All"
log_dir =  "training_results/logs/" + model.name + "/"
checkpoint_path =  "training_results/weights/" + model.name + "/"
last_model = "training_results/last_model/" + model.name + "/"

try:
    os.makedirs(log_dir)
    os.makedirs(checkpoint_path)
    os.makedirs(last_model)
except:
    pass

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor='val_accuracy',
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    mode='auto') 

model.fit(train_ds,epochs=60,validation_data=val_ds,callbacks=[tensorboard_callback,cp_callback,early_stopping_callback , reduce_lr])



