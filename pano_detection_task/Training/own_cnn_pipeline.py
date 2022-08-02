# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pano_generator
import os
import argparse   

parser = argparse.ArgumentParser()
parser.add_argument('--drop_rate',type=float,default=0.2)
parser.add_argument("--name",type=str)
parser.add_argument("--batch_size",type=int ,default=64)
parser.add_argument("--initial_epochs",type=int ,default=20)
parser.add_argument("--initial_lr",type=float ,default=1e-3)
parser.add_argument("--testing",type=bool ,default=False)
parser.add_argument("--input_size",type=int ,default=260)


img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def frozen_model(base,name,drop_rate,input_size):


    inputs = tf.keras.Input([260,260,3])

    x = img_augmentation(inputs)

    if input_size != 260:

        x = tf.keras.layers.Resizing(input_size,input_size) (x)

    base = base(
        include_top=False,
        weights='imagenet',
        input_tensor= x)

    base.trainable = False
    # Rebuild top
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = drop_rate
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred",kernel_regularizer=tf.keras.regularizers.L2())(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name=name)

    return model

def unfreeze_model(model,n_layers = 20):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-n_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


    return model
    

def get_callbacks(model,lr_reduder = False,stop_patience = 3):

    log_dir =  "training_results/logs/" + model.name + "/"
    checkpoint_path =  "training_results/weights/" + model.name + "/"
    last_model = "training_results/last_model/" + model.name + "/"

    try:
        os.makedirs(log_dir)
        os.makedirs(checkpoint_path)
        os.makedirs(last_model)
    except:
        pass

    callbacks = []

    callbacks.append( tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_best_only=True))

    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=stop_patience,
        verbose=1,
        mode='auto'))

    if  lr_reduder:

        callbacks.append( tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001))

    return callbacks



if __name__ == "__main__":

    args = parser.parse_args()
    

    train_ds , val_ds = pano_generator.get_train_and_val(args.batch_size)

    train_ds = train_ds.cache()
    val_ds = val_ds.cache() 

    MODEL_NAME_STEM = args.name
    INITIAL_EPOCHS = args.initial_epochs

    if args.testing:
        print("Testing")
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(2)
        MODEL_NAME_STEM = MODEL_NAME_STEM + "_testing"
        INITIAL_EPOCHS = 2
        LATER_EPOCHS = 2


    inputs = tf.keras.Input([260,260,3])

    x = img_augmentation(inputs)

    x = tf.keras.layers.Conv2D(32,9,strides=4) (x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU() (x)
    x = tf.keras.layers.MaxPool2D(2) (x)
    x = tf.keras.layers.Conv2D(64,7,strides=3) (x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU() (x)
    x = tf.keras.layers.MaxPool2D(2) (x)

    x = tf.keras.layers.Conv2D(64,3,strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU() (x)
    x = tf.keras.layers.GlobalMaxPooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.15, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred",kernel_regularizer=tf.keras.regularizers.L2())(x)
    model = tf.keras.Model(inputs, outputs, name=MODEL_NAME_STEM)


    model.compile(tf.keras.optimizers.Adam(args.initial_lr),loss = tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])
    callbacks = get_callbacks(model,lr_reduder=False,stop_patience=4)

    model.fit(train_ds,
            epochs=INITIAL_EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks)

    
    model.save(f"training_results/last_model/{model.name}/last_model")
    # load best weights
    model.load_weights(f"training_results/weights/{model.name}/")
