# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pano_generator
import os
import argparse   
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str)
parser.add_argument('--drop_rate',type=float,default=0.2)
parser.add_argument("--name",type=str)
parser.add_argument("--n_layers",type=int ,default=20)
parser.add_argument("--batch_size",type=int ,default=64)
parser.add_argument("--initial_epochs",type=int ,default=20)
parser.add_argument("--later_epochs",type=int ,default=60)
parser.add_argument("--initial_lr",type=float ,default=1e-3)
parser.add_argument("--later_lr",type=float ,default=1e-4)
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
        filepath=f"{checkpoint_path}cp.ckpt",
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

def plotting(history,history_fine,name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    initial_epochs = len(acc)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    x = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.suptitle(f"{name} View Direction Training, Top Val Acc = {max(val_acc):.2f}")
    plt.subplot(2, 1, 1)
    plt.plot(x,acc, label='Training Accuracy')
    plt.plot(x,val_acc, label='Validation Accuracy')
    plt.ylim([0.85, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xticks(x,x)

    plt.subplot(2, 1, 2)
    plt.plot(x,loss, label='Training Loss')
    plt.plot(x,val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.xticks(x,x) 
    plt.savefig(f"training_results/plots/{name}.jpg")



if __name__ == "__main__":
    print("\n"*2,"Working Directory: ",os.getcwd(),"\n"*2)


    args = parser.parse_args()
    

    train_ds , val_ds = pano_generator.get_train_and_val(args.batch_size)
    MODEL_NAME_STEM = args.name
    INITIAL_EPOCHS = args.initial_epochs
    LATER_EPOCHS = args.later_epochs

    if args.testing:
        print("Testing")
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(2)
        MODEL_NAME_STEM = MODEL_NAME_STEM + "_testing"
        INITIAL_EPOCHS = 2
        LATER_EPOCHS = 2


    model = eval(args.model)
    model = frozen_model(model, MODEL_NAME_STEM + "_Frozen", args.drop_rate,args.input_size)
    model.compile(tf.keras.optimizers.Adam(args.initial_lr),loss = tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])
    callbacks = get_callbacks(model,lr_reduder=False,stop_patience=3)

    hist1 = model.fit(train_ds,
            epochs=INITIAL_EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks)
    initial_epochs_runned = len(hist1.history["accuracy"])
    
    # load best weights
    model.load_weights(f"training_results/weights/{model.name}/cp.ckpt")


    model = unfreeze_model(model,n_layers = args.n_layers)
    model._name = f"{MODEL_NAME_STEM}_All"

    model.compile(tf.keras.optimizers.Adam(args.later_lr),loss = tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])
    callbacks = get_callbacks(model,lr_reduder=True,stop_patience=10)

    hist2 = model.fit(train_ds,
        epochs=LATER_EPOCHS + initial_epochs_runned,
        validation_data=val_ds,
        callbacks=callbacks,initial_epoch = initial_epochs_runned)

    model.save(f"training_results/last_model/{model.name}/model")
    
    plotting(hist1,hist2,MODEL_NAME_STEM)

