import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--crop',type=bool)
parser.add_argument("--name",type=str)
parser.add_argument("--initial_lr",type=float ,default=1e-3)
parser.add_argument("--later_lr",type=float ,default=1e-4)
parser.add_argument("--testing",type=bool ,default=False)
parser.add_argument("--initial_epochs",type=int ,default=20)
parser.add_argument("--later_epochs",type=int ,default=60)
parser.add_argument("--batch_size",type=int ,default=64)
parser.add_argument("--n_layers",type=int ,default=20)



def mapping(path,label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    label = tf.one_hot(label,2)
    return image, label

def get_ds(df,bs):
    ds = tf.data.Dataset.from_tensor_slices((df.local_path.values,(df.view_direction=="Sideways").apply(int).values))

    ds = ds.map(mapping).shuffle(1000).batch(bs).prefetch(tf.data.AUTOTUNE)

    return ds

img_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=0.15),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def frozen_model(crop,name):

    inputs = tf.keras.Input([260,260,3])

    if crop:
        x = tf.keras.layers.RandomCrop(224,224) (inputs)
    else:
        x = tf.keras.layers.Resizing(224,224) (inputs)

    x = img_augmentation(x)

    base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor= x)

    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="pred",kernel_regularizer=tf.keras.regularizers.L2())(x)

    model = tf.keras.Model(inputs, outputs, name=name)

    return model

def unfreeze_model(model,n_layers = 20):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-n_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


    return model

def get_callbacks(model,lr_reduder = False,stop_patience = 3):

    log_dir =  "cnn_base/logs/" + model.name + "/"
    checkpoint_path =  "cnn_base/weights/" + model.name + "/"
    last_model = "cnn_base/last_model/" + model.name + "/"

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
    plt.suptitle(f"{name} View Direction Training CNN Base, Top Val Acc = {max(val_acc):.2f}")
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
    plt.savefig(f"cnn_base/plots/{name}.jpg")


if __name__ == "__main__":
    print("\n"*2,"Working Directory: ",os.getcwd(),"\n"*2)
    args = parser.parse_args()

    df = pd.read_feather("cleaned_images.df").set_index("key")
    train_df = df.loc[df.ds_type=="train"]
    val_df = df.loc[df.ds_type=="val"]
    train_ds = get_ds(train_df,args.batch_size)
    val_ds = get_ds(val_df,args.batch_size)

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

    model = frozen_model(args.crop, MODEL_NAME_STEM + "_Frozen")
    

    model.compile(tf.keras.optimizers.Adam(args.initial_lr),loss =  tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])
    callbacks = get_callbacks(model,lr_reduder=False,stop_patience=3)


    hist1 = model.fit(train_ds,
            epochs=INITIAL_EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks)
    initial_epochs_runned = len(hist1.history["accuracy"])
    
    # load best weights
    model.load_weights(f"cnn_base/weights/{model.name}/cp.ckpt")


    model = unfreeze_model(model,n_layers = args.n_layers)
    model._name = f"{MODEL_NAME_STEM}_All"

    model.compile(tf.keras.optimizers.Adam(args.later_lr),loss = tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])
    callbacks = get_callbacks(model,lr_reduder=True,stop_patience=10)

    hist2 = model.fit(train_ds,
        epochs=LATER_EPOCHS + initial_epochs_runned,
        validation_data=val_ds,
        callbacks=callbacks,initial_epoch = initial_epochs_runned)

    model.save(f"cnn_base/last_model/{model.name}/model")
    
    plotting(hist1,hist2,MODEL_NAME_STEM)