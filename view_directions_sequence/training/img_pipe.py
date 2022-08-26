import tensorflow as tf
import numpy as np
import pandas as pd
import seq_generator
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str)
parser.add_argument("--name",type=str)
parser.add_argument("--n_layers",type=int ,default=20)
parser.add_argument("--later_epochs",type=int ,default=60)
parser.add_argument("--testing",type=bool ,default=False)
parser.add_argument("--input_size",type=int ,default=260)
parser.add_argument("--aug_mode",type=int ,default=1)


def build_aug(mode,inp_shape):

    match(mode):

        case 1:

            m = tf.keras.models.Sequential([
                        tf.keras.layers.Resize(inp_shape,inp_shape),
                        tf.keras.layers.RandomRotation(factor=0.15),
                        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                        tf.keras.layers.RandomContrast(factor=0.1),
                    ],
                    name=f"img_augmentation_{mode}")

            return m

        case 2:

            m = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.RandomCrop(inp_shape,inp_shape),
                        tf.keras.layers.RandomRotation(factor=0.25),
                        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
                        tf.keras.layers.RandomContrast(factor=0.2),
                        tf.keras.layers.RandomBrightness(0.2),
                    ],
                    name=f"img_augmentation_{mode}")

            return m


        case 3:

            m = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.RandomCrop(inp_shape,inp_shape),
                        tf.keras.layers.RandomZoom(.2, .2),
                        tf.keras.layers.RandomRotation(factor=0.25),
                        tf.keras.layers.RandomTranslation(height_factor=0.25, width_factor=0.25),
                        tf.keras.layers.RandomContrast(factor=0.25),
                    ],
                    name=f"img_augmentation_{mode}")

            return m


def build_cnn(appli,aug):

    inputs = tf.keras.Input([260,260,3])

    x = aug(inputs)

    base = eval(appli)(
            include_top=False,
            weights='imagenet',
            input_tensor= x)

    base.trainable = False
    
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    x = tf.keras.layers.BatchNormalization()(x)

    return tf.keras.Model(inputs,x)

def build_model(hps,cnn):


    inputs = tf.keras.Input([5,260,260,3])
    x = layers.TimeDistributed(cnn)(inputs)

    drop1 = hps["drop1"]

    x = layers.TimeDistributed( layers.Dropout(drop1), name = "drop1") (x)

    hidden_units =  hps["hidden_units"]

    if hps["rnn"] == "lstm":
        x = layers.LSTM(hidden_units, name = "lstm")(x)
    else:
        x = layers.GRU(hidden_units, name = "gru") (x)

    drop2 = hps["drop2"]
    x = layers.Dropout(drop2,name ="drop2")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pred",kernel_regularizer=tf.keras.regularizers.L2())(x)

    learning_rate = hps["lr"]

    model = tf.keras.Model(inputs,outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model



def warm_up(cnn, n = 20):

    cnn.trainable = True
    for layer in cnn.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    for layer in cnn.layers[:-n]:

            layer.trainable = False

    return cnn


def get_callbacks(warm_path,log_dir):

    callbacks = []

    callbacks.append( tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        filepath= warm_path + "/cp.ckpt",
        verbose=1,
        save_weights_only=True,
        save_best_only=True))

    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
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
    plt.suptitle(f"{name} View Direction Training, Top Val Acc = {max(val_acc):.2f}")
    plt.subplot(2, 1, 1)
    plt.plot(x,acc, label='Training Accuracy')
    plt.plot(x,val_acc, label='Validation Accuracy')
    plt.ylim([min(x), 1])
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

    os.makedirs("training_results/plots", exist_ok= True)
    plt.savefig(f"training_results/plots/{name}.jpg")


if __name__ == '__main__':

    args = parser.parse_args()
    pars = json.load(open("tuner_run/best_hps.json"))
    print("\n"*2,"Working Directory: ",os.getcwd(),"\n"*2)

    train_ds , val_ds = seq_generator.get_train_and_val(batch_size=128)

    MODEL_NAME_STEM = args.name
    LATER_EPOCHS = args.later_epochs

    if args.testing:
        print("Testing")
        train_dY = train_ds.take(2)
        val_ds = val_ds.take(2)
        MODEL_NAME_STEM = MODEL_NAME_STEM + "_testing"
        LATER_EPOCHS = 2


    aug = build_aug(args.aug_mode,args.input_shape)
    cnn = build_cnn(args.model, aug)
    model = build_model(pars,cnn)

    hist1 = model.fit(train_ds, epochs=1, validation_data=val_ds)

    c_cold_path = f"training_results/weights/{MODEL_NAME_STEM}_Frez"
    os.makedirs(c_cold_path , exist_ok = True)

    model.save_weights(c_cold_path + "/cp.ckpt")

    # ----------------------------------------------------------

    cnn = warm_up(cnn)
    model = build_model(pars,cnn)
    model.load_weights(c_cold_path + "/cp.ckpt")

    c_warm_path = f"training_results/weights/{MODEL_NAME_STEM}_Warm"
    log_dir = f"training_results/logs/{MODEL_NAME_STEM}_Warm"
    os.makedirs(c_warm_path, exist_ok = True)
    os.makedirs(log_dir, exist_ok = True)
    cbs = get_callbacks(MODEL_NAME_STEM,log_dir)
    hist2 = model.fit(train_ds, epochs = LATER_EPOCHS, validation_data=val_ds, callbacks = cbs)

    save_path = f"training_results/last_model/{MODEL_NAME_STEM}/model"
    os.makedirs(save_path, exist_ok = True)
    model.save(save_path)

    plotting(hist1,hist2,MODEL_NAME_STEM)

