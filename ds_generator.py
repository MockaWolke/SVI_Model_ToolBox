"""
Generate tf Datasets for training
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

label_dict = {"Forward":0, "Backward":1,"Sideways":2}
batch_size = 64

def get_csv():
    df = pd.read_csv("data.csv")

    train_df = df.loc[df["ds_type"]=="train"]
    val_df = df.loc[df["ds_type"]=="val"]
    test_df = df.loc[df["ds_type"]=="test"]

    return train_df,val_df,test_df

def mapping(path,label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    label = tf.one_hot(label,3)
    return image, label



def get_ds(df, kind = "train"): 
    assert kind in ["train","test","val"]

    path = f"Our_Data/{kind}/"
    imgs = df["key"] + ".jpg"

    #assert all([img in os.listdir(path) for img in imgs]), "Immage not found"

    image_paths = path + imgs

    labels = df["view_direction"].map(label_dict)

    ds = tf.data.Dataset.from_tensor_slices((image_paths,labels))

    ds = ds.map(mapping)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_train_and_val():

    train_df,val_df,test_df = get_csv()

    train_ds = get_ds(train_df,kind="train")
    val_ds = get_ds(val_df,kind="val")

    return train_ds,val_ds

def get_test_ds():

    train_df,val_df,test_df = get_csv()

    test_ds = get_ds(test_df,kind="test")

    return test_ds
