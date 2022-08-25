"""
Generate tf Datasets for training
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_csv():
    df = pd.read_csv("../view_directions_data.csv",index_col=0)

    train_df = df.loc[df["ds_type"]=="train"]
    val_df = df.loc[df["ds_type"]=="val"]
    test_df = df.loc[df["ds_type"]=="test"]

    return train_df,val_df,test_df

def mapping(path,label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    return image, label



def get_ds(df,batch_size, kind = "train"): 
    assert kind in ["train","test","val"]

    path = f"../VD_Data/{kind}/"
    imgs = df["key"] + ".jpg"

    image_paths = path + imgs

    labels = df["vd"].map(int)

    ds = tf.data.Dataset.from_tensor_slices((image_paths,labels))

    ds = ds.map(mapping)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_train_and_val(batch_size = 64,excemption = None):

    if excemption is None:
        train_df,val_df,test_df = get_csv()
        train_ds = get_ds(train_df,batch_size=batch_size,kind="train")
        val_ds = get_ds(val_df,batch_size = batch_size,kind="val")
    else: 
        df = pd.read_feather("../../view_directions_sequence/training/new_train_imgs.df").set_index("key")
        df["local_path"] = df.local_path.apply(lambda x: "../../view_directions_sequence" +x[2:])
        train_df = df.loc[df.ds_type=="train"].sample(frac=1)
        val_df = df.loc[df.ds_type=="val"].copy()
        train_ds = tf.data.Dataset.from_tensor_slices((train_df.local_path.values,(train_df.view_direction=="Sideways").apply(int).values)).map(mapping).batch(batch_size=batch_size, drop_remainder=True).shuffle(100).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((val_df.local_path.values,(val_df.view_direction=="Sideways").apply(int).values)).map(mapping).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_ds,val_ds

def get_test_ds(batch_size = 64):

    train_df,val_df,test_df = get_csv()

    test_ds = get_ds(test_df,batch_size=batch_size,kind="test")

    return test_ds
