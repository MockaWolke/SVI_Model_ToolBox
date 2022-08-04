import pandas as pd
import tensorflow_datasets as tfds
import numpy as np
import tqdm
import tensorflow as tf


def mapping(path,label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    label = tf.one_hot(label,2)
    return image, label



def get_ds(): 
    df = pd.read_csv("Relabeled_Test_DS/cleaned_data.csv",index_col=0)
    image_paths = "Relabeled_Test_DS/" + df["Label"] +"/" + df.index + ".jgp"
    labels = df["Label"].apply(lambda x: int(x=="Night"))
    ds = tf.data.Dataset.from_tensor_slices((image_paths,labels))
    ds = ds.map(mapping)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


get_ds()