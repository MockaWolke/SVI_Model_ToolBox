import tensorflow as tf
import pandas as pd
import numpy as np

def get_df():
    df = pd.read_feather("training.df").set_index("index")

    train_df = df.loc[df["ds_type"]=="train"]
    val_df = df.loc[df["ds_type"]=="val"]
    test_df = df.loc[df["ds_type"]=="test"]

    return train_df,val_df,test_df


def mapping(img1,img2,img3,img4,img5,label):

    img1 = tf.io.decode_jpeg(tf.io.read_file(img1), channels=3)
    img2 = tf.io.decode_jpeg(tf.io.read_file(img2), channels=3)
    img3 = tf.io.decode_jpeg(tf.io.read_file(img3), channels=3)
    img4 = tf.io.decode_jpeg(tf.io.read_file(img4), channels=3)
    img5 = tf.io.decode_jpeg(tf.io.read_file(img5), channels=3)

    img_seq = tf.stack([img1,img2,img3,img4,img5])

    label = tf.one_hot(label,2)

    return img_seq, label


def get_ds(df,batch_size, kind = "train",colab = False): 
    assert kind in ["train","test","val"]

    paths = np.array([i for i in df.Paths.apply(np.array).values])

    if colab:
        func = lambda s: s[s.find("images/")+7:]
        paths = np.array([[func(s) for s in row]  for row in paths])
    else:
        paths = "../../" + paths

    labels = (df.view_direction == "Sideways").apply(int).values

    ds = tf.data.Dataset.from_tensor_slices((paths[:,0],paths[:,1],paths[:,2],paths[:,3],paths[:,4],labels))
    ds = ds.map(mapping)
    ds = ds.batch(batch_size=batch_size, drop_remainder= kind!='test')
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_train_and_val(batch_size = 64,colab = False):

    train_df,val_df,test_df = get_df()

    train_ds = get_ds(train_df,batch_size=batch_size,kind="train",colab = colab)
    val_ds = get_ds(val_df,batch_size = batch_size,kind="val",colab = colab)

    return train_ds,val_ds

def get_test_ds(batch_size = 64,colab = False):

    train_df,val_df,test_df = get_df()

    test_ds = get_ds(test_df,batch_size=batch_size,kind="test",colab = colab)

    return test_ds


