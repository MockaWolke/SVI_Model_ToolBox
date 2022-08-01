import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--df_path',type=str)
parser.add_argument('--output_path',type=str)
parser.add_argument('--input_size',type=int)
parser.add_argument("--with_test",type=int,default=1)

def mapping(path,key):
    global Image_shape

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [Image_shape,Image_shape])

    return image, key

def resize_data(df,path_out):

    image_paths = df["Path"] + "/images/" + df["key"] + ".jpg"

    ds = tf.data.Dataset.from_tensor_slices((image_paths,df["key"]))
    ds = ds.map(mapping,num_parallel_calls=tf.data.AUTOTUNE)

    for img, path in tqdm.tqdm(tfds.as_numpy(ds)):
        
        tf.keras.utils.save_img(path_out + path.decode("utf-8") + ".jpg", img)

if __name__ == "__main__":

    args = parser.parse_args()

    df = pd.read_csv(args.df_path)

    if args.output_path not in os.listdir():
        os.makedirs(args.output_path)
        os.makedirs(f"{args.output_path}/train/")
        os.makedirs(f"{args.output_path}/val/")

        if args.with_test:
            os.makedirs(f"{args.output_path}/test/")
    

    train_df = df.loc[df["ds_type"]=="train"]
    val_df = df.loc[df["ds_type"]=="val"]

  

    Image_shape = args.input_size

    resize_data(train_df,f"{args.output_path}/train/")
    resize_data(val_df,f"{args.output_path}/val/")

    if args.with_test:
        test_df = df.loc[df["ds_type"]=="test"]
        resize_data(test_df,f"{args.output_path}/test/")

    df.to_csv(f"{args.output_path}/data.csv")




