https://drive.google.com/drive/folders/1a8JduBfbPb78Yl1KOc0ny31Oy4iLYo6S?usp=sharing

# Street View Image View Direction Recognition

In this repo I used the [Mapillarly Places Dataset](https://www.mapillary.com/dataset/places) to collect Street View images annoted with their view directions. 

On that I trained a EfficientNetB2-Model using Transfer Learning with Imagenet weights.

I took some code from [this](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)  keras tutorial on transfer learning.

### Procedure and repo structure

To start with I analysed the data an collected all non-pano images in the analyse_data.ipynb. Then I disregarded a lot of images facing forward to make the classes more balances and shuffled all images. I split the data into 80% Train, 10% Validation and 10% Test Data.

This is the distribution of classes in the datasets.
![Datasets](Imgs/Training%20Split.jpg)

create_ds.py was used to resize the image to 260 x 260, which is used by the B2 Model.

ds_generator.py contains all the code to generator the Tensorflow Datasetes for traingin/testing.

models.py contains all the code for the models.

To work with this Repo you would need to store the Places Data locally in a Folder "Data/" and the resized images into "Our_Data/".

In the "Training Notebooks" folder you can find all the notebooks used to train on colab/kaggle.

### Results

This are the results of Training.

![History](Imgs/training_history.jpg)
After 27 Epochs we achieved a Testing Accuracy of 95.7%.

These are some predictions:
![Predictions](Imgs/Predictions_high_resolution.jpg)
The real labels are written first and the model predictions in the bracket. You can see by the red marked pictuer that the data is a bit noisy and contains some false labels.


This is the confusian Matrix:
![Confusion Matrix](Imgs/confusion_matrix.jpg)


## Model

You can the model in the "models" folder. There is one version with and without a image augmentation head.

You can find the the logs also [here](https://tensorboard.dev/experiment/DxlFeRTKRhyKl3oAQbY1dw/#scalars).