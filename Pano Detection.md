# Panoramic Image Detection of SVI


## Data Set Generation

These are examplary Pano Images from the Mapillary Places Dataset:

![Day](pano_detection_task\Imgs\Examples_of_Pano_Images.jpg)


First of all I deletet every Image, of which Series contains instances of both Normal and Pano Images. Some examples can be found here:

![Imgs](pano_detection_task\Imgs\Sequences_with_different_View_Pano_types.jpg)


Then I chose and apropriate split of the Citys for the Validation and Test Data:

![Citys](pano_detection_task\Imgs\Citys_Split.jpg)

Additionally I randomly droped to Normal images to get a more even Split for training. This was the distribution I then worked with:

![Split](pano_detection_task\Imgs\Training_Split.jpg)

For Training I resized the Images to 260 x 260.

The whole dataset for training, validataion & testing can be downloaded from [here.](needs to be changed)


## Model Training

I tried out Transfer Learning with EfficientNet 0,1 & 2 and MobileNetV3 Small & Big. For all architectures I unfroze the last 20 Layers after 3 consecutive epochs with out improvement in the Val Accuracy.

Plots of all the runs can be found [here](pano_detection_task\Imgs\Training_Logs). **Please Note** that all plots have a error in the title. It says "View Direction Task" instead of "Pano Detection Task".

## Results

The Model with the highest Validation Accuracy was the EfficientNetB0, which achieved a staggering 99.9% Accuracy on the Test Dataset. 

Some examplary predicitons:
![Preds](pano_detection_task\Imgs\Predictions.jpg)

The wrong predictions:

![false_labels](pano_detection_task\Imgs\Wrong_Predictions.jpg).
 
Confusion Matrix

![con](pano_detection_task\Imgs\Confusion_Matrix.jpg)

## Models

The model can be loaded either [with the inital augmentation block](pano_detection_task\final_model) or [without](pano_detection_task\final_model_without_aug).

Both models take an Input of 260 x 260.


