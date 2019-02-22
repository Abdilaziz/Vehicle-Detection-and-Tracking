# Vehicle-Detection-and-Tracking
Detect and Track Vehicles in a Video

This project contains a Software Pipeline to detect vechicles in a video.
All code is in Final_VehicleDetectionPipeline.py

## Overall Process

Extract features from a 64x64 Image, and feed that data to a trained machine learning classifier.

The classifier can classify an Image as either vehicle, or non-vehicle.

With the classifier that works on 64x64 images, we can window through larger images and classify them as either vehicle or non-vehicle.

We can then filter out bad classifications by grouping up detections in the same area by creating a Heat Map.

## Discussion


![Spatial Binning of Colour](https://github.com/Abdilaziz/Vehicle-Detection-and-Tracking/blob/master/images/spatial_binning.jpg "Spatial Binning of Colour")




![Vehicle Image](https://github.com/Abdilaziz/Vehicle-Detection-and-Tracking/blob/master/images/vehicle_image.png "Vehicle Image")



![Non-Vehicle Image](https://github.com/Abdilaziz/Vehicle-Detection-and-Tracking/blob/master/images/non_vehicle_image.png "Non-Vehicle Image")


![Histogram of Oriented Gradients](https://github.com/Abdilaziz/Vehicle-Detection-and-Tracking/blob/master/images/HOG_Image.jpg "Histogram of Oriented Gradients")




![Windows Classified as Vehicles](https://github.com/Abdilaziz/Vehicle-Detection-and-Tracking/blob/master/images/classified_windows.png "Windows Classified as Vehicles")




![Heat Map of Detected Windows](https://github.com/Abdilaziz/Vehicle-Detection-and-Tracking/blob/master/images/HeatMap_image.jpg "Heat Map of Detected Windows")



![Final Output Image](https://github.com/Abdilaziz/Vehicle-Detection-and-Tracking/blob/master/images/final_output_image.png "Final Output Image")


Links:

Labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  

These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. 
