# Typhoon prediction and ENSO analysis
- Running *Train_Forecast_model.py* trains the forecast model and save weights within given epoch intervals
- *Evaluate_forecast_model.py* can be used evaluate the performance of a trained model.
- *Image_grade_classifier* is used to classify the intensity/grade
- *library* contains more functions and classed which are used by the files above
- *requirements* are suggestions for installing the required modules

In the image below, a sequence of images(512x512 p) which makes up a typhoon storm is shown as a gif file.
![Example of typhoon data sequence. Contains a series of 512x512 images](Imaging/infrared_animation_2025-05-10_23-12-02.gif)

In the image below is a forecasts prediction compared to the ground truth of a storm
![Example of typhoon prediction of a data sequence.](Imaging/202402_8.261.png)
