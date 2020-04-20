# Covid-19-Classifier
Classify COVID-19 CT images using ResNeXt and other networks

I have used the data given at https://github.com/UCSD-AI4H/COVID-CT to train a neural network. After 300 Epochs, the accuracy was above 98% with a confidence of > 95%.

Accuracy was calculated by counting the True Positve as well as the True Negative classifications and divide the sum by the total number of images calssified.

Confidence is calculated by averaging the network prediction for each classification

The code is based on ResNeXt50_32x4d with an input image size of 512x512
On a NVIDIA GTX1660, one epoch running through the ~700 images takes about 1 minute, the 300 epochs took about 32 hours
