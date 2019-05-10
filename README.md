# [WIP] GANonymizerV2
## What is the GANonymizer and GANonymizerV2?
GANonymizer is the Image Anonymization method for urban images.  
GANonymizer anonymize an urban image by removing privacy objects such as people and cars from the urban image completely.  
If you input the urban image to GANonymizer, GANonymizer give you the  annonymized image.  
This GANonymizerV2 is the updated version of [GANonymizer](https://github.com/tanimutomo/ganonymizer).

## Detail of the method
GANonymizerV2 is consisted of three modules.  
The first module is a Privacy Detection Module.  
We use the existing semantic segmentation method, [DeepLabV3](https://github.com/fregu856/deeplabv3) as the privacy detection module.  
Second module is the Object Shadow Detection Module.  
In the Object Shadow Detection Module, the shadow area detected using superpixel segmentation and superpixel's relational adjacent graph.  
Third module is the Background Generation Module.  
In order to genertate the background of the privacy object area, we use the state-of-the-art image inpainting method, [EdgeConnect](https://github.com/knazeri/edge-connect).

## Demo and Paper
Please try our [Demo](https://bacchus.ht.sfc.keio.ac.jp/ganonymizerv2).  
Please check our paper for more details.


