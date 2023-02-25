# VGG16-implementation-for-Classifying-Anatomical-Structures-in-2D-fetal-Ultrasounds

# Abstract
This research project aims to investigate how to employ Convolutional Neural Networks (CNN) in image classification. The project entails reading a related research paper, analyzing a supplied dataset, and then shortlisting architectures for feature extraction and classification using a fully connected neural network. Basic preprocessing techniques link images to their respective labels in the dataset and perform transformations. Transfer learning is then utilized to improve the performance of the CNN by leveraging the pre-trained Iights of a network trained on the ImageNet dataset. The network's convolutional layers are fixed, while only the weights of the fully connected layers are changed. The model's classification accuracy is validated by examining the outcomes of the training and validation datasets. Lastly, the model's performance is evaluated by sending test images across the network, and the results are assessed to determine the efficacy of the suggested methodology. Overall, this effort advances the science of computer vision and deep learning by investigating the use of CNNs in picture classification.

# Introduction
I decided to process the input photos using a modified VGG16 network. To ensure that the ML algorithms do not learn improperly, the target labels were one hot encoded beforehand. Several variables influenced the final model's selection: Several models, such as DenseNet-168, ResNeXt-101, and VGG16, shoId appreciable results in the research paper, but due to previous experience in constructing VGG16 and several tasks at hand in a short period of time - I found the implementation of VGG16 to be the appropriate choice due to prior working experience and because the other architectures did not provide a significant boost in accuracy. One disadvantage of using the VGG16 architecture was that the model built was the heaviest feasible, with around 134M trainable parameters.

# Data Pre-Processing/Analysis
The training dataset in Task1 consisted of 1646 images out of which the training-validation split was of 90:10. The basic transformations that were utilised included resizing the image to a size of 224x224. This size was chosen since I implemented transfer learning by using pretrained VGG16 model trained on ImageNet dataset takes 224x224 images as input - hence to prevent any errors from occuring this size was chosen. I normalized the value of the image with mean=0.5 and std deviation=0.5; upon paying further attention to normalization - the function call is different from usual calls for RGB image- this happened since our image is grayscale and has only one image channel. Lastly, I added Random Horizontal with probability of 80% to add some simple but helpful augmentations. Model was run with and without Random Horizontal Flip, and it was observed that validation accuracy was increased when augmentations were used. Lastly, for the functioning of any ML model the input image must be in the form of a tensor and hence the torch.ToTensor() function was utilised to make this change.
Labels were present in a csv file. Since the labels were categorical, they were first factorised and hence post factorisation of labels, a dictionary was created for quick and easy access of the original labels. Before feeing the labels as ground truth values into the network, the labels were one hot encoded.
**Note: The dataset is available to whoever wants upon request (send mail: [full_name_in_all_small_leters_followed_by_'16'] AT gmail DOT com**
# Model Architecture

The network architecture above depicts a typical VGG16 architecture with 13 convolution layers and 3 fully connected layers. The model used for completion of Task1 was pretrained on ImageNet dataset, but since our training images were slightly different, the architecture was
modified to have only 1 input channels.
Network layers were frozen for the convolution portion, and no learning took place in the convolution layers; the only learning which took place during training was done on the fully connected layers.

The fully connected layer of original pretrained model had 1000 output channels since it was trained on ImageNet. In the modified architecture, only 4 output channels are present. 1 additional Linear layer and 1 additional ReLu were also added due to observed marginal increase in training and validation accuracy. This model was ideal to extract necessary features and then pass these features through fully connected network to perform high accuracy classification.

# Experimental Setting

● All computations have been performed using the help of PyTorch
● The batch size given the dataset had to be large enough but not too large since that would
result in the RAM crashing on google colab. Hence a batch size of 32 was chosen
appropriate for the Data Loader.
● Adam optimizer has been ecployed with a learning rate of 1e-3(as discussed in the
referred research paper)
● A learning rate scheduler of the type:ReduceLROnPlateau has also been implemented to
aid the training process(as discussed in the referred research paper)
● Cross Entropy Loss has been chosen as the appropriate loss function



