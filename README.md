# CNN_MNIST
A basic CNN model for MNIST dataset with cross validation approach is there.

# Data Preparation 
I have used Python libraries like tensorflow and scikit-learn, for data manipulation and preprocessing./
The original images are resized to 28x28 pixels. This resizing ensures compatibility with standard CNN architecture. Following resizing, the pixel values of the images are normalized to a range of 0 to 1. Normalization helps in stabilizing the training process by scaling down pixel values to a uniform range.
The labels associated with each image are converted from integer formats to one-hot encoded vectors. This conversion is crucial for multi-class classification tasks, as it allows the neural network to output a probability distribution across various classes.
# Convolutional Neural Network Architecture

Model: "sequential_7"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_20 (Conv2D)          (None, 28, 28, 32)        320       
                                                                 
 max_pooling2d_12 (MaxPooli  (None, 14, 14, 32)        0         
 ng2D)                                                           
                                                                 
 conv2d_21 (Conv2D)          (None, 14, 14, 64)        18496     
                                                                 
 max_pooling2d_13 (MaxPooli  (None, 7, 7, 64)          0         
 ng2D)                                                           
                                                                 
 conv2d_22 (Conv2D)          (None, 7, 7, 128)         73856     
                                                                 
 flatten_7 (Flatten)         (None, 6272)              0         
                                                                 
 dense_14 (Dense)            (None, 128)               802944    
                                                                 
 dropout_7 (Dropout)         (None, 128)               0         
                                                                 
 dense_15 (Dense)            (None, 10)                1290      
                                                                 
=================================================================
Total params: 896906 (3.42 MB)
Trainable params: 896906 (3.42 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________


First Convolutional Layer: The model starts with a Conv2D layer with 32 filters of size 3x3, using the ReLU activation function. The padding='same' parameter ensures that the output of this convolution has the same spatial dimensions as the input, preventing reduction in size due to the convolution. The image filters increases from 28 x 28 x 1 to 28 x 28 x 32
Max Pooling Layer makes the image dimension becomes half from 28 x 28 x 32 to 14 x 14 x 32
Second Convolutional Layer: Another Conv2D layer follows, this time with 64 filters, also 3x3. This layer increases the depth of the feature maps while still maintaining the same size due to the padding='same'.The image size goes from 14 x 14 x 32 to 14 x 14 x 64.
Max Pooling Layer positioned after this layer, make the image dimension half from 14 x 14 x 64 to 7 x 7 x 64
Third Convolutional Layer: The final convolutional layer has 128 filters of size 3x3, which further deepens the feature maps, extracting more complex features from the input. Again, padding='same' is used to maintain spatial dimensions. The image dimension goes from 7, 7, 64 to 7, 7, 128
Architecture Diagram : 





3. Max Pool Layers: 
Positioned after the first and second convolutional layers, these MaxPooling2D layers with a pool size of 2x2 serve to reduce the spatial dimensions of the feature maps by half. This layer has 0 parameters.
 This downsampling reduces the computational complexity for the network and helps in achieving translational invariance. Any overlapping information is removed and only the important information remains keeping the number of filters the same. 
4. Fully Connected Layer and Softmax : 
First Dense Layer: This fully connected layer has 128 nodes and uses the ReLU activation function. It serves as a decision layer that uses the features extracted by the convolutions to determine the image's class. It has 802944 parameters in our model above.
Output Layer: The final layer is a Dense layer with 10 nodes, corresponding to the 10 classes. It has 1290 parameters in our model. It uses the softmax activation function to output probabilities for each class, ensuring that the output values sum up to 1 and each value lies between 0 and 1.
5. Training and Evaluation
K-Fold Cross-Validation is utilized to validate the model's effectiveness more reliably by dividing the entire dataset into 'K' equal subsets. Here, StratifiedKFold is used with n_splits=5, meaning the dataset is split into 5 distinct folds, maintaining the percentage of samples for each class.
shuffle=True ensures the data is shuffled before being split, and random_state=42 provides a seed for reproducibility of shuffle randomness.
Model Training within Each Fold:
Model Creation and Compilation: For each fold, a fresh instance of the model is created by calling create_model(). The model is compiled with the Adam optimizer and categorical_crossentropy loss function, which is good for multi-class classification tasks.
Training : The model is trained using the training portion of the fold (train_images[train], train_labels[train]) for 10 epochs. During this phase, model weights are adjusted to minimize the loss function, using not only the training data but also validating the model on a separate validation set (train_images[val], train_labels[val]), which provides a measure of model performance on unseen data during the training process.
Model Evaluation:
Validation Performance: After training, each model instance is evaluated on the validation subset of the current fold using model.evaluate(), which returns the loss value and metrics (accuracy in this case) computed on the validation set.
The accuracy for each fold is printed and stored in cvscores, a list that captures model performance across all folds. This metric gives insight into how well the model is expected to perform on unseen data.
Overall Performance Assessment:
After all folds have been processed, the average accuracy across all folds is calculated and typically reported as the model's performance metric. 
Confusion Matrix : 
[[ 976    1    0    0    0    0    1    1    1    0]
 [   0 1133    0    0    0    0    1    1    0    0]
 [   2    0 1028    0    0    0    0    1    1    0]
 [   0    0    0 1002    0    6    0    0    2    0]
 [   0    0    1    0  973    0    3    0    0    5]
 [   1    0    0    2    0  888    1    0    0    0]
 [   4    2    0    0    0    1  950    0    1    0]
 [   0    4    4    0    0    0    0 1015    1    4]
 [   1    0    1    0    1    0    0    0  970    1]
 [   0    1    1    0    5    3    1    0    3  995]]

