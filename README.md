# GameVision-Sports-AI-Classifier

GameVision is a deep learning-powered image classification model designed to identify sports video game images across five categories: Basketball, Volleyball, Tennis, Soccer, and American Football. Leveraging the ResNet101V2 model, the system fine-tunes its capabilities using the Sports10 dataset to achieve high accuracy. The model applies data augmentation techniques for improved generalization and utilizes a transfer learning approach to classify sports scenes efficiently. GameVision is developed as part of the Datathon AI X DATA: Vision Pillar (Spring 2025), pushing the boundaries of AI-driven sports analytics.

## Objective
The goal is to create or fine-tune a computer vision model that can accurately predict the sport category of each image in the test set. The final model will be evaluated based on its classification accuracy.

## Features

- Deep Learning with ResNet101V2: Uses a pre-trained ResNet model with transfer learning.

- Multi-Class Classification: Classifies images into five distinct sports categories.

- Data Augmentation: Improves generalization through transformations like rotation, flipping, and zooming.

- Efficient Training: Optimized training process for accurate predictions.

## Dataset

The model is trained on the Sports10 dataset, which contains labeled images of different sports. If you need to prepare your dataset, ensure images are organized in subdirectories by category.

## Models Used
We experimented with three different models during the project:

- ResNet101V2: A deep residual network that initially did not perform well, but after tuning, its performance improved.
  
- EfficientNet: A lightweight yet powerful model that provided a solid baseline for image classification.

- MobileNetV2: Another efficient model suitable for mobile and edge devices, which also performed well in our tests.

## Model Performance

- Achieves high accuracy through transfer learning.

- Regularization techniques used to prevent overfitting.

## Technologies Used

- TensorFlow and Keras for model building and training
  
- ResNet101V2 pre-trained model for transfer learning
  
- Pandas for data manipulation
  
- Matplotlib and Seaborn for data visualization

- Scikit-learn for splitting the dataset

## Steps

1. Import Libraries

- Necessary libraries for data manipulation, model creation, and visualization are imported.

2. Configuration and Hyperparameters

- Here we define key parameters and settings like image dimensions, batch size, and learning rate.

3. Load and Explore the Dataset

- The dataset is loaded into Pandas DataFrames for training and testing. We also create a validation set by splitting the training data.

4. Data Augmentation

- We define the data augmentation process to help prevent overfitting and enhance model generalization.

5. Create Data Generators

- Data generators are created for the training, validation, and test sets to feed the model with batches of images.

6. Model Definition - ResNet101V2

- We load the pre-trained ResNet101V2 model and add custom layers for classification.

7. Compile the Model

- The model is compiled with the Adam optimizer and categorical crossentropy loss function.

8. Model Training

- The model is trained using the training and validation data generators.

9. Evaluation

- The model's accuracy is evaluated using the test dataset.

## Submission

The final model will generate predictions for the test set. The results will be saved in a CSV file, with each row containing the image ID and the predicted class label.

## Results
The model achieved an accuracy of 46% on the test set, demonstrating the potential for further improvements. Currently, we only used ResNet101V2, but we believe combining the predictions from all three models could lead to improved accuracy. By leveraging the strengths of each model, we can enhance the overall performance of the image classification system.

## Future Improvements
Model Combination: One area for improvement could be combining the models to take advantage of their individual strengths and improve overall accuracy.

Hyperparameter Tuning: Further tuning of hyperparameters could help improve the performance of the models.

Data Augmentation: Implementing data augmentation techniques could also increase the dataset size and enhance the model’s generalization.

## Conclusion
This project demonstrates the process of building a deep learning model to classify sports video game images into multiple categories using a pre-trained model. The model's performance can be enhanced further by adjusting hyperparameters and using advanced techniques such as fine-tuning or other architectures.

## License

This project is licensed under the MIT License.


