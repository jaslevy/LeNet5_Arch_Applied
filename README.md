# Fashion_MNIST practical 

Applying the architecture from LeNet-5 CNN (LeCun et .al., 1998) for Classification on the Fashion-MNIST dataset by Zalando (2014)

See http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf & https://github.com/zalandoresearch/fashion-mnist 

## Content Overview

This project includes a Jupyter Notebook file which creates a training and tuning regime for finding a champion model. 
The functions  'tune_hyperparameters', 'train_epoch', 'evaluate', 'plot_accuracies' are generalized to be applied to models 
with different regularization techniques.

## Dependencies

## Data Upload

## Key Functions

### 'evaluate'

### 'train_epoch'

### 'plot_accuracies'

### 'tune_hyperparameters'

## Architecture Adjustmnet


## Results

### Base Model (No Regularization)
Final Test Accuracy: 89.05
![base_epoch_acc](https://github.com/user-attachments/assets/ec9b8aa5-6439-4516-96d4-682f1167c220)

### Dropout Regularization
Final Test Accuracy: 89.92
![dropout_eval_acc](https://github.com/user-attachments/assets/a0fc64b0-23cb-4755-a452-040f9aa2b69b)

### Batch Normalization 
Final Test Accuracy: 89.72
![batchnorm_eval_acc](https://github.com/user-attachments/assets/3d53aa23-07c5-4899-9481-b6213cee4348)


### Weight Decay (l2 regularization)
Final Test Accuracy: 89.40
![weightdecay_eval_acc](https://github.com/user-attachments/assets/d0d48c22-55f9-47f2-b6ab-e900a9d1e953)
![weightdecay_eval_acc](https://github.com/user-attachments/assets/d0d48c22-55f9-47f2-b6ab-e900a9d1e953)

### Final Results

Training and Testing Accuracies:
                      Model  Training Accuracy (%)  Testing Accuracy (%)
0           Without Dropout              91.480000                 89.05
1              With Dropout              91.014545                 89.92
2  With Batch Normalization              92.432727                 89.72
3         With Weight Decay              91.892727                 89.40




## Conclusion & Discussion

Information about the project's license.
