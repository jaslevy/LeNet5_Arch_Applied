# Fashion_MNIST practical 

Applying the architecture from LeNet-5 CNN (LeCun et .al., 1998) for Classification on the Fashion-MNIST dataset by Zalando (2014)

See http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf & https://github.com/zalandoresearch/fashion-mnist 

## Content Overview

This project includes a Jupyter Notebook file which creates a training and tuning regime for finding a champion model. 
The functions  'tune_hyperparameters', 'train_epoch', 'evaluate', 'plot_accuracies' are generalized to be applied to models 
with different regularization techniques.

## Dependencies
All dependencies are imported within the markdown file. Simply run the file to import correct packages.
Please use **Python 3.10+**

## Data Upload
It is difficult to persist data files in Google Colab. As such, we decided to use wget so that we can retrieve the dataset easily whenever we run the notebook. After importing the dataset, we unzipped and read the idx files to store into our train and test sets and to convert these into tensors. We then split our train set into train and validation by setting the validation size to 5000 and using a random split.

## Key Functions

### 'train_epoch'
The 'train_epoch' function trains and updates the parameters of the model for one epoch.
It performs one forward pass and backpropogation to update the weights using the specified optimizer
(SGD or Adam is what we use in this project). The function returns the loss and accuracy of the training epoch

### 'evaluate'
The 'evaluate' function evaluates the performance of a model. it used torch.no_grad() to disable gradient calculation after training. This function then gets predictions from the images put into the model calculates loss (usingthe loss function used for the 'criterion' argument) between predictions and true labels. Next, it calculates the epoch loss as an average of loss across all batches, and returns this average in addition to the epoch accuracy.

### 'plot_accuracies'
The 'plot_accuracies' function is used to plot the train and test accuracies as a function of training epoch. This function is called once within 'tune_hyperparameters' in order to plot these values for the champion model determined by hyperparameter tuning 

### 'tune_hyperparameters'
The 'tune_hyperparameters' function performs a brute-force grid-search through all possible hyperparameter combinations in order to find a champion model with the optimal hyperparameters. The function trains a model for each combination for a default of 'num_epochs' = 10. The following are default arguments for hyperparameters:  

* batch_sizes=[32, 64, 128]
* learning_rates=[0.1, 0.01, 0.001]
* optimizers=['SGD', 'Adam']

These default values leave us with 18 tuning iterations by default. It is recommended to connect to a GPU for this computationally difficult task. This function also has arguments 'dropout_rates' and 'weight_decay_weights' which defualt to 'None' but can take a list of floats to add to the hyperparameter grid-search. Importantly this function handles the deactivation of dropout for model evaluation. 
&nbsp;
This function uses the above three functions defined in order to track a champion (optimal) model based on validation accuracy. The function creates an accuracy plot for the best model and returns a champion model along with champion parameters and the highest training accuracy.
&nbsp;
To call the function on a model without dropout (i.e. base model, batch normalization), you should create a 'partial' of the model so that 'tune_hyperparameters' can accept the model input

```python
from functools import partial

LeNet5_no_dropout = partial(LeNet5)

best_model, best_params, lenet_train_acc = tune_hyperparameters(LeNet5_no_dropout, train_loader, val_loader, test_loader, device, title='Accuracy vs. Epochs for non-regularized model')
```

Calling a model with dropout can be done the following way: 

```python
def create_le_net5_dropout(dropout_rate):
    return LeNet5_Dropout(dropout_rate)

dropout_rates = [0.2, 0.3, 0.4, 0.5]

best_model_dropout, best_params_dropout, dropout_train_acc = tune_hyperparameters(lambda dropout_rate: create_le_net5_dropout(dropout_rate), train_loader, val_loader, test_loader, device, dropout_rates=dropout_rates, title='Accuracy vs. Epochs for model with dropout')
```

## Architecture Adjustment

The FashionMNIST dataset consists of images that are of dimensions 28x28. Seeing that the LeNet5 architecture expects images of size 32x32, we simply padded the images from the FashionMNIST dataset to work with the LeNet5 model. We also normalized the images using torch vision transforms.
&nbsp;
Upon preparing the dataset to fit our models, we then set up data loaders to enable mini-batch sampling.
&nbsp;
For deciding dropout percentages, we initially chose p=0.5 and then decided to tune p as a hyperparameter.
We started with p=0.5 becaause research on preventing overfitting with dropout suggested, "p can be chosen using a validation
set or can simply be set at 0.5, which seems to be close to optimal for a wide range of networks and tasks" (Shrivastava et. al., 2014) - https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf


## Results

### Base Model (No Regularization)
Final Test Accuracy: **89.05**

![base_epoch_acc](https://github.com/user-attachments/assets/ec9b8aa5-6439-4516-96d4-682f1167c220)

### Dropout Regularization
Final Test Accuracy: **89.92**

![dropout_eval_acc](https://github.com/user-attachments/assets/a0fc64b0-23cb-4755-a452-040f9aa2b69b)

### Batch Normalization 
Final Test Accuracy: **89.72**

![batchnorm_eval_acc](https://github.com/user-attachments/assets/3d53aa23-07c5-4899-9481-b6213cee4348)


### Weight Decay (l2 regularization)
Final Test Accuracy: **89.40**

![weightdecay_eval_acc](https://github.com/user-attachments/assets/d0d48c22-55f9-47f2-b6ab-e900a9d1e953)

### Final Results
<img width="630" alt="Screenshot 2024-09-11 at 9 02 28 PM" src="https://github.com/user-attachments/assets/9b507e6b-df7d-431e-8de8-75c08fd435c0">

## Conclusion & Discussion

Information about the project's license.
