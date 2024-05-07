# Code Challenge for Research Engineer @ Telefónica Research - Flower + Pytorch with CelebA dataset

- **Summary**:
Build a Federated Learning setting using Flower (https://flower.ai) and PyTorch that uses Large-scale CelebFaces Attributes (CelebA) Dataset.

- **Method**:
  - Design a data splitter to federate the data creating 2 data distributions:
    - IID distribution
    - non-IID distribution
- Clearly define training and testing datasets, classes, etc.
- Use a pre-trained version of MobileNetV2; Freeze the feature extractor; Train the classifier head from scratch
- Execute the federated learning training with 50 clients for at least 10 FL rounds
- Report the training and testing performance with appropriately selected metrics
- Compare performance across demographic groups present in the CelebFaces dataset


- **Prerequisites**:  
      - Install libraries/dependencies (PyTorch, Flower FL framework, etc.)  
      - Download the CelebA dataset if required.  


## Status

For this *unfinished* code challenge:  
    - I have developped a data splitter and dataloaders using 2 data distributions,  
    - I have used a pre-trained version of MobileNetV2, frozen its feature extractor, and trained the classifier head from scratch,   
    - I executed the federated learning training with 50 clients for at least 10 FL rounds.  
    
  Missing areas:  - Report training and testing performance.  
                    - Compare performance across demographic groups.  
                    - Select appropiate metrics.  
                    - Deal with overfitting, specially with non-iid distributions.  


## Implementation details

I have chosen as task to train the models to classify the attribute "Smiling", an attribute part of the CelebA dataset.  
The code is divided in three Python files:  

- In *datasets.py* the data splitter is implemented. The methods "create_iid_splits()" and "create_non_iid_splits" creates 2 different data distributions.  
    - While "create_iid_splits()" divides the dataset evenly among clients, sampling each datapoint uniformely, "create_non_iid_splits()" splits the dataset by user, sending to each client just the data from a single *Celeb*.  
    - Both methods opperates only on the data from a constant NUM_CLIENTS, defined in this code challenge as 50.  
    - For analysis comparison and debug purposes, I have also implemented a Pytorch DataLoader that samples data from the whole dataset.  

- In *flwr_mobilenet_celeba.py* the rest of the code for this challenge is wrriten.  
    - The class *MobileNet* loads a pre-trained MobileNetV2 and freezes its feature extractor.  
    - It defines a new classifier head for binary classification ("Smiling" or not)  
    - Implementation of Flower-based federated learning simulation.  
- *mobilenet_celeba.py* has the code for tackling the task in a non-federated strategy is shown for debugging purposes and results comparisons.  

