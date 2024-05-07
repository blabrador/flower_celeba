import pandas as pd
import random
from PIL import Image 
import torch 
import torchvision.transforms as transforms

# Dataset class for PyTorch data loaders
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, split, data_dict, attribute="Smiling", demographics=["Male", "Young"], transform=None):
        self.image_names = split
        self.data_dict = data_dict
        self.attribute = attribute
        self.demographics = demographics
        self.transform = transform
        self.path = '/media/data2/flowers/leaf/data/celeba/data/raw/img_align_celeba/'

    def __getitem__(self, index):
        image_name = self.image_names[index]
        attributes = self.data_dict[image_name]
        label = attributes[self.attribute]  # 1 for smiling, 0 otherwise
        demographics = [attributes[attr] for attr in self.demographics]  
        image_path = self.path + image_name 
        image = load_and_preprocess(image_path, self.transform)
        return image, label, demographics

    def __len__(self):
        return len(self.image_names)

def load_and_preprocess(image_path, transform):
    image = Image.open(image_path) #.convert('RGB')  # Ensure RGB format
    image = transform(image)
    return image

def create_iid_splits(data_dict, train_image_names, num_clients):
    """Splits training data into IID splits for clients"""
    random.seed(42) # Seed for consisting eval
    random.shuffle(train_image_names)  # Shuffle image names for IID distribution

    # Divide images approximately evenly among clients 
    splits = [[] for _ in range(num_clients)]  # Create empty lists for client data
    for idx, name in enumerate(train_image_names): 
        client_idx = idx % num_clients
        splits[client_idx].append(name)

    return splits

def create_non_iid_splits(user_dict_rev, train_selected_users, num_clients):
    """Splits training data into non-IID splits, one for each client (user-based)"""
    
    splits = [[] for _ in range(num_clients)]  # Create empty lists for client data

    for idx, name in enumerate(train_selected_users):
        splits[idx] = user_dict_rev.get(name)

    return splits

def load_datasets(num_clients, iid=True, train_test_ratio=0.999, attribute="Smiling", full_dataset_debug=False):
    """ Loads CelebA, creates splits, and returns DataLoaders """

    # Load attribute and user data with Pandas
    user_df = pd.read_csv("/media/data2/flowers/leaf/data/celeba/data/raw/identity_CelebA.txt", 
                          delim_whitespace=True, header=None)
    user_dict = dict(zip(user_df[0], user_df[1])) 
    user_dict_rev = {} # Reverse user dict
    for k, v in user_dict.items():
        user_dict_rev.setdefault(v, []).append(k)
    

    attr_df = pd.read_csv("/media/data2/flowers/leaf/data/celeba/data/raw/list_attr_celeba.txt",
                          delim_whitespace=True, header=0, skiprows=1) 

    data_dict = {}
    for index, row in attr_df.iterrows():
        attributes = row.to_dict()  # Convert row to dictionary for direct attributes
        attributes = {k: 0 if v == -1 else v for k, v in attributes.items()}  # Change -1 to 0
        data_dict[index] = attributes

    # Split data into train and test sets
    all_image_names = list(data_dict.keys())
    random.seed(42) # Seed for consisting eval
    random.shuffle(all_image_names) 
    split_idx = int(len(all_image_names) * train_test_ratio) 
    train_image_names = all_image_names[:split_idx]
    test_image_names = all_image_names[split_idx:]
    print(len(test_image_names))

    # Filter training data with minimum image count and get data from NUM_USERS
    # making sure that users in train are not in the test set
    min_user_images = 5 
    train_selected_users = []
    train_selected_images = []
    for user, image_list in user_dict_rev.items():
        train_image_list = []
        skip_user = False
        for image in image_list:
            if image not in test_image_names:
                train_image_list.append(image)
            else:
                skip_user = True
        if not skip_user:
            if len(train_image_list) >= min_user_images:
                train_selected_images.extend(train_image_list)
                train_selected_users.append(user)
            if len(train_selected_users) == num_clients:
                break

    # Create IID vs. non-IID splits
    if iid:
        train_splits = create_iid_splits(data_dict, train_selected_images, num_clients) 
    else:
        train_splits = create_non_iid_splits(user_dict_rev, train_selected_users) 

    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),  # Deterministic crop
                                     transforms.ToTensor(),
                                     normalize])
    # Datasets
    train_datasets = []
    for split in train_splits:
        dataset = CelebADataset(split, data_dict,attribute=attribute, transform=transform) 
        # TODO: data augmentation
        train_datasets.append(dataset)

    test_dataset = CelebADataset(test_image_names, data_dict,attribute=attribute, transform=transform)

    # DataLoaders
    trainloaders = [torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True) for ds in train_datasets]
    if full_dataset_debug:
        dataset = CelebADataset(train_image_names, data_dict,attribute=attribute, transform=transform) # Full dataset 
        trainloaders = [torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)]
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)  

    return trainloaders, testloader