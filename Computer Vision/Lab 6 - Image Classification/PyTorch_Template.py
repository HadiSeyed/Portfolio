import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self,filename):
        super(MyDataset, self).__init__()
#         TODO: implement what happens when someone calls: dataset = MyDataset()
#         Pull in relevant file information, images lists, etc.
    
    def __getitem__(self, idx):
#         TODO: implement what happens when someone calls dataset[idx]
#         Return an image and it's associated label at location idx 
    
    def __len__(self):
#         TODO: implement what happens when someone calls len(dataset)
#         Determine the number of images in the dataset

train_dataset = MyDataset("training.txt")
test_dataset = MyDataset("test.txt")

train_loader = DataLoader(train_dataset,batch_size=1)
test_loader = DataLoader(test_dataset,batch_size=1)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
#         TODO: setup network here

    def forward(self, x):
#         TODO: perform the forward pass, which happens when someone calls network(x)


model = MyNetwork()


##### Training Loop #####

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

num_epochs = 5

for epoch in range(num_epochs):
    for im, label, in train_loader:

        optimizer.zero_grad()

        # Your code here

        loss.backward()
        optimizer.step()


##### Testing Loop #####

model.eval()
with torch.no_grad():
    for im,label in test_loader:
        
        # Your code here


# Additional code to plot results placed here