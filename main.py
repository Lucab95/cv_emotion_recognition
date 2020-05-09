import torch
import pandas as pd

# x= torch.rand(4,2)
# x= torch.zeros(4,2, dtype=torch.long)
# x = torch.tensor([2.3,1.2,1])
x=torch.randn(2,3,requires_grad=False)

df = pd.read_csv('fer2013/fer2013/fer2013.csv')
x = df.emotion
training_set = df.Usage
for img in training_set:
   if img != "Training" :
      print(img)
print(x)

# trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
#    download = True, transform = transform)
#
# print(x)
