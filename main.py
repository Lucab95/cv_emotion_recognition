import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# x= torch.rand(4,2)
# x= torch.zeros(4,2, dtype=torch.long)
# x = torch.tensor([2.3,1.2,1])
x=torch.randn(2,3,requires_grad=False)

df = pd.read_csv('../fer2013/fer2013/fer2013.csv')
# df.info()
print(df.head())

img = df["pixels"][5]
val = img.split(" ")
x_pixels = np.array(val, 'float32')
x_pixels /= 255
x_reshaped = x_pixels.reshape(48,48)


plt.imshow(x_reshaped, cmap= "gray", interpolation="nearest")
plt.axis("off")
plt.show()
# x = df.emotion
# training_set = df.Usage
# for img in training_set:
#    if img != "Training" :
#       print(img)
# print(x)

# trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
#    download = True, transform = transform)
#
# print(x)
