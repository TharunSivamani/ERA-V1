# ERA - V1
## Session 5 - MNIST Data Classification using Deep NN

<br>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Build Status](https://github.com/TylerYep/torchinfo/actions/workflows/test.yml/badge.svg)](https://github.com/TylerYep/torchinfo/actions/workflows/test.yml)
[![pytorch](https://img.shields.io/badge/pytorch-2.0-orange)](https://pytorch.org/)

<br>

## About the Data 

- The MNIST dataset consists of a large collection of handwritten digits from ```0 - 9```. 
- It contains *60,000* training examples and *10,000* test examples. 
- Each example is a grayscale image of size ```28x28``` pixels, representing a single handwritten digit. 
- The images are accompanied by corresponding labels indicating the true digit value.



# Usage

- `git clone https://github.com/`
<br>
- `pip install torch torchvision torchsummary`
<br>
<pre>
Session 5  
    ├─ model.py  
    ├─ utils.py  
    ├─ S5.ipynb
</pre>
<br>



**Modular** structure followed for writing the code.

* **utils.py** is the Python Script having functions right from datasets import , data-transformation , train-test codes , plotting and optimizing functions.

* **model.py** is the Python Script having the network architecture.

* **S5.ipynb** is the main code file where functions from the utils.py and model.py are called to meet the objective.

<br>

# How to Use

```python
from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

- Note that the `input_size` is required to make a forward pass through the network.

<br>

# NN Architecture for MNIST Data
<br>

```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
```

<br>

# Model Summary
<br>

```python
================================================================
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
================================================================
```
<br>

# Results (Accuracy and Loss)
<br>

![Results](Results\Session 5\result.png)
<br>
# How to Run
1. Install all the prerequisites.
2. Clone this repository.
3. Place all the python scripts in the same folder.
4. Run the ```S5.ipynb``` notebook in Jupyter.

