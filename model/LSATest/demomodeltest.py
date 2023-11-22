from json import decoder
import torch 
from tqdm import tqdm
import time,os
import matplotlib.pyplot as plt


#for our test we need a custom layer that transposes the input
class MyLinear(torch.nn.Linear):
    def forward(self,x):
        return super().forward(x.T).T


os.makedirs("results",exist_ok=True)




linear_layer= torch.nn.Linear(100,100)
linear_layer2= torch.nn.Linear(100,100)#MyLinear(100,100)
linear_layer3= torch.nn.Linear(100,100)
linear_layer4= torch.nn.Linear(100,100)#MyLinear(100,100)
linear_layer5= torch.nn.Linear(100,100)
linear_layer6= torch.nn.Linear(100,100)#MyLinear(100,100
#noise_layer.weight.requires_grad=False

#list of activation functions
activations=[
    torch.nn.ReLU(),
    torch.nn.Tanh(),
    torch.nn.Sigmoid(),
    torch.nn.Softmax(dim=1),
    torch.nn.Softmax(dim=0),
    torch.nn.Softmax(dim=-1),
    torch.nn.Softmax(dim=-2),
    #activation to make all positive
]

model2=torch.nn.Sequential(linear_layer,
                            torch.nn.ReLU(),
                            linear_layer2,
                            torch.nn.ReLU(),
                            linear_layer3,
                            torch.nn.Sigmoid(),
                            linear_layer4,
                            torch.nn.ReLU(),
                            linear_layer5,
                            torch.nn.ReLU(),
                            linear_layer6,
                            torch.nn.ReLU())
if torch.cuda.is_available():
    model2=model2.cuda()
    model2.to(device="cuda")
optimizer=torch.optim.Adam(model2.parameters(),lr=0.000005)

Loss=torch.nn.CrossEntropyLoss()
prog_bar=tqdm(range(100000))
training_Losses=[]
#begin timer
for i in prog_bar:# do 1000 iterations
    optimizer.zero_grad()

    #memory=torch.randn(10,10,device="cuda" if torch.cuda.is_available() else "cpu")
    input_tensor=torch.randn(10,10,requires_grad=True,device="cuda" if torch.cuda.is_available() else "cpu")
    labels=torch.arange(10,device="cuda" if torch.cuda.is_available() else "cpu")
    #in theory.... this could be batched to run faster. However, in this case, it's not really necessary and this is meant to simulate a single batch of data anyway. 
    #output=model(input_tensor.flatten()).reshape(10,10)
    #output=model2(input_tensor,tgt=input_tensor)
    output=model2(input_tensor.flatten()).reshape(10,10)

    # output=output+torch.randn(10,10,device="cuda" if torch.cuda.is_available() else "cpu")*0.1
    # output=model2(output.flatten()).reshape(10,10)

    loss1=Loss(output,labels)
    loss2=Loss(output.T,labels)
    loss=loss1+loss2
    loss=loss.mean()/2
    loss.backward()
    #print(loss.item())

    optimizer.step()
    training_Losses.append(loss.item())
    #update tqdm bar
    log={"Loss":loss.item()}
    prog_bar.set_postfix(log)

input_tensor=torch.randn(10,10,requires_grad=True,device="cuda" if torch.cuda.is_available() else "cpu")
plt.subplot(1,2,1)
plt.title("Input")
plt.imshow(input_tensor.detach().cpu().numpy())
plt.subplot(1,2,2)
plt.title("Output")
plt.imshow(model2(input_tensor.flatten()).reshape(10,10).detach().cpu().numpy())
plt.savefig("results/LSATest.png")