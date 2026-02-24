# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Design Steps

### Step 1:
Write your own steps

### Step 2:

### Step 3:



## Program
#### Name:V KAMALESH VIJAYAKUMAR
#### Register Number:212224110028
Include your code here
```Python 
## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), 
                         self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Take last time step output
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out



model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the Model
## Step 3: Train the Model

epochs = 20
train_losses = []   

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        outputs = model(xb)
        loss = criterion(outputs, yb)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)   

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")


```

## Output

### True Stock Price, Predicted Stock Price vs time
<img width="737" height="506" alt="image" src="https://github.com/user-attachments/assets/cc678d7e-4c57-41aa-a71b-0e104aaccc2d" />



### Predictions 
<img width="317" height="49" alt="image" src="https://github.com/user-attachments/assets/a2297af8-b892-4721-b20e-669cd704124b" />


## Result
The Recurrent Neural Network model for stock price prediction has been implemented and executed successfully.

