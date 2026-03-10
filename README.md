# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.


## Problem Statement and Dataset

The objective of this work is to design and implement a deep learning model for predicting stock closing prices using historical time-series data. The dataset consists of training and testing stock price records, where only the “Close” price is considered for analysis. The data is normalized and converted into sequential input samples to capture temporal dependencies over a fixed time window. A Recurrent Neural Network (RNN) model is developed using PyTorch to learn patterns from past price movements. The model is trained using Mean Squared Error loss and optimized with the Adam optimizer. Finally, the trained model is evaluated on unseen test data, and the predicted prices are compared with actual prices to analyze forecasting performance.


## Design Steps

### Step 1:

Load the training and testing datasets and extract the closing price column for analysis. Normalize the data using MinMax scaling to ensure stable and faster convergence during training.

### Step 2:

Convert the normalized time-series data into sequential input-output pairs using a fixed window size. This helps the model learn temporal dependencies from past stock prices.

### Step 3:

Transform the generated sequences into PyTorch tensors and create a DataLoader for batch-wise training. This enables efficient data handling and model optimization.

### Step 4:

Define a Recurrent Neural Network model with multiple hidden layers and a fully connected output layer. Configure the loss function as Mean Squared Error and use the Adam optimizer for training.

### Step 5:

Train the model over multiple epochs and monitor the training loss to evaluate learning progress. Plot the loss curve to analyze convergence behavior.

### Step 6:

Use the trained model to predict stock prices on the test dataset. Perform inverse scaling and compare predicted values with actual prices to evaluate forecasting performance.





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
<img width="1579" height="238" alt="image" src="https://github.com/user-attachments/assets/0ede8c38-169a-4dbf-9f54-bbe72ad6936a" />


## Result
The Recurrent Neural Network model for stock price prediction has been implemented and executed successfully.

