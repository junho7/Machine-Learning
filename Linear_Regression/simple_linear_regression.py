import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Prepare Data ---
# (Assuming you have your features in X and targets in y as PyTorch tensors)
# For demonstration:
X = torch.randn(100, 5)  # 100 samples, 5 features per sample
y = 3 * X[:, 0] + 2 * X[:, 1] - 5  # True relationship for generating the target

# Important: Ensure y has the correct shape (100 samples, 1 output per sample)
y = y.unsqueeze(1)  # Reshape y to (100, 1)

# --- 2. Define the Model ---
class LinearRegressionModel(nn.Module):  # Subclassing nn.Module
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()  # Calling nn.Module's constructor
        self.linear = nn.Linear(input_dim, output_dim)  # The linear layer (y = Wx + b)

    def forward(self, x):  # Defines the forward pass
        out = self.linear(x)
        return out

input_dim = X.shape[1]  # Number of features
output_dim = 1         # We're predicting a single value

model = LinearRegressionModel(input_dim, output_dim)  # Create the model instance

# --- 3. Define Loss and Optimizer ---
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# --- 4. Training Loop ---
num_epochs = 500
losses = []  # Store losses for plotting

for epoch in range(num_epochs):
    # Forward Pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward Pass and Optimization
    optimizer.zero_grad()  # Reset gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    # Save Loss
    losses.append(loss.item())

    # Visualization (every 10 epochs)
    if (epoch + 1) % 10 == 0:
        plt.cla()  # Clear previous plot
        
        # Get Predictions and True Values
        predicted = model(X).detach().numpy()  
        x_values = X[:, 0].numpy()
        y_values = y.numpy()
        
        # Calculate and Plot Regression Line
        z = np.polyfit(x_values, predicted.flatten(), 1)  
        p = np.poly1d(z)
        # Convert tensors to numpy arrays for plotting
        predicted = model(X).detach().numpy()
        plt.scatter(X[:, 0].numpy(), y.numpy(), s=10, label='True Data', color='blue', alpha=0.5)
        plt.scatter(X[:, 0].numpy() + np.random.normal(0, 0.05, size=x_values.shape),
                    predicted + np.random.normal(0, 0.05, size=predicted.shape),
                    s=10, label='Predicted', color='orange', alpha=0.5)
        plt.plot(x_values, p(x_values), "r--", label='Regression Line')  # Dashed red line

        plt.xlabel('Feature 1 (X[:, 0])')
        plt.ylabel('Target (y)')
        plt.title(f'Epoch {epoch + 1}')
        plt.legend()
        plt.pause(0.01)  # Short pause for dynamic update

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# --- 5. Prediction ---
# (New data should have the same number of features as your training data)
new_data = torch.randn(1, 5)  # Example new data point
predicted_value = model(new_data)
print(f'new_data: {new_data}')
print(f'Predicted value: {predicted_value.item():.4f}')

# Final plot
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()