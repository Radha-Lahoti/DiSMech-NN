import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

# Step 1: Define the Neural ODE Model
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),   # Input dimension is 2, hidden layer size is 50
            nn.Tanh(),          # Activation function (Tanh)
            nn.Linear(50, 2)    # Output dimension is 2
        )

    def forward(self, t, y):
        return self.net(y)  # Forward pass: returns the time derivative of the state
    

def generate_data():
    y0 = torch.tensor([[0.0, 1.0]])  # [1, 2]
    t = torch.linspace(0, 25, 100)

    # Circular motion: y = [sin(t), cos(t)]
    true_y = torch.stack([torch.sin(t), torch.cos(t)], dim=1).unsqueeze(1)  # [100, 1, 2]

    data = {'y0': y0, 't': t, 'y_true': true_y}
    return data

# Step 3: Define the Training Loop
def train_ode(model, optimizer, criterion, data):
    for epoch in range(100):                    # Number of epochs (iterations)
        optimizer.zero_grad()                   # Clear gradients from the previous step
        pred_y = odeint(model, data['y0'], data['t'])  # Solve ODE for current model state
        loss = criterion(pred_y, data['y_true'])  # Compute loss between predicted and true values
        loss.backward()                         # Backpropagate the error
        optimizer.step()                        # Update the model parameters
        if (epoch + 1) % 10 == 0:               # Print every 10 epochs
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def plot_results(model, data):
    with torch.no_grad():
        pred_y = odeint(model, data['y0'], data['t'])  # [T, 1, 2]

    t = data['t'].cpu().numpy()
    true_y = data['y_true'].squeeze(1).cpu().numpy()  # [T, 2]
    pred_y = pred_y.squeeze(1).cpu().numpy()          # [T, 2]

    plt.figure(figsize=(8,4))
    plt.plot(t, true_y[:,0], 'b', label='True $y_1$ (cos)')
    plt.plot(t, true_y[:,1], 'g', label='True $y_2$ (sin)')
    plt.plot(t, pred_y[:,0], 'b--', label='Pred $y_1$')
    plt.plot(t, pred_y[:,1], 'g--', label='Pred $y_2$')
    plt.xlabel('Time')
    plt.ylabel('States')
    plt.legend()
    plt.title('True vs Predicted Components over Time')
    plt.tight_layout()
    plt.show()

def plot_trajectory(model, data):
    with torch.no_grad():
        pred_y = odeint(model, data['y0'], data['t'])  # [T, 1, 2]

    true_y = data['y_true'].squeeze(1).cpu().numpy()  # [T, 2]
    pred_y = pred_y.squeeze(1).cpu().numpy()          # [T, 2]

    plt.figure(figsize=(5,5))
    plt.plot(true_y[:,0], true_y[:,1], 'b', label='True trajectory')
    plt.plot(pred_y[:,0], pred_y[:,1], 'r--', label='Predicted trajectory')
    plt.xlabel('$y_1$')
    plt.ylabel('$y_2$')
    plt.axis('equal')
    plt.legend()
    plt.title('Circular Motion: True vs Predicted Trajectories')
    plt.tight_layout()
    plt.show()

def main():
    # Step 4: Putting It All Together
    model = ODEFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    data = generate_data()

    train_ode(model, optimizer, criterion, data)

    plot_results(model, data)
    plot_trajectory(model, data)

if __name__ == "__main__":
    main()
