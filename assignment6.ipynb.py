import numpy as np
import matplotlib.pyplot as plt

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Create synthetic 5x6 binary patterns for A, B, and C
def create_letter_data():
    A = np.array([
        [0,1,1,1,0],
        [1,0,0,0,1],
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
    ])

    B = np.array([
        [1,1,1,1,0],
        [1,0,0,0,1],
        [1,1,1,1,0],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,1,1,1,0],
    ])

    C = np.array([
        [0,1,1,1,1],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [0,1,1,1,1],
    ])

    # Flatten each pattern to 1D and normalize to float
    X = np.array([A.flatten(), B.flatten(), C.flatten()], dtype=np.float32)
    y = np.array([[1,0,0], [0,1,0], [0,0,1]])  # One-hot encoded labels: A, B, C

    return X, y

# Initialize weights randomly
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.1
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Feedforward
def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backpropagation
def backprop(X, y, z1, a1, z2, a2, W2):
    m = X.shape[0]
    dz2 = (a2 - y) * sigmoid_derivative(a2)
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# Training
def train(X, y, hidden_size=10, epochs=1000, lr=0.1):
    input_size = X.shape[1]
    output_size = y.shape[1]

    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    loss_history = []

    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)

        loss = np.mean((y - a2) ** 2)
        loss_history.append(loss)

        dW1, db1, dW2, db2 = backprop(X, y, z1, a1, z2, a2, W2)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss:.4f}")

    return W1, b1, W2, b2, loss_history

# Predict
def predict(X, W1, b1, W2, b2):
    _, _, _, a2 = forward(X, W1, b1, W2, b2)
    return np.argmax(a2, axis=1)

# Main execution
X, y = create_letter_data()
W1, b1, W2, b2, loss_history = train(X, y, hidden_size=10, epochs=1000, lr=0.5)

# Plot training loss
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Time")
plt.grid(True)
plt.show()

# Prediction & visualization
preds = predict(X, W1, b1, W2, b2)
letters = ['A', 'B', 'C']
for i, pred in enumerate(preds):
    plt.imshow(X[i].reshape(6, 5), cmap='gray')
    plt.title(f"Predicted: {letters[pred]}")
    plt.axis('off')
    plt.show()
