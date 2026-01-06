import numpy as np
import matplotlib.pyplot as plt

# ReLU function
def relu(z):
    return np.maximum(0, z)

# Create grid of points
x1 = np.linspace(0, 1, 300)
x2 = np.linspace(0, 1, 300)
X1, X2 = np.meshgrid(x1, x2)

# Compute network output
Z1 = X1 - 0.5
Z2 = X2 - 0.5
A1 = relu(Z1)
A2 = relu(Z2)
Y = A1 + A2

# Plot contour (decision regions)
plt.figure(figsize=(7, 6))
contour = plt.contourf(X1, X2, Y, levels=20, cmap="viridis")
plt.colorbar(contour, label="Output y")

# Draw ReLU boundaries
plt.axvline(0.5, color="red", linestyle="--", label="x1 = 0.5")
plt.axhline(0.5, color="blue", linestyle="--", label="x2 = 0.5")

# Plot sample points
A = (0.6, 0.6)
B = (0.2, 0.6)

plt.scatter(*A, color="white", edgecolors="black", s=100, label="Point A (y=0.2)")
plt.scatter(*B, color="orange", edgecolors="black", s=100, label="Point B (y=0.1)")

# Labels
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("ReLU Neural Network – Piecewise Linear Regions")
plt.legend()
plt.show()
