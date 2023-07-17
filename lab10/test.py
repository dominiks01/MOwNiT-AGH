import numpy as np
import deepxde as dde
from deepxde.backend import tf

# Define the Schrödinger equation
def schrodinger_eq(inputs, outputs):
    x, y = inputs[:, 0:1], inputs[:, 1:2]
    psi = outputs[:, 0:1]
    n1, n2 = inputs[:, 2:3], inputs[:, 3:4]
    d2_psi_dx2 = dde.grad.hessian(outputs, inputs, component=0, use_jax=False)[:, 0:1]
    d2_psi_dy2 = dde.grad.hessian(outputs, inputs, component=0, use_jax=False)[:, 1:2]
    eqn = 0.5 * (d2_psi_dx2 + d2_psi_dy2) + ((n1**2 + n2**2) * np.pi**2 / 8) * psi
    return eqn

# Define the domain and quantum numbers
domain = [-1, 1]
n1_values = [1, 1, 2, 2]
n2_values = [1, 2, 1, 2]

# Generate training data
data = np.random.rand(10000, 4)
data[:, 0:1] = data[:, 0:1] * (domain[1] - domain[0]) + domain[0]
data[:, 1:2] = data[:, 1:2] * (domain[1] - domain[0]) + domain[0]
data[:, 2:3] = np.random.choice(n1_values, size=(10000, 1))
data[:, 3:4] = np.random.choice(n2_values, size=(10000, 1))

# Create the deepxde model
geom = dde.geometry.Rectangle(*domain)
bc = dde.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
data = dde.data.PDE(geom, schrodinger_eq, bc, num_domain=10000, num_boundary=100)
net = dde.maps.FNN([4] + [32] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Train the model
model.compile("adam", lr=0.001)
model.train(epochs=10000)

# Generate points for plotting the results
x = np.linspace(domain[0], domain[1], 100)
y = np.linspace(domain[0], domain[1], 100)
X, Y = np.meshgrid(x, y)
points = np.column_stack([X.flatten(), Y.flatten(), np.ones(X.size), np.ones(Y.size)])

# Predict the wave function
pred = model.predict(points)

# Plot the contour plot of the predicted wave function
plt.contourf(X, Y, pred[:, 0].reshape(X.shape), cmap="viridis")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted Wave Function")
plt.colorbar(label="ψ(x, y)")
plt.grid(True)
plt.show()