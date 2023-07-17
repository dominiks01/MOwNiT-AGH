import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

def schrodinger_eq(x, y, n1, n2):
    psi, psi_conj = y[:, 0:1], y[:, 1:2]
    psi_x, psi_y = dde.grad.jacobian(y, x, i=0, j=0), dde.grad.jacobian(y, x, i=1, j=1)
    psi_xx, psi_yy = dde.grad.hessian(y, x, component=0), dde.grad.hessian(y, x, component=1)
    eq1 = 1j * (n1**2 - n2**2) * psi + psi_xx + psi_yy - (n1**2 + n2**2) * psi_conj * psi
    eq2 = -1j * (n1**2 - n2**2) * psi_conj + psi_xx + psi_yy - (n1**2 + n2**2) * psi_conj * psi
    return eq1, eq2

def boundary_conditions(x, on_boundary):
    if on_boundary:
        if np.isclose(x[0], 1):
            return True
        if np.isclose(x[1], 1):
            return True
    return False

geom = dde.geometry.Rectangle(0, 0, 1, 1)
data = dde.data.PDE(geom, schrodinger_eq, boundary_conditions, num_domain=1000, num_boundary=100)
net = dde.maps.FNN([4] + [32] * 3 + [2], "tanh")
model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(epochs=50000)

x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
x_grid, y_grid = np.meshgrid(x, y)
points = np.vstack([x_grid.ravel(), y_grid.ravel(), np.ones_like(x_grid.ravel()), np.ones_like(y_grid.ravel())]).T
pred = model.predict(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], np.real(pred[:, 0]), c=np.real(pred[:, 0]), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Re(psi)')
plt.show()