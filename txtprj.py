import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0              # Length of rod
T = 2.0              # Total time
nx = 20              # Spatial points
nt = 100             # Time steps

dx = L / (nx - 1)
dt = T / nt
r = dt / (dx**2)

# Grid
x = np.linspace(0, L, nx)

# Initial condition (hotspot)
u = np.exp(-100 * (x - 0.5)**2)

# Store results
U = [u.copy()]

# Create matrices A and B
A = np.zeros((nx, nx))
B = np.zeros((nx, nx))

for i in range(1, nx-1):
    A[i,i-1] = -r/2
    A[i,i]   = 1 + r
    A[i,i+1] = -r/2

    B[i,i-1] = r/2
    B[i,i]   = 1 - r
    B[i,i+1] = r/2

# Boundary conditions (Dirichlet: u=0 at edges)
A[0,0] = A[-1,-1] = 1
B[0,0] = B[-1,-1] = 1

# Time stepping
for n in range(nt):
    u = np.linalg.solve(A, B @ u)
    if n % 20 == 0:
        U.append(u.copy())

# Plot results
for i, sol in enumerate(U):
    plt.plot(x, sol, label=f"t={i*(T/5):.2f}")

plt.xlabel("Position")
plt.ylabel("Temperature")
plt.title("Heat Distribution Over Time")
plt.legend()
plt.grid()

# Save figure
plt.savefig("heat_distribution.png", dpi=300)
plt.show()