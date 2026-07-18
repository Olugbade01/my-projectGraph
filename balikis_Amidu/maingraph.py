"""
APPLICATION OF COMMON FIXED POINT THEOREMS
IN HILBERT SPACE TO NONLINEAR EQUATIONS

Python Codes for the Numerical Examples in Chapter Four

Author: AMIDU BALIKIS OMOTOYOSI
Supervised by Dr. A. Y. AKINYELE
Department of Mathematics
University of Ilorin
2026
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Example 4.1
# =====================================================

# Data from the iteration table
n = np.array([0, 1, 2, 3])

Txn = np.array([
    1.5000,
    1.4167,
    1.4142,
    1.4142
])

xn = np.array([
    1.0000,
    1.5000,
    1.4167,
    1.4142
])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=25, azim=225)

ax.plot(
    n,
    Txn,
    xn,
    linestyle=':',
    linewidth=2,
    marker='o',
    markersize=6
)

ax.scatter(
    n[-1],
    Txn[-1],
    xn[-1],
    s=180,
    marker='o',
    label='Fixed Point'
)

ax.text(
    n[-1],
    Txn[-1],
    xn[-1],
    f'({n[-1]}, {Txn[-1]:.4f}, {xn[-1]:.4f})',
    fontsize=10
)

ax.set_xlabel('Iteration Number, n')
ax.set_ylabel(r'$T(x_n)$')
ax.set_zlabel(r'$x_n$')

ax.set_title(
    '3D Convergence Graph for Example 4.1',
    pad=20
)

ax.legend()

plt.savefig(
    'Example4_1_Graph.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# =====================================================
# Example 4.2
# =====================================================

n = np.array([0, 1, 2, 3, 4, 5, 6])

Txn = np.array([
    1.2599,
    1.3123,
    1.3224,
    1.3243,
    1.3247,
    1.3248,
    1.3248
])

xn = np.array([
    1.0000,
    1.2599,
    1.3123,
    1.3224,
    1.3243,
    1.3247,
    1.3248
])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=25, azim=225)

ax.plot(
    n,
    Txn,
    xn,
    linestyle=':',
    linewidth=2,
    marker='o',
    markersize=6
)

ax.scatter(
    n[-1],
    Txn[-1],
    xn[-1],
    s=180,
    marker='o',
    label='Fixed Point'
)

ax.text(
    n[-1],
    Txn[-1],
    xn[-1],
    f'({n[-1]}, {Txn[-1]:.4f}, {xn[-1]:.4f})',
    fontsize=10
)

ax.set_xlabel('Iteration Number, n')
ax.set_ylabel(r'$T(x_n)$')
ax.set_zlabel(r'$x_n$')

ax.set_title(
    '3D Convergence Graph for Example 4.2',
    pad=20
)

ax.legend()

plt.savefig(
    'Example4_2_Graph.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# =====================================================
# Example 4.3
# =====================================================

n = np.arange(15)

xn = np.array([
    0.0000,
    1.0000,
    0.3679,
    0.6922,
    0.5005,
    0.6062,
    0.5452,
    0.5796,
    0.5601,
    0.5712,
    0.5659,
    0.5679,
    0.5670,
    0.5672,
    0.5671
])

Txn = np.array([
    1.0000,
    0.3679,
    0.6922,
    0.5005,
    0.6062,
    0.5452,
    0.5796,
    0.5601,
    0.5712,
    0.5659,
    0.5679,
    0.5670,
    0.5672,
    0.5671,
    0.5671
])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=25, azim=225)

ax.plot(
    n,
    Txn,
    xn,
    linestyle=':',
    linewidth=2,
    marker='o',
    markersize=6
)

ax.scatter(
    n[-1],
    Txn[-1],
    xn[-1],
    s=180,
    label='Fixed Point'
)

ax.text(
    n[-1],
    Txn[-1],
    xn[-1],
    f'({n[-1]}, {Txn[-1]:.4f}, {xn[-1]:.4f})'
)

ax.set_xlabel('Iteration Number, n')
ax.set_ylabel(r'$T(x_n)$')
ax.set_zlabel(r'$x_n$')

ax.set_title(
    '3D Convergence Graph for Example 4.3',
    pad=20
)

ax.legend()

plt.savefig(
    'Example4_3_Graph.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =====================================================
# Example 4.4
# =====================================================

n = np.arange(20)

Txn = np.array([
    0.8776,
    0.6390,
    0.8027,
    0.6948,
    0.7682,
    0.7192,
    0.7524,
    0.7301,
    0.7451,
    0.7356,
    0.7411,
    0.7375,
    0.7401,
    0.7385,
    0.7396,
    0.7389,
    0.7393,
    0.7391,
    0.7392,
    0.7392
])

xn = np.array([
    0.5000,
    0.8776,
    0.6390,
    0.8027,
    0.6948,
    0.7682,
    0.7192,
    0.7524,
    0.7301,
    0.7451,
    0.7356,
    0.7411,
    0.7375,
    0.7401,
    0.7385,
    0.7396,
    0.7389,
    0.7393,
    0.7391,
    0.7392
])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=25, azim=225)

ax.plot(
    n,
    Txn,
    xn,
    linestyle=':',
    linewidth=2,
    marker='o',
    markersize=5
)

ax.scatter(
    n[-1],
    Txn[-1],
    xn[-1],
    s=180,
    label='Fixed Point'
)

ax.text(
    n[-1],
    Txn[-1],
    xn[-1],
    f'({n[-1]}, {Txn[-1]:.4f}, {xn[-1]:.4f})',
    fontsize=10
)

ax.set_xlabel('Iteration Number, n')
ax.set_ylabel(r'$T(x_n)$')
ax.set_zlabel(r'$x_n$')

ax.set_title(
    '3D Convergence Graph for Example 4.4',
    pad=20
)

ax.legend()

plt.savefig(
    'Example4_4_Graph.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# =====================================================
# Example 4.5
# =====================================================

n = np.arange(20)

Txn = np.array([
    0.8148,
    0.8879,
    0.9297,
    0.9554,
    0.9719,
    0.9826,
    0.9897,
    0.9945,
    0.9963,
    0.9975,
    0.9983,
    0.9989,
    0.9993,
    0.9995,
    0.9996,
    0.9997,
    0.9998,
    0.9999,
    1.0000,
    1.0000
])

xn = np.array([
    0.6667,
    0.8148,
    0.8879,
    0.9297,
    0.9554,
    0.9719,
    0.9826,
    0.9897,
    0.9945,
    0.9963,
    0.9975,
    0.9983,
    0.9989,
    0.9993,
    0.9995,
    0.9996,
    0.9997,
    0.9998,
    0.9999,
    1.0000
])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=25, azim=225)

ax.plot(
    n,
    Txn,
    xn,
    linestyle=':',
    linewidth=2,
    marker='o',
    markersize=5
)

ax.scatter(
    n[-1],
    Txn[-1],
    xn[-1],
    s=180,
    label='Fixed Point'
)

ax.text(
    n[-1],
    Txn[-1],
    xn[-1],
    f'({n[-1]}, {Txn[-1]:.4f}, {xn[-1]:.4f})'
)

ax.set_xlabel('Iteration Number, n')
ax.set_ylabel(r'$T(x_n)$')
ax.set_zlabel(r'$x_n$')

ax.set_title(
    '3D Convergence Graph for Example 4.5',
    pad=20
)

ax.legend()

plt.savefig(
    'Example4_5_Graph.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# =====================================================
# Example 4.6
# =====================================================

n = np.arange(11)

xn = np.array([
    0.0000,
    0.6931,
    0.9905,
    1.0958,
    1.1297,
    1.1403,
    1.1436,
    1.1446,
    1.1450,
    1.1451,
    1.1452
])

Txn = np.array([
    0.6931,
    0.9905,
    1.0958,
    1.1297,
    1.1403,
    1.1436,
    1.1446,
    1.1450,
    1.1451,
    1.1452,
    1.1452
])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=25, azim=225)

ax.plot(
    n,
    Txn,
    xn,
    linestyle=':',
    linewidth=2,
    marker='o',
    markersize=6
)

ax.scatter(
    n[-1],
    Txn[-1],
    xn[-1],
    s=180,
    label='Fixed Point'
)

ax.text(
    n[-1],
    Txn[-1],
    xn[-1],
    f'({n[-1]}, {Txn[-1]:.4f}, {xn[-1]:.4f})'
)

ax.set_xlabel('Iteration Number, n')
ax.set_ylabel(r'$T(x_n)$')
ax.set_zlabel(r'$x_n$')

ax.set_title(
    '3D Convergence Graph for Example 4.6',
    pad=20
)

ax.legend()

plt.savefig(
    'Example4_6_Graph.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# =====================================================
# Example 4.7
# =====================================================

n = np.arange(10)

xn = np.array([
    1.0000,
    1.9129,
    2.0536,
    2.0840,
    2.0910,
    2.0930,
    2.0938,
    2.0942,
    2.0944,
    2.0945
])

Txn = np.array([
    1.9129,
    2.0536,
    2.0840,
    2.0910,
    2.0930,
    2.0938,
    2.0942,
    2.0944,
    2.0945,
    2.0945
])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.view_init(elev=25, azim=225)

ax.plot(
    n,
    Txn,
    xn,
    linestyle=':',
    linewidth=2,
    marker='o',
    markersize=6
)

ax.scatter(
    n[-1],
    Txn[-1],
    xn[-1],
    s=180,
    label='Fixed Point'
)

ax.text(
    n[-1],
    Txn[-1],
    xn[-1],
    f'({n[-1]}, {Txn[-1]:.4f}, {xn[-1]:.4f})'
)

ax.set_xlabel('Iteration Number, n')
ax.set_ylabel(r'$T(x_n)$')
ax.set_zlabel(r'$x_n$')

ax.set_title(
    '3D Convergence Graph for Example 4.7',
    pad=20
)

ax.legend()

plt.savefig(
    'Example4_7_Graph.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(f"All files saved")