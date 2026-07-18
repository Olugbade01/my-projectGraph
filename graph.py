'''Akeeb Abubakr Busayo 

Topic: Application of Semigroups of Linear Operators In Haet Mananagement In Microprocessors 

A project submitted to the department of Mathemathematics University of Ilorin, Nigeria. In fufilment for the Award of Bachelor Of Science in Mathematic

Supervised by Dr. A. Y. Akinyele (Olizo)
On this day 12th of July, 2026.'''

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ── Output folder ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "adi_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
Lx, Ly  = 1.0, 1.0
T_end   = 0.1
dt      = 0.001
alpha   = 1.0

# ── Exact solution ─────────────────────────────────────────────────────────────
def exact(x, y, t):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2.0 * np.pi**2 * t)

# ── Thomas algorithm (tridiagonal solver) — Section 4.6 ──────────────────────
def thomas(a, d, c, b):
    M  = len(d)
    d  = d.copy().astype(float)
    b  = b.copy().astype(float)
    c  = c.copy().astype(float)

    for k in range(1, M):
        w    = a[k] / d[k-1]
        d[k] = d[k] - w * c[k-1]
        b[k] = b[k] - w * b[k-1]

    u = np.zeros(M)
    u[-1] = b[-1] / d[-1]
    for k in range(M-2, -1, -1):
        u[k] = (b[k] - c[k] * u[k+1]) / d[k]

    return u

# ── ADI solver ────────────────────────────────────────────────────────────────
def adi_solve(Nx, Ny, dt, T_end, alpha, Lx, Ly):
    dx = Lx / (Nx + 1)
    dy = Ly / (Ny + 1)

    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2

    x = np.linspace(dx, Lx - dx, Nx)
    y = np.linspace(dy, Ly - dy, Ny)

    X, Y = np.meshgrid(x, y, indexing='ij')
    u = exact(X, Y, 0.0)

    nt = int(round(T_end / dt))

    ax_sub   = np.full(Nx, -rx / 2.0)
    ax_diag  = np.full(Nx,  1.0 + rx)
    ax_super = np.full(Nx, -rx / 2.0)

    ay_sub   = np.full(Ny, -ry / 2.0)
    ay_diag  = np.full(Ny,  1.0 + ry)
    ay_super = np.full(Ny, -ry / 2.0)

    for _ in range(nt):

        u_half = np.zeros_like(u)
        for j in range(Ny):
            u_jm1 = u[:, j-1] if j > 0      else np.zeros(Nx)
            u_jp1 = u[:, j+1] if j < Ny - 1 else np.zeros(Nx)
            b1 = (ry/2.0)*u_jm1 + (1.0 - ry)*u[:, j] + (ry/2.0)*u_jp1
            u_half[:, j] = thomas(ax_sub, ax_diag, ax_super, b1)

        u_new = np.zeros_like(u_half)
        for i in range(Nx):
            u_im1 = u_half[i-1, :] if i > 0      else np.zeros(Ny)
            u_ip1 = u_half[i+1, :] if i < Nx - 1 else np.zeros(Ny)
            b2 = (rx/2.0)*u_im1 + (1.0 - rx)*u_half[i, :] + (rx/2.0)*u_ip1
            u_new[i, :] = thomas(ay_sub, ay_diag, ay_super, b2)

        u = u_new

    return u, x, y, rx, ry

# ── Convergence study ─────────────────────────────────────────────────────────
grid_sizes = [20, 40, 60]
errors     = []
cpu_times  = []

print(f"{'Grid':>10}  {'Max Error':>14}  {'CPU Time (s)':>14}")
print("-" * 44)

for N in grid_sizes:
    t0 = time.time()
    u_num, x, y, rx, ry = adi_solve(N, N, dt, T_end, alpha, Lx, Ly)
    cpu = time.time() - t0

    X, Y    = np.meshgrid(x, y, indexing='ij')
    u_exact = exact(X, Y, T_end)
    err     = np.max(np.abs(u_num - u_exact))

    errors.append(err)
    cpu_times.append(cpu)
    print(f"{N:>4}x{N:<4}  {err:>14.6e}  {cpu:>14.4f}")

# ── Plot 1: Numerical solution heatmap ───────────────────────────────────────
u_fine, x_fine, y_fine, _, _ = adi_solve(60, 60, dt, T_end, alpha, Lx, Ly)
X60, Y60 = np.meshgrid(x_fine, y_fine, indexing='ij')

fig1, ax1 = plt.subplots(figsize=(6, 5))
im = ax1.contourf(X60, Y60, u_fine, levels=20, cmap='hot')
plt.colorbar(im, ax=ax1)
ax1.set_title("ADI Numerical Solution (60×60)\nt = 0.1 s")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heatmap.png"), dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/heatmap.png")

# ── Plot 2: Error vs grid size ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(grid_sizes, errors, marker='o', color='navy', linewidth=2)
ax2.set_title("Max Error vs Grid Size")
ax2.set_xlabel("Grid Size (N × N)")
ax2.set_ylabel("Maximum Absolute Error")
ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_vs_grid.png"), dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/error_vs_grid.png")

# ── Plot 3: CPU time vs grid size ─────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.plot(grid_sizes, cpu_times, marker='s', color='darkred', linewidth=2)
ax3.set_title("CPU Time vs Grid Size")
ax3.set_xlabel("Grid Size (N × N)")
ax3.set_ylabel("CPU Time (s)")
ax3.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cpu_time_vs_grid.png"), dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/cpu_time_vs_grid.png")
