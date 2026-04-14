"""
heat_semigroup.py
=================
Application of C0-Semigroup Theory to Heat Management in Microprocessors
Author  : Akeeb Abubakr Busayo
Chapter : 3 & 4 — Full Analysis, Analytical Comparison, and Numerical Simulation

Steps performed
---------------
1. Solve the 1-D transcendental equation for Robin eigenvalues beta_m, gamma_n
2. Build normalised Robin eigenfunctions X_m(x), Y_n(y)
3. Compute Fourier-Robin coefficients c_mn for the Gaussian initial condition
4. Construct the analytical eigenfunction-series solution
5. Solve the same problem numerically via the Crank-Nicolson scheme
6. Compare numerical max-temperature decay against the analytical bound e^{lambda_11 * t}
7. Produce four heatmaps and the decay comparison plot
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
from scipy.linalg import solve_banded
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1.  PARAMETERS  (Table 3.2 in Chapter 3)
# ============================================================
Lx   = 0.01          # chip length  [m]
Ly   = 0.01          # chip width   [m]
alpha= 8.8e-5        # thermal diffusivity of silicon [m^2/s]
h    = 500.0         # convective heat-transfer coefficient [W/m^2 K]
u_inf= 300.0         # ambient temperature [K]
Q    = 50.0          # hotspot peak amplitude [K]
sigma= 0.002         # hotspot Gaussian width [m]
Nx   = 50            # grid points in x
Ny   = 50            # grid points in y
dt   = 1e-4          # time step [s]
T_end= 0.1           # total simulation time [s]
N_modes = 20         # Robin modes per direction for analytical series

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
x_grid = np.linspace(0, Lx, Nx)
y_grid = np.linspace(0, Ly, Ny)
XX, YY = np.meshgrid(x_grid, y_grid, indexing='ij')  # shape (Nx, Ny)

# ============================================================
# 2.  ANALYTICAL PART — ROBIN EIGENVALUE PROBLEM
# ============================================================

def transcendental(beta, L, alpha, h):
    """
    Transcendental equation (3.36):
        tan(beta * L) = 2*alpha*h*beta / (alpha^2*beta^2 - h^2)
    Rearranged as F(beta)=0 for root-finding.
    """
    lhs = np.tan(beta * L)
    denom = alpha**2 * beta**2 - h**2
    if abs(denom) < 1e-30:
        return np.inf
    rhs = 2.0 * alpha * h * beta / denom
    return lhs - rhs

def find_robin_eigenvalues(L, alpha, h, N):
    """
    Find the first N positive roots beta_1 < beta_2 < ... of the
    transcendental equation by scanning for sign changes.
    """
    betas = []
    # Singularities of tan at (2k-1)*pi/(2L) and of rhs at h/alpha
    candidates = []
    for k in range(1, 5*N):
        candidates.append((2*k-1)*np.pi / (2*L))
    candidates.append(h/alpha)
    candidates = sorted(set(candidates))

    # also scan from near zero
    beta_scan = np.linspace(1e-3, N * np.pi / L * 1.5, 40000)
    signs = np.array([np.sign(transcendental(b, L, alpha, h)) for b in beta_scan])
    for i in range(len(signs)-1):
        if signs[i] * signs[i+1] < 0:
            # narrow bracket — avoid singularities
            a_, b_ = beta_scan[i], beta_scan[i+1]
            try:
                root = brentq(transcendental, a_+1e-10, b_-1e-10,
                              args=(L, alpha, h), xtol=1e-14)
                # discard if it's a singularity
                if abs(np.tan(root*L)) < 1e8:
                    betas.append(root)
            except Exception:
                pass
        if len(betas) >= N:
            break
    betas = sorted(set([round(b,12) for b in betas]))[:N]
    return np.array(betas)

print("Computing Robin eigenvalues ...")
betas  = find_robin_eigenvalues(Lx, alpha, h, N_modes)
gammas = find_robin_eigenvalues(Ly, alpha, h, N_modes)
print(f"  beta_1  = {betas[0]:.4f}  rad/m")
print(f"  gamma_1 = {gammas[0]:.4f} rad/m")

# Eigenvalues lambda_mn (all strictly negative)
lambda_mn = -alpha * (betas[:,None]**2 + gammas[None,:]**2)  # (M,N)
lambda_11  = lambda_mn[0,0]
tau        = 1.0 / abs(lambda_11)
print(f"  lambda_11 = {lambda_11:.4f} s^-1   (dominant decay rate)")
print(f"  e-folding time tau = {tau:.4f} s")

def eigenfunction_1d(x, beta, L, alpha, h):
    """
    Un-normalised Robin eigenfunction X(x) = cos(beta*x) + (h/(alpha*beta))*sin(beta*x)
    """
    return np.cos(beta*x) + (h/(alpha*beta))*np.sin(beta*x)

def norm_1d(beta, L, alpha, h):
    """L2-norm of X_beta on (0,L) computed analytically."""
    f = lambda x: eigenfunction_1d(x, beta, L, alpha, h)**2
    val, _ = quad(f, 0, L, limit=200)
    return np.sqrt(val)

print("Normalising eigenfunctions ...")
X_norm = np.array([norm_1d(b, Lx, alpha, h) for b in betas])
Y_norm = np.array([norm_1d(g, Ly, alpha, h) for g in gammas])

# ============================================================
# 3.  FOURIER-ROBIN COEFFICIENTS  (equation 3.38)
# ============================================================
# v0(x,y) = u0(x,y) - u_inf = Q * exp(-((x-Lx/2)^2+(y-Ly/2)^2)/(2*sigma^2))

def coeff_1d(beta, norm, L, alpha, h, sigma, centre):
    """
    c_m = integral_0^L  X_m(x) * exp(-(x-centre)^2/(2*sigma^2)) dx  /  ||X_m||
    Evaluated by Gaussian quadrature.
    """
    f = lambda x: eigenfunction_1d(x, beta, L, alpha, h) * \
                  np.exp(-(x - centre)**2 / (2*sigma**2))
    val, _ = quad(f, 0, L, limit=400)
    return val / norm

print("Computing Fourier-Robin coefficients ...")
cx = np.array([coeff_1d(betas[m],  X_norm[m],  Lx, alpha, h, sigma, Lx/2)
               for m in range(N_modes)])
cy = np.array([coeff_1d(gammas[n], Y_norm[n],  Ly, alpha, h, sigma, Ly/2)
               for n in range(N_modes)])
c_mn = Q * np.outer(cx, cy)   # shape (M,N)

# ============================================================
# 4.  ANALYTICAL SOLUTION — eigenfunction series evaluation
# ============================================================
def analytical_solution(t, betas, gammas, c_mn, lambda_mn,
                         X_norm, Y_norm, x_grid, y_grid,
                         Lx, Ly, alpha, h, u_inf):
    """
    u(x,y,t) = u_inf + sum_{m,n} c_mn * e^{lambda_mn * t} * phi_mn(x,y)
    Returns array of shape (Nx, Ny).
    """
    u = np.full((len(x_grid), len(y_grid)), u_inf)
    for m in range(len(betas)):
        Xm = eigenfunction_1d(x_grid, betas[m], Lx, alpha, h) / X_norm[m]
        for n in range(len(gammas)):
            decay = np.exp(lambda_mn[m,n] * t)
            if abs(decay) < 1e-16:
                continue
            Yn = eigenfunction_1d(y_grid, gammas[n], Ly, alpha, h) / Y_norm[n]
            u += c_mn[m,n] * decay * np.outer(Xm, Yn)
    return u

print("Evaluating analytical solution at t=0, 0.025, 0.05, 0.1 s ...")
t_snapshots = [0.0, 0.025, 0.05, 0.1]
u_analytical = {}
for t_ in t_snapshots:
    u_analytical[t_] = analytical_solution(
        t_, betas, gammas, c_mn, lambda_mn,
        X_norm, Y_norm, x_grid, y_grid, Lx, Ly, alpha, h, u_inf)
print("  Done.")

# ============================================================
# 5.  NUMERICAL SCHEME — Crank-Nicolson
# ============================================================
# Assemble the 1-D tridiagonal matrix L_x (alpha * d^2/dx^2 with Robin BCs)
# and similarly L_y, then use operator splitting:
#   each full step = half-step implicit in x, half-step implicit in y
# This is the ADI (Alternating Direction Implicit) variant of Crank-Nicolson,
# exactly the Pade(1,1) approximation described in Section 3.8.

def build_1d_matrix(N, d, alpha, h):
    """
    Returns the tridiagonal finite-difference matrix for alpha*d^2/dx^2
    with Robin BCs:  alpha * u'(0) + h*u(0) = 0
                    -alpha * u'(L) + h*u(L) = 0
    Shape: (N,N).  The first-order Robin ghost-point elimination is used.
    """
    r = alpha / d**2
    # Interior second-order difference
    diag  = -2*r * np.ones(N)
    upper =    r * np.ones(N-1)
    lower =    r * np.ones(N-1)

    # Left Robin BC: alpha*(u[1]-u[0])/dx = -h*u[0]
    #   => effective coefficient for u[0]: -r + alpha*h/(alpha+h*d)... see derivation
    # Using ghost-point:  u[-1] = ((alpha/dx - h)/(alpha/dx + h)) * u[1]
    kappa = (alpha/d - h) / (alpha/d + h)
    diag[0]  = r * (kappa - 2)
    upper[0] = r * (1 - kappa)    # net effect on u[1] absorbed here

    # Right Robin BC: -alpha*(u[N-1]-u[N-2])/dx = -h*u[N-1]
    kappa_r = (alpha/d - h) / (alpha/d + h)
    diag[-1]   = r * (kappa_r - 2)
    lower[-1]  = r * (1 - kappa_r)

    return diag, upper, lower

def tridiag_solve(diag, upper, lower, rhs):
    """
    Thomas algorithm for tridiagonal system. O(N).
    """
    n = len(diag)
    c = upper.copy().astype(float)
    d = rhs.copy().astype(float)
    a = lower.copy().astype(float)
    b = diag.copy().astype(float)
    # Forward sweep
    for i in range(1, n):
        m_ = a[i-1] / b[i-1]
        b[i] -= m_ * c[i-1]
        d[i] -= m_ * d[i-1]
    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i]*x[i+1]) / b[i]
    return x

# Build matrices
diag_x, up_x, lo_x = build_1d_matrix(Nx, dx, alpha, h)
diag_y, up_y, lo_y = build_1d_matrix(Ny, dy, alpha, h)

# Crank-Nicolson in 2D via Peaceman-Rachford ADI:
#   Step 1:  (I - dt/2 * Ax) u* = (I + dt/2 * Ay) u^k   (solve along x for each row)
#   Step 2:  (I - dt/2 * Ay) u^{k+1} = (I + dt/2 * Ax) u*   (solve along y for each col)

def cn_step(u):
    """One full Crank-Nicolson ADI time step. Returns u^{k+1}."""
    r_x = dt/2
    r_y = dt/2
    # ---- Half-step: implicit x, explicit y ----
    # RHS row j: (I + r_y * Ay) @ u[:,j]
    Ny_rows = u.shape[1]
    Nx_cols = u.shape[0]
    u_star = np.zeros_like(u)
    for j in range(Ny_rows):
        rhs = u[:,j].copy()
        # apply (I + r_y * Ay) in y direction at this j ... 
        # For ADI we apply explicit Ay to the column first:
        # Since we're doing x-implicit / y-explicit in first half:
        rhs_j = rhs + r_y * (
            np.concatenate([[diag_y[0]*u[j,0] + (up_y[0] if 0<Ny_rows-1 else 0)*u[j,1]],
                             [lo_y[i-1]*u[j,i-1] + diag_y[i]*u[j,i] + (up_y[i]*u[j,i+1] if i<Ny_rows-1 else 0)
                              for i in range(1, Ny_rows-1)],
                             [lo_y[-1]*u[j,-2] + diag_y[-1]*u[j,-1]]]))
        # Solve (I - r_x * Ax) u* = rhs_j
        lhs_diag  = 1 - r_x * diag_x
        lhs_upper = -r_x * up_x
        lhs_lower = -r_x * lo_x
        u_star[:,j] = tridiag_solve(lhs_diag, lhs_upper, lhs_lower, rhs_j)

    # ---- Half-step: implicit y, explicit x ----
    u_new = np.zeros_like(u)
    for i in range(Nx_cols):
        rhs_i = u_star[i,:] + r_x * (
            np.concatenate([[diag_x[0]*u_star[0,i] + (up_x[0]*u_star[1,i] if Nx_cols>1 else 0)],
                             [lo_x[k-1]*u_star[k-1,i] + diag_x[k]*u_star[k,i] + (up_x[k]*u_star[k+1,i] if k<Nx_cols-1 else 0)
                              for k in range(1, Nx_cols-1)],
                             [lo_x[-1]*u_star[-2,i] + diag_x[-1]*u_star[-1,i]]]))
        lhs_diag  = 1 - r_y * diag_y
        lhs_upper = -r_y * up_y
        lhs_lower = -r_y * lo_y
        u_new[i,:] = tridiag_solve(lhs_diag, lhs_upper, lhs_lower, rhs_i)
    return u_new

# Initial condition
u0 = u_inf + Q * np.exp(-((XX - Lx/2)**2 + (YY - Ly/2)**2) / (2*sigma**2))
u  = u0.copy()

# Time-stepping
n_steps  = int(T_end / dt)
t_snap   = [0.0, 0.025, 0.05, 0.1]
snap_steps = [int(t/dt) for t in t_snap]

u_numerical = {0.0: u0.copy()}
max_temps    = [u0.max()]What the script does, step by step

Step 1 — Robin eigenvalues. The transcendental equation (3.36) is solved numerically using Brent's method. The first root came out as β1=314.15β1​=314.15 rad/m, which is extremely close to π/Lx=314.16π/Lx​=314.16 rad/m — exactly as the Chapter 3 bound predicted.

Step 2 — Eigenvalue λ11λ11​. Computed as −α(β12+γ12)=−17.37−α(β12​+γ12​)=−17.37 s⁻¹, giving τ=0.0576τ=0.0576 s. This confirms the analytical estimate of ≈17.4≈17.4 s⁻¹ from equation (3.41).

Step 3 — Fourier–Robin coefficients cmncmn​. The integrals in equation (3.38) were evaluated using SciPy's Gaussian quadrature.

Step 4 — Analytical series solution. Evaluated using all 20×20 eigenpairs at each snapshot time.

Step 5 — Crank–Nicolson simulation. The ADI (Alternating Direction Implicit) variant of the Crank–Nicolson scheme was used — this is the Padé(1,1) approximation described in Section 3.8, and it is unconditionally stable.
Key results to report in Chapter 4
Quantity	Value
β1β1​ (dominant Robin eigenvalue) 	314.15314.15 rad/m
λ11λ11​ (dominant decay rate) 	−17.37−17.37 s⁻¹
e-folding time ττ	0.05760.0576 s
Max temperature at t=0t=0	350350 K 
times        = [0.0]

print(f"Running Crank-Nicolson: {n_steps} steps ...")
for k in range(1, n_steps+1):
    u = cn_step(u)
    t_now = k * dt
    times.append(t_now)
    max_temps.append(u.max())
    if k in snap_steps[1:]:
        u_numerical[t_now] = u.copy()
        print(f"  t = {t_now:.4f} s   max T = {u.max():.4f} K")

print("  Simulation complete.")

# ============================================================
# 6.  ANALYTICAL BOUND  e^{lambda_11 * t} * ||v0||  + u_inf
# ============================================================
times_arr   = np.array(times)
v0_max      = Q        # ||v0||_inf at t=0
bound       = u_inf + v0_max * np.exp(lambda_11 * times_arr)
# Also compute analytical max-temperature series
print("Computing analytical max-temp series ...")
t_fine = np.linspace(0, T_end, 201)
ana_max = []
for t_ in t_fine:
    u_a = analytical_solution(t_, betas, gammas, c_mn, lambda_mn,
                               X_norm, Y_norm, x_grid, y_grid,
                               Lx, Ly, alpha, h, u_inf)
    ana_max.append(u_a.max())
ana_max = np.array(ana_max)

# ============================================================
# 7.  FIGURES
# ============================================================
print("Generating figures ...")
t_keys = [0.0, 0.025, 0.05, 0.1]

# ---- Figure 1: 4 heatmaps (numerical) ----------------------
fig1, axes = plt.subplots(2, 2, figsize=(11, 9))
fig1.suptitle(
    "Temperature Distribution — Numerical Solution (Crank–Nicolson)\n"
    r"$\Omega = (0,0.01)\times(0,0.01)$ m,  $\alpha=8.8\times10^{-5}$ m²/s,  "
    r"$h=500$ W/m²K",
    fontsize=13, y=1.01)

vmin = u_inf
vmax = u_inf + Q + 2

for ax, t_ in zip(axes.flat, t_keys):
    data = u_numerical.get(t_, u0)
    im = ax.pcolormesh(
        x_grid*1e3, y_grid*1e3, data.T,
        cmap='hot', shading='auto', vmin=vmin, vmax=vmax)
    cb = fig1.colorbar(im, ax=ax, label='Temperature [K]')
    ax.set_title(f"$t = {t_:.3f}$ s", fontsize=12)
    ax.set_xlabel("$x$ [mm]", fontsize=10)
    ax.set_ylabel("$y$ [mm]", fontsize=10)
    # Mark hotspot centre
    ax.plot(Lx/2*1e3, Ly/2*1e3, 'c+', ms=10, mew=1.5, label='Hotspot centre')
    ax.legend(fontsize=8, loc='upper right')

fig1.tight_layout()
fig1.savefig("/mnt/user-data/outputs/heatmaps_numerical.png", dpi=150, bbox_inches='tight')
plt.close(fig1)

# ---- Figure 2: 4 heatmaps (analytical series) --------------
fig2, axes2 = plt.subplots(2, 2, figsize=(11, 9))
fig2.suptitle(
    "Temperature Distribution — Analytical Eigenfunction Series\n"
    r"$u(x,y,t)=u_\infty + \sum_{m,n} c_{mn}\,e^{\lambda_{mn}t}\,\varphi_{mn}(x,y)$",
    fontsize=13, y=1.01)

for ax, t_ in zip(axes2.flat, t_keys):
    data = u_analytical[t_]
    im = ax.pcolormesh(
        x_grid*1e3, y_grid*1e3, data.T,
        cmap='hot', shading='auto', vmin=vmin, vmax=vmax)
    fig2.colorbar(im, ax=ax, label='Temperature [K]')
    ax.set_title(f"$t = {t_:.3f}$ s", fontsize=12)
    ax.set_xlabel("$x$ [mm]", fontsize=10)
    ax.set_ylabel("$y$ [mm]", fontsize=10)
    ax.plot(Lx/2*1e3, Ly/2*1e3, 'c+', ms=10, mew=1.5)

fig2.tight_layout()
fig2.savefig("/mnt/user-data/outputs/heatmaps_analytical.png", dpi=150, bbox_inches='tight')
plt.close(fig2)

# ---- Figure 3: Max-temperature decay comparison ------------
fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.plot(times_arr, np.array(max_temps), 'b-', lw=1.5, label='Numerical (Crank–Nicolson)')
ax3.plot(t_fine,    ana_max,             'g--', lw=1.5, label='Analytical series')
ax3.plot(times_arr, bound,               'r:',  lw=2.0,
         label=r'Semigroup bound $u_\infty + Q\,e^{\lambda_{11}t}$')
ax3.axhline(u_inf, color='gray', ls='-.', lw=1, label=f'Ambient $u_\\infty={u_inf}$ K')

# Annotate e-folding time
ax3.axvline(tau, color='purple', ls=':', lw=1)
ax3.text(tau+0.001, u_inf+46, r'$\tau=1/|\lambda_{11}|\approx$'+f'{tau:.3f} s',
         color='purple', fontsize=9)

ax3.set_xlabel("Time $t$ [s]", fontsize=12)
ax3.set_ylabel("Maximum temperature [K]", fontsize=12)
ax3.set_title(
    "Max-Temperature Decay: Numerical vs Analytical vs Semigroup Bound\n"
    r"$\lambda_{11}\approx$"+f"{lambda_11:.2f} s⁻¹,  "+
    r"$\tau\approx$"+f"{tau:.4f} s",
    fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("/mnt/user-data/outputs/max_temp_decay.png", dpi=150, bbox_inches='tight')
plt.close(fig3)

# ---- Figure 4: Cross-section at y = Ly/2 (centre line) ----
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
mid_j = Ny//2
colors = ['navy', 'steelblue', 'darkorange', 'firebrick']
for t_, col in zip(t_keys, colors):
    u_n = u_numerical.get(t_, u0)
    u_a = u_analytical[t_]
    axes4[0].plot(x_grid*1e3, u_n[:,mid_j], color=col, lw=1.8, label=f't={t_:.3f} s')
    axes4[1].plot(x_grid*1e3, u_a[:,mid_j], color=col, lw=1.8, ls='--', label=f't={t_:.3f} s')

for ax, title in zip(axes4, ['Numerical', 'Analytical']):
    ax.axhline(u_inf, color='gray', ls=':', lw=1)
    ax.set_xlabel("$x$ [mm]", fontsize=11)
    ax.set_ylabel("Temperature [K]", fontsize=11)
    ax.set_title(f"Centre-line profile ($y=L_y/2$)  —  {title}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig("/mnt/user-data/outputs/centreline_profiles.png", dpi=150, bbox_inches='tight')
plt.close(fig4)

# ---- Figure 5: Eigenvalue spectrum -------------------------
fig5, ax5 = plt.subplots(figsize=(8, 4))
M = min(8, N_modes)
lam_plot = lambda_mn[:M, :M]
im5 = ax5.imshow(lam_plot.T, origin='lower', cmap='RdYlGn_r',
                  aspect='auto')
fig5.colorbar(im5, ax=ax5, label=r'$\lambda_{mn}$ [s$^{-1}$]')
ax5.set_xticks(range(M)); ax5.set_xticklabels([f'm={i+1}' for i in range(M)], fontsize=8)
ax5.set_yticks(range(M)); ax5.set_yticklabels([f'n={i+1}' for i in range(M)], fontsize=8)
ax5.set_title(
    r"Eigenvalue spectrum $\lambda_{mn} = -\alpha(\beta_m^2+\gamma_n^2)$  "
    "(all $\leq 0$, confirming dissipativity)",
    fontsize=11)
ax5.set_xlabel("Mode index m"); ax5.set_ylabel("Mode index n")
# Annotate (1,1)
ax5.add_patch(plt.Rectangle((-0.5,-0.5),1,1, fill=False, edgecolor='blue', lw=2))
ax5.text(0, 0, f'{lambda_mn[0,0]:.1f}', ha='center', va='center',
         color='blue', fontsize=8, fontweight='bold')
fig5.tight_layout()
fig5.savefig("/mnt/user-data/outputs/eigenvalue_spectrum.png", dpi=150, bbox_inches='tight')
plt.close(fig5)

# ============================================================
# 8.  SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("SUMMARY OF KEY ANALYTICAL RESULTS")
print("="*60)
print(f"  beta_1  (first Robin eigenvalue, x) = {betas[0]:.6f} rad/m")
print(f"  gamma_1 (first Robin eigenvalue, y) = {gammas[0]:.6f} rad/m")
print(f"  lambda_11 (dominant decay rate)     = {lambda_11:.6f} s^-1")
print(f"  e-folding time tau                  = {tau:.6f} s")
print(f"  Dirichlet lower bound on |lambda_11|= {alpha*2*(np.pi/Lx)**2:.4f} s^-1  (approx 17.4)")
print(f"\n  c_11 (dominant Fourier coeff)       = {c_mn[0,0]:.6f} K")
print(f"  ||v0||_inf = Q                      = {Q:.1f} K")
print(f"\n  At t={T_end} s (Crank-Nicolson):")
print(f"    max temperature                   = {max_temps[-1]:.4f} K")
print(f"    temperature rise above ambient    = {max_temps[-1]-u_inf:.4f} K")
print(f"    semigroup bound gives max          = {bound[-1]:.4f} K")
print(f"\n  All {N_modes}x{N_modes} eigenvalues lambda_mn < 0  =>  exponential stability confirmed.")
print("="*60)
print("\nAll figures saved to /mnt/user-data/outputs/")