import numpy as np
import matplotlib.pyplot as plt
#Constants
R   = 69 / (2 * np.pi) * 1e-2   # Ball radius [m]
D   = 2 * R                     # Ball diameter [m]
mu  = 1.82e-5                   # Dynamic viscosity [Pa s]
rho = 1.204                     # Air density [kg/m^3]

# ── Telstar 18 drag at Sp = 0 
#These are the parameters obtained from the Goff drag fit in the other script.
#term1 is the drop of the drag crisis, term2 the turbulent plateau and term3 the
#slow supercritical rise of the tail.
a           = 0.5677
b_min       = 0.1592
v_c         = 9.9346
v_s         = 1.1298
v_min_fixed = 13.756801532109767
p           = 0.0012

def Cd0(v):
    term1 = (a - b_min) / (1 + np.exp((v - v_c) / v_s))
    term2 = b_min
    term3 = ((v - v_min_fixed) / (1 + np.exp(-(v - v_min_fixed) / v_s))) * p
    return term1 + term2 + term3

# Spin-dependence of the drag 
#b_telstar is interpolated from the spin-drag sensitivity at 30 m/s between the
#Teamgeist (lower bound) and the Brazuca (upper bound). The Telstar 18 sits in
#between, so it gets an intermediate value (K&L eq. 6).
Cd_brazuca_30   = 0.2000425746
Cd_telstar_30   = 0.1731581914
Cd_teamgeist_30 = 0.1497636901
b_teamgeist     = 0.514

b_telstar = b_teamgeist * (1 + 0.05 * (Cd_telstar_30 - Cd_teamgeist_30) / (Cd_brazuca_30 - Cd_teamgeist_30))

#Full drag including the spin term
def Cd(v, Sp):
    return Cd0(v) + b_telstar * Sp

#Reference state used by K&L to anchor the lift 
Re_ref = 333793.0
Sp_ref = 0.19
v_ref  = Re_ref * mu / (rho * D)

Cd_lam = Cd0(0.0)            #laminar maximum drag, reached as v -> 0
Cd_ref = Cd(v_ref, Sp_ref)   #drag evaluated at the reference (Re, Sp)

# ── Lift coefficient 
#alpha and beta are the published K&L 2018 values calibrated on the Teamgeist
#spinning wind-tunnel + trajectory data.
alpha = 1.15
beta  = 0.83

#Boundary-layer factor built from the drag
def S_D(v, Sp):
    cd = Cd(v, Sp)
    return np.maximum(Cd_lam - np.minimum(cd, Cd_lam), 0.0) / (Cd_lam - Cd_ref)

#Final Magnus / lift coefficient for the Telstar 18.
def C_L(Re, Sp):
    v = Re * mu / (rho * D)
    return alpha * Sp**beta * S_D(v, Sp)

#Sanity checks 
print(f"Law: C_L(Re,Sp) = {alpha:.2f} * Sp**{beta:.2f} * S_D(Re,Sp)")
print(f"  b_telstar = {b_telstar:.4f}")
print(f"  Cd_lam    = {Cd_lam:.4f}")
print(f"  Cd_ref    = {Cd_ref:.4f}")

#Check that the law reproduces the Teamgeist anchor points at the reference Re.
print("\nTeamgeist anchors at Re_ref:")
for Sp, target in [(0.06, 0.15), (0.19, 0.29), (0.31, 0.33)]:
    print(f"  Sp = {Sp:.2f}  ->  C_L = {C_L(Re_ref, Sp):.3f}   (empirical {target:.2f})")

#Limit behaviour: no spin -> no Magnus, laminar -> no Magnus, turbulent -> full.
print("\nLimit behaviour:")
print(f"  Sp -> 0            ->  C_L = {C_L(Re_ref, 1e-6):.4f}")
print(f"  v = 8  m/s (lam.)  ->  C_L = {C_L(rho * 8.0  * D / mu, 0.20):.4f}")
print(f"  v = 30 m/s (turb.) ->  C_L = {C_L(rho * 30.0 * D / mu, 0.20):.4f}")

#Sweep Sp to locate the built-in maximum. 
Sp_sweep = np.linspace(0.01, 0.9, 2000)
Cl_sweep = C_L(Re_ref, Sp_sweep)
print(f"\nC_L peaks at Sp = {Sp_sweep[np.argmax(Cl_sweep)]:.2f} and then decays")

# Ward data (not used to calibrate) 
Re_100 = np.array([1.3644, 1.8390, 2.3305, 2.8220, 3.2797, 3.7712, 4.2373]) * 1e5
Cy_100 = np.array([0.02548, 0.07209, 0.10131, 0.13247, 0.15781, 0.15804, 0.16020])
Re_300 = np.array([1.3729, 1.8390, 2.3220, 2.8220, 3.2712, 3.7627, 4.2373]) * 1e5
Cy_300 = np.array([0.26703, 0.24986, 0.25203, 0.26966, 0.28341, 0.28364, 0.28001])
Sp_100 = R * (100 * 2 * np.pi / 60) / (Re_100 * mu / (rho * D))
Sp_300 = R * (300 * 2 * np.pi / 60) / (Re_300 * mu / (rho * D))

#Plots 
c_blue, c_cyan, c_red, c_yel = "#264D75", "#2EA8E0", "#d1495b", "#edae49"
fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))

#(a) Spin shape at the turbulent reference Re. The dashed line is the bare shape
#without the boundary-layer correction; the solid line is the full law.
sp = np.linspace(0, 0.55, 400)
ax[0, 0].plot(sp, alpha * sp ** beta, "--", lw=2, color=c_cyan,
              label=r"bare shape $1.15\,Sp^{0.83}$ (Teamgeist)")
ax[0, 0].plot(sp, C_L(Re_ref, sp), lw=2.6, color=c_blue,
              label=r"full law $\times\,S_D$ (Telstar 18)")
ax[0, 0].scatter([0.06, 0.19, 0.31], [0.15, 0.29, 0.33], s=70, zorder=3,
                 color=c_red, edgecolor="k", label="Teamgeist anchors (K&L)")
ax[0, 0].set_xlabel(r"Spin parameter  $Sp$")
ax[0, 0].set_ylabel(r"$C_L$")
ax[0, 0].set_title("(a)  Spin shape at turbulent reference $Re$")
ax[0, 0].legend(fontsize=8.5)
ax[0, 0].grid(True, alpha=0.4)

#(b) Lift vs speed at a few fixed Sp. Below the critical speed the boundary layer
#is laminar and the Magnus effect collapses.
vv = np.linspace(5, 35, 400)
Re_vv = rho * vv * D / mu
for spv, c in [(0.10, c_yel), (0.20, c_cyan), (0.30, c_blue)]:
    ax[0, 1].plot(vv, C_L(Re_vv, spv), lw=2.4, color=c, label=f"Sp = {spv:.2f}")
ax[0, 1].axvspan(10, 17, color="gray", alpha=0.12)
ax[0, 1].text(13.5, 0.02, "transition\nregion", ha="center", fontsize=8.5, color="gray")
ax[0, 1].set_xlabel(r"Speed  $v$ [m/s]")
ax[0, 1].set_ylabel(r"$C_L$")
ax[0, 1].set_title("(b)  Lift vs speed: collapse below critical speed")
ax[0, 1].legend()
ax[0, 1].grid(True, alpha=0.4)

#(c) Full C_L(v, Sp) surface, the whole model in one map.
VV, SP = np.meshgrid(np.linspace(5, 35, 200), np.linspace(0.0, 0.45, 200))
CL_map = C_L(rho * VV * D / mu, SP)
cf = ax[1, 0].contourf(VV, SP, CL_map, levels=20, cmap="viridis")
fig.colorbar(cf, ax=ax[1, 0], label=r"$C_L$")
ax[1, 0].set_xlabel(r"Speed  $v$ [m/s]")
ax[1, 0].set_ylabel(r"Spin parameter  $Sp$")
ax[1, 0].set_title(r"(c)  Full surface  $C_L(v, Sp)$  for Telstar 18")

#(d) Independent check against Ward. Filled markers are Ward's measurements and
#    the open markers are our model evaluated at the same (Re, Sp). We only want
#    to see that the model lands in the right region, not a perfect match.
ax[1, 1].scatter(Sp_100, Cy_100, s=55, zorder=3, color=c_red, edgecolor="k",
                 label="Ward 100 rpm (check)")
ax[1, 1].scatter(Sp_300, Cy_300, s=55, zorder=3, marker="s", color=c_yel,
                 edgecolor="k", label="Ward 300 rpm (check)")
ax[1, 1].scatter(Sp_100, C_L(Re_100, Sp_100), s=35, facecolor="none",
                 edgecolor=c_blue, lw=1.6, label="model @ Ward points")
ax[1, 1].scatter(Sp_300, C_L(Re_300, Sp_300), s=35, marker="s", facecolor="none",
                 edgecolor=c_blue, lw=1.6)
ax[1, 1].set_xlabel(r"Spin parameter  $Sp$")
ax[1, 1].set_ylabel(r"$C_L$ / $C_y$")
ax[1, 1].set_title("(d)  Independent check vs Ward (not calibration)")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(True, alpha=0.4)

fig.suptitle("Telstar 18 Magnus law  —  Teamgeist spin shape + Telstar 18 drag transfer",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()