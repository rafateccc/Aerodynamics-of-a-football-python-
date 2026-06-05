import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
v_goff_0 = [
    4.347355106947853, 5.285778690577568, 6.381994189250112, 7.552284225817762,
    8.648938035365425, 9.898562340329608, 11.143365225667488, 12.325489655863347,
    13.563717878074439, 14.577092621349497, 15.806554626058208, 16.957558984120624,
    18.110316585683513, 19.338463657766866, 20.49341281370535, 21.722436507538948,
    22.807694234333518, 23.9648349446476, 25.12285227671192, 26.280869608776236,
    27.438448629965436, 28.595589340279513, 29.75360667234383, 30.984383609677902,
    32.2134073035115, 33.372739568201176, 34.674084556429406, 35.83122526674348,
    36.91692130441317
]

Cd_goff_0 = [
    0.5740019035215149, 0.5808846365776688, 0.5331563392275711, 0.4781245303812053,
    0.42857786905775686, 0.3444221810349146, 0.28026849671893006, 0.17614086059209533,
    0.13926263587637122, 0.13520512948955565, 0.13469418424084556, 0.15967039022191054,
    0.17737314030957274, 0.18231728698091465, 0.1909282172018234, 0.19223563592646398,
    0.1899664379101338, 0.18948554826428898, 0.18536793067174273, 0.1812503130791966,
    0.178951059460001, 0.1784701698141562, 0.17435255222161006, 0.16838651505284774,
    0.16969393377748843, 0.16012122426489006, 0.16139858738666535, 0.16091769774082054,
    0.1568301357511397
]

v_goff_45 = [
    4.492753623188406, 5.434782608695652, 6.594202898550725,
    7.753623188405797, 8.840579710144928, 10.144927536231885,
    11.376811594202898, 12.391304347826086, 14.63768115942029,
    13.55072463768116, 15.797101449275363, 17.028985507246375,
    18.26086956521739, 19.42028985507246, 20.507246376811594,
    21.666666666666668, 22.826086956521742, 24.130434782608695,
    25.217391304347828, 26.376811594202895, 27.53623188405797,
    28.913043478260867, 30.144927536231883, 31.15942028985507,
    32.2463768115942, 33.333333333333336, 34.42028985507246,
    35.869565217391305, 37.028985507246375
]

Cd_goff_45 = [
    0.5636186222814359, 0.5852733106903605, 0.5507121277514028,
    0.5270270777429585, 0.4743465125443321, 0.3817642753935937,
    0.2837514777354526, 0.18576119795087365, 0.12933515978307786,
    0.1276350603291363, 0.1419038862096791, 0.1562777955001784,
    0.16702632714717314, 0.18503312003903105, 0.19398397477998164,
    0.19023850181081248, 0.19918185059390903, 0.19904674335253625,
    0.1953087763412209, 0.19337599219380386, 0.19325589686813915,
    0.19130059484716, 0.1875476159201367, 0.18562984368842772,
    0.18370456549886482, 0.1799665984875496, 0.17985400911973892,
    0.18332926760616253, 0.18320917228049782
]

#Transform lists into arrays
v_goff_0 = np.array(v_goff_0, dtype=float)
Cd_goff_0 = np.array(Cd_goff_0, dtype=float)

v_goff_45 = np.array(v_goff_45, dtype=float)
Cd_goff_45 = np.array(Cd_goff_45, dtype=float)

#Sort arrays
idx0 = np.argsort(v_goff_0)
v_goff_0 = v_goff_0[idx0]
Cd_goff_0 = Cd_goff_0[idx0]

idx45 = np.argsort(v_goff_45)
v_goff_45 = v_goff_45[idx45]
Cd_goff_45 = Cd_goff_45[idx45]

#Define a common min and common max and generate equally spaced velocity values between these two
v_min_common = max(v_goff_0.min(), v_goff_45.min())
v_max_common = min(v_goff_0.max(), v_goff_45.max())
v_common = np.linspace(v_min_common, v_max_common, 29)

#Interpolate smoothly between data points in order to avoid overshoot
f0 = PchipInterpolator(v_goff_0, Cd_goff_0)
f45 = PchipInterpolator(v_goff_45, Cd_goff_45)

#Evaluate both functions in the same set of values 
Cd_0_common = f0(v_common)
Cd_45_common = f45(v_common)
#Mean value between 0º and 45º seam orientatin
Cd_mean = (Cd_0_common + Cd_45_common) / 2
v_data = v_common
Cd_data = Cd_mean


#The vmin parameter has low identifiability in the fit, since large changes in this parameter result in only small changes in
#the residuals. Therefore, it is set based on the observed experimental minimum, thereby reducing the number of free parameters
#and stabilizing the fit.
v_min_fixed = v_data[np.argmin(Cd_data)]   
print(v_min_fixed)
def Cd_model(v, a, b_min, v_c, v_s,p):
    term1 = (a - b_min) / (1 + np.exp((v - v_c) / v_s))
    term2 = b_min
    term3 = ((v - v_min_fixed) / (1 + np.exp(-(v - v_min_fixed) / v_s))) * p
    return term1 + term2 + term3


# Initial parameter guess: [a, b_min, v_c, v_s,p]
p0 = [0.57,0.16,12,1.5,0.001]

# Bounds keep parameters physically meaningful and avoid divide-by-zero
bounds = (
    [0.3,  0.05, 5.0,  0.2, -0.01],  # lower
    [0.8,  0.30, 20.0, 5.0, 0.01],  # upper
)

optimal_parameters, covariance_matrix = curve_fit(Cd_model, v_data, Cd_data, p0=p0, bounds=bounds, maxfev=10_000)
# covariance_matrix is a 5x5 matrix(we have 5 parameters) and its diagonal is the variance of each parameter 
# so its sqrt equals the standard deviation sigma

perr = np.sqrt(np.diag(covariance_matrix))  

parameter_names = ["a", "b_min", "v_c", "v_s", "p"]
print("Fitted parameters:")
for name, val, err in zip(parameter_names, optimal_parameters, perr):
    print(f"  {name:6s} = {val:.4f} ± {err:.4f}")

# Residuals & R²
Cd_pred = Cd_model(v_data, *optimal_parameters)
ss_res = np.sum((Cd_data - Cd_pred) ** 2)
ss_tot = np.sum((Cd_data - Cd_data.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"\nR² = {r_squared:.5f}")

# Plot 
v_plot = np.linspace(v_data.min(), v_data.max(), 500)
Cd_fit = Cd_model(v_plot, *optimal_parameters)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(v_data, Cd_data, s=40, zorder=3, label="Data (Goff)")
ax.plot(v_plot, Cd_fit, lw=2, label=f"Fitted model  ($R^2={r_squared:.4f}$)")
ax.set_xlabel("Free-stream velocity (m/s)")
ax.set_ylabel("$C_d$")
ax.set_title("Drag coefficient vs free-stream velocity")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()