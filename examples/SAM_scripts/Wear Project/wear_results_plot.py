import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.io import loadmat

sys.path.append('../..')
sys.path.append('./data')
folder_name = './WearResults/'

study_ID = '2'

wear_data = np.load(folder_name + "wear_study_" + study_ID + ".npz", allow_pickle=True)
results_Epatch = wear_data['results_Epatch'][()]   # dictionary
results_Edot   = wear_data['results_Edot'][()]
slow_dts = wear_data['slow_dts']

# -------------------------
# Load data
# -------------------------
data = loadmat('./data/hysteresis_data.mat')
t_blocks_hours = data['t_blocks_hours'].flatten()
diss = data['DISS'].flatten()




# # -----------------------
# # Figure 1: Total Wear
# # -----------------------
# plt.figure()

# for dti, dt in enumerate(slow_dts):
#     t = np.arange(0, wear_data['final_time'], dt)
#     patch_wear = results_Epatch[dt]
#     total_wear = patch_wear.sum(axis=1)
#     plt.plot(t, total_wear, label=f"dt = {dt}")

# plt.grid(True)
# plt.xlabel('Time (s)')
# plt.ylabel('Total Wear (J)')
# plt.title('Wear Evolution Over Time')
# plt.legend()
# plt.tight_layout()


# -----------------------
# Figure 2: Wear Rate (E_dot)
# -----------------------
plt.figure()

for dti, dt in enumerate(slow_dts):
    t = np.arange(0, wear_data['final_time'], dt)[:-1]
    patch_Edot = results_Edot[dt].T[:-1]
    total_Edot = patch_Edot.sum(axis=1)
    plt.plot(t, total_Edot, label=f"dt = {dt}")

plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Total Wear Rate (J/s)')
plt.title('Wear Rate Evolution Over Time')
plt.legend()
plt.tight_layout()

plt.show()

# ================================================================
# Additional Section: Compare Study 2 (Heun) vs Study 3 (RK4)
# ================================================================

folder_name = "./WearResults/"

# --- Load Study 2 (Heun) ---
study2 = np.load(folder_name + "wear_study_2.npz", allow_pickle=True)
Epatch_2 = study2["results_Epatch"][()]
Edot_2   = study2["results_Edot"][()]
slow2    = study2["slow_dts"]
T2       = study2["final_time"]

# --- Load Study 3 (RK4) ---
study3 = np.load(folder_name + "wear_study_3.npz", allow_pickle=True)
Epatch_3 = study3["results_Epatch"][()]
Edot_3   = study3["results_Edot"][()]
slow3    = study3["slow_dts"]
T3       = study3["final_time"]

# --- Mapping of intended comparisons ---
# Heun dt = 1800   <--> RK4 dt = 3600
# Heun dt = 900    <--> RK4 dt = 1800

pairs = [
    (1800, 3600),
    (900,  1800)
]

# ================================================================
# Plot 1: All six wear-rate curves together
# ================================================================
plt.figure()

# --- Heun curves ---
for dt in slow2:
    t = np.arange(0, T2, dt)[:-1]
    total_Edot = Edot_2[dt].T[:-1].sum(axis=1)
    plt.plot(t, total_Edot, label=f"Heun dt={dt}", linestyle='--')

# --- RK4 curves ---
for dt in slow3:
    t = np.arange(0, T3, dt)[:-1]
    total_Edot = Edot_3[dt].T[:-1].sum(axis=1)
    plt.plot(t, total_Edot, label=f"RK4 dt={dt}", linestyle='-')

plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Wear Rate (J/s)")
plt.title("Wear Rate Comparison: Heun vs RK4 (All dt)")
plt.legend()
plt.tight_layout()


# ================================================================
# Plot 2: Heun dt=1800 minus RK4 dt=3600
# ================================================================
dt_H, dt_R = pairs[0]

# Build aligned time vector (use the smaller dt)
dt_common = min(dt_H, dt_R)
Tf = min(T2, T3)
t_common = np.arange(0, Tf, dt_common)[:-1]

# Interpolate each wear-rate trace to the common time grid
def interp_Edot(Edot_dict, dt_original, Tfinal, t_common):
    t_orig = np.arange(0, Tfinal, dt_original)[:-1]
    total_Edot = Edot_dict[dt_original].T[:-1].sum(axis=1)
    return np.interp(t_common, t_orig, total_Edot)

Edot_H = interp_Edot(Edot_2, dt_H, T2, t_common)
Edot_R = interp_Edot(Edot_3, dt_R, T3, t_common)

plt.figure()
plt.plot(t_common, Edot_H - Edot_R)
plt.axhline(0, color='k', linewidth=0.8)
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Δ Wear Rate (J/s)")
plt.title(f"Difference: Heun dt={dt_H}  –  RK4 dt={dt_R}")
plt.tight_layout()


# ================================================================
# Plot 3: Heun dt=900 minus RK4 dt=1800
# ================================================================
dt_H, dt_R = pairs[1]

dt_common = min(dt_H, dt_R)
t_common = np.arange(0, Tf, dt_common)[:-1]

Edot_H = interp_Edot(Edot_2, dt_H, T2, t_common)
Edot_R = interp_Edot(Edot_3, dt_R, T3, t_common)

plt.figure()
plt.plot(t_common, Edot_H - Edot_R)
plt.axhline(0, color='k', linewidth=0.8)
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Δ Wear Rate (J/s)")
plt.title(f"Difference: Heun dt={dt_H}  –  RK4 dt={dt_R}")
plt.tight_layout()

plt.show()

# ================================================================
# Additional Raw Wear Rate Curve Plots for Matching Pairs
# ================================================================

def extract_raw_Edot(Edot_dict, dt, Tfinal):
    """Extract raw wear-rate curve and its raw time vector."""
    t = np.arange(0, Tfinal, dt)[:-1]
    total_Edot = Edot_dict[dt].T[:-1].sum(axis=1)
    return t, total_Edot


# ================================================================
# Plot A: Raw Wear Rate -- Heun dt=1800 vs RK4 dt=3600
# ================================================================
dt_H = 1800
dt_R = 3600

tH, EdotH = extract_raw_Edot(Edot_2, dt_H, T2)
tR, EdotR = extract_raw_Edot(Edot_3, dt_R, T3)

plt.figure()
plt.plot(tH, EdotH, 'b--', label=f"Heun dt={dt_H}")
plt.plot(tR, EdotR, 'r-',  label=f"RK4 dt={dt_R}")

plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Wear Rate (J/s)")
plt.title("Raw Wear Rate Comparison: Heun 1800 vs RK4 3600")
plt.legend()
plt.tight_layout()


# ================================================================
# Plot B: Raw Wear Rate -- Heun dt=900 vs RK4 dt=1800
# ================================================================
dt_H = 900
dt_R = 1800

tH, EdotH = extract_raw_Edot(Edot_2, dt_H, T2)
tR, EdotR = extract_raw_Edot(Edot_3, dt_R, T3)

plt.figure()
plt.plot(tH, EdotH, 'b--', label=f"Heun dt={dt_H}")
plt.plot(tR, EdotR, 'r-',  label=f"RK4 dt={dt_R}")

plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Wear Rate (J/s)")
plt.title("Raw Wear Rate Comparison: Heun 900 vs RK4 1800")
plt.legend()
plt.tight_layout()

plt.show()

# ================================================================
# Section: Compare RK4 Study 3 (4 hours, red) vs Study 4 (1 hour, blue)
# ================================================================

# --- Load Study 4 ---
study4 = np.load(folder_name + "wear_study_4.npz", allow_pickle=True)
Epatch_4 = study4["results_Epatch"][()]
Edot_4   = study4["results_Edot"][()]
slow4    = study4["slow_dts"]
T4       = study4["final_time"]

# dt values in common and unique
common_dts = sorted(list(set(slow3) & set(slow4)))
unique4_dts = sorted(list(set(slow4) - set(slow3)))

# Define line styles for each dt value
dt_styles = {
    7200: '-',
    3600: '--',
    1800: '-.',
    900:  ':'     # Study 4 only
}

# Define fixed colors
color_3 = 'red'     # Study 3
color_4 = 'blue'    # Study 4

plt.figure()

# ================================================================
# Common dt values: plot both studies with same line style
# ================================================================
for dt in common_dts:
    style = dt_styles.get(dt, '-')

    # Study 3
    t3 = np.arange(0, T3, dt)[:-1]
    Edot3 = Edot_3[dt].T[:-1].sum(axis=1)
    plt.plot(t3, Edot3, linestyle=style, color=color_3,
             label=f"Study 3 (4 hr), dt={dt}")

    # Study 4
    t4 = np.arange(0, T4, dt)[:-1]
    Edot4 = Edot_4[dt].T[:-1].sum(axis=1)
    plt.plot(t4, Edot4, linestyle=style, color=color_4,
             label=f"Study 4 (1 hr), dt={dt}")

# ================================================================
# Unique Study 4 dt values (dt = 900)
# ================================================================
for dt in unique4_dts:
    style = dt_styles.get(dt, ':')

    t4 = np.arange(0, T4, dt)[:-1]
    Edot4 = Edot_4[dt].T[:-1].sum(axis=1)

    plt.plot(t4, Edot4, linestyle=style, color=color_4,
             label=f"Study 4 only, dt={dt}")

# -------------------------------------------------

# ================================================================
# Experiment Dissipation
# ================================================================
plt.plot(t_blocks_hours*3600, diss, color = 'g', label = f'Experiment Value')

plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Wear Rate (J/s)")
plt.title("RK4 Wear Rate Comparison: 4-Hour vs 1-Hour Parameter Identification")

# Handle duplicate labels cleanly
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), fontsize=8, ncol=1)

plt.tight_layout()
plt.show()


