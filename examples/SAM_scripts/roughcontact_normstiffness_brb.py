import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.stats import gmean


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tmdsimpy.continuation import Continuation
from tmdsimpy.jax.nlforces.roughcontact import _asperity_functions
from tmdsimpy.nlforces.normstiffness import NormStiffness
from tmdsimpy.nlforces.unilateral_spring import UnilateralSpring
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.vibration_system import VibrationSystem
import tmdsimpy.utils.harmonic as hutils


def resolve_backbone_file():
    candidate = SCRIPT_DIR.parent / "structures" / "results" / "Uwxa_full.npy"

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Unable to find the rough-contact BRB backbone at:\n"
        f"{candidate}\n"
        "Run the rough-contact `brb_epmc.py` workflow first so this script "
        "uses a backbone from a rough-contact source."
    )


def extract_mode_shape(Uwxa_row, Ndof, M):
    h1c = Uwxa_row[Ndof:2 * Ndof]
    h1s = Uwxa_row[2 * Ndof:3 * Ndof]

    phi_complex = h1c + 1j * h1s
    phi = np.real(phi_complex * np.exp(-1j * np.angle(phi_complex[0])))
    phi /= np.sqrt(phi.T @ M @ phi)

    return phi


def kt_eff_model_A(A, kt_pred, bA, sA):
    return kt_pred * np.power(np.power(A / sA, bA) + 1.0, -1.0 - 1.0 / bA)


###############################################################################
####### User Inputs                                                     #######
###############################################################################

system_fname = SCRIPT_DIR / "data" / "BRB_ROM_U_122ELS4py.mat"
surface_fname = SCRIPT_DIR / "data" / "brb_surface_data.mat"
backbone_fname = resolve_backbone_file()

fit_output_fname = SCRIPT_DIR / "data" / "roughcontact_normstiffness_fit.npz"
norm_output_fname = SCRIPT_DIR / "data" / "Uwxa_full_roughcontact_normstiffness.npy"

mesoscale_TF = True
show_plots = True
skip_continuation = False

prestress = (12002.0 + 12075.0 + 12670.0) / 3.0
damp_ab = [0.087e-2 * 2.0 * (168.622 * 2.0 * np.pi), 0.0]

# Continuation settings chosen to match iwan_normstiffness_brb.py.
Nt = 1 << 7
ds = 0.002
dsmax = 0.005
dsmin = 0.0005
FracLam = 1


###############################################################################
####### 1. Traditional Analysis Replacement                             #######
###############################################################################
# This replaces the "traditional analysis" section from iwan_normstiffness_brb.
# Here, the rough-contact backbone is imported directly instead of being run.

print(f"Using system file: {system_fname}")
print(f"Using surface file: {surface_fname}")
print(f"Using rough-contact backbone: {backbone_fname}")
print(f"Mesoscale topology enabled: {mesoscale_TF}")

system_matrices = sio.loadmat(system_fname)
surface_pars = sio.loadmat(surface_fname)
Uwxa_full = np.load(backbone_fname)

M = np.asarray(system_matrices["M"])
K = np.asarray(system_matrices["K"])
Ndof = M.shape[0]

Nhc = (Uwxa_full.shape[1] - 3) // Ndof
h_max = (Nhc - 1) // 2
h = np.array(range(h_max + 1))
Nhc = hutils.Nhc(h)

rough_freqs = Uwxa_full[:, -3] / (2.0 * np.pi)
rough_xis = Uwxa_full[:, -2]
rough_amps = Uwxa_full[:, -1]
Astart = rough_amps[0]
Aend = rough_amps[-1]

Qm = np.asarray(system_matrices["Qm"].todense())
Tm = np.asarray(system_matrices["Tm"].todense())
L = np.asarray(system_matrices["L"])
Fv = system_matrices["Fv"][:, 0]

Npatches, Nnodes = Qm.shape
L3 = L[:3 * Nnodes, :]

# Rough-contact BRB operators. These play the same role that Qxyn/Txyn do in
# iwan_normstiffness_brb.py, but they already include the reduced interface map.
Q = np.kron(Qm, np.eye(3)) @ L3
T = L3.T @ np.kron(Tm, np.eye(3))

xy_rows = np.column_stack(
    (np.arange(0, Q.shape[0], 3), np.arange(1, Q.shape[0], 3))
).reshape(-1)
n_rows = np.arange(2, Q.shape[0], 3)

Qxy = Q[xy_rows, :]
Qn = Q[n_rows, :]
Txy = T[:, xy_rows]
Tn = T[:, n_rows]

patch_areas = np.sum(Tm, axis=0)
Axy = np.diag(np.repeat(patch_areas, 2))
An = np.diag(patch_areas)

ElasticMod = 192.31e9
PoissonRatio = 0.3
Radius = float(surface_pars["Re"][0, 0])
TangentMod = 620e6
YieldStress = 331.7e6
area_density = float(surface_pars["area_density"][0, 0])
max_gap = float(surface_pars["z_max"][0, 0])

normzinterp = np.asarray(surface_pars["normzinterp"][0]).squeeze()
pzinterp = np.asarray(surface_pars["pzinterp"][0]).squeeze()
mesoscale_xygap = np.asarray(surface_pars["mesoscale_xygap"])

gaps = np.linspace(0.0, 1.0, 101) * max_gap
trap_weights = np.ones_like(gaps)
trap_weights[1:-1] = 2.0
trap_weights = trap_weights / trap_weights.sum()
gap_weights = area_density * trap_weights * np.interp(gaps / max_gap, normzinterp, pzinterp)

interp_obj = LinearNDInterpolator(mesoscale_xygap[:, :2], mesoscale_xygap[:, 2])
meso_gap_nodes = interp_obj(system_matrices["node_coords"][:, 0], system_matrices["node_coords"][:, 1])
meso_gap_quads = Qm @ meso_gap_nodes
meso_gap_quads = meso_gap_quads - meso_gap_quads.min()
if not mesoscale_TF:
    meso_gap_quads[:] = 0.0

Xpre_rough = Uwxa_full[0, :Ndof].copy()


###############################################################################
####### 2. Greenwood-Williamson Derivation of a Good kn                #######
###############################################################################

Estar = ElasticMod / 2.0 / (1.0 - PoissonRatio**2)
ShearMod = ElasticMod / 2.0 / (1.0 + PoissonRatio)
Gstar = ShearMod / 2.0 / (2.0 - PoissonRatio)

C = 1.295 * np.exp(0.736 * PoissonRatio)
delta_y1s = (np.pi * C * YieldStress / (2.0 * (2.0 * Estar))) ** 2 * (2.0 * Radius)
delta_y = 2.0 * delta_y1s

KN_patches_gw = np.zeros(Npatches)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in power",
    )

    for patch in range(Npatches):
        uxyn_patch = Q[3 * patch:3 * patch + 3, :] @ Xpre_rough

        delta = uxyn_patch[2] - meso_gap_quads[patch] - gaps
        deltam = -meso_gap_quads[patch] - gaps
        Fm = np.zeros_like(delta)

        _, a, _, _ = _asperity_functions._normal_asperity_general(
            delta,
            deltam,
            Fm,
            Radius,
            PoissonRatio,
            Estar,
            ElasticMod,
            TangentMod,
            delta_y,
            YieldStress,
        )

        a = np.asarray(a)
        KN_patches_gw[patch] = float((2.0 * Estar * a) @ gap_weights)
print(
    "GW patch stiffness summary: "
    f"KN mean={KN_patches_gw.mean():.4e}"
)

gamma_patches = KN_patches_gw / KN_patches_gw.mean()
rough_overlaps = np.maximum((Qn @ Xpre_rough) - meso_gap_quads, 0.0)
normal_force_shape = Tn @ (gamma_patches * rough_overlaps)
rhs_normal = prestress * Fv - K @ Xpre_rough

if np.linalg.norm(normal_force_shape) <= np.finfo(float).eps:
    raise RuntimeError("Unable to identify a system-level kn because the normal shape vector is zero.")

kn_pred = (normal_force_shape.T @ rhs_normal) / (normal_force_shape.T @ normal_force_shape)
if kn_pred <= 0.0:
    raise RuntimeError(f"Identified system-level kn is non-positive: {kn_pred:.6e}")

KN_patches = kn_pred * gamma_patches

print(
    "Identified normal stiffness parameters: "
    f"kn={kn_pred:.6e}, gamma range=[{gamma_patches.min():.4e}, {gamma_patches.max():.4e}]"
)

normal_sys = VibrationSystem(M, K, ab=damp_ab)
for patch in range(Npatches):
    Ls = Qn[patch:patch + 1, :]
    Lf = Tn[:, patch:patch + 1]
    normal_sys.add_nl_force(
        UnilateralSpring(Ls, Lf, KN_patches[patch], delta=meso_gap_quads[patch])
    )

normal_solver = NonlinearSolver()
normal_pre_fun = lambda U, calc_grad=True: normal_sys.static_res(U, Fv * prestress)

Knl_n0 = Tn @ np.diag(KN_patches) @ Qn
X0_normal_linear = np.linalg.solve(K + Knl_n0, Fv * prestress).squeeze()
R0_normal_linear, _ = normal_pre_fun(X0_normal_linear)
R0_normal_rough, _ = normal_pre_fun(Xpre_rough)

if np.linalg.norm(R0_normal_linear) <= np.linalg.norm(R0_normal_rough):
    X0_normal = X0_normal_linear
    print("Using linearized normal prestress guess.")
else:
    X0_normal = Xpre_rough.copy()
    print("Using rough-contact prestress state as the normal prestress guess.")

t0 = time.time()
Xpre_normal, R_normal, _, sol_normal = normal_solver.nsolve(
    normal_pre_fun,
    X0_normal,
    verbose=True,
    xtol=1e-13,
)
t1 = time.time()

print("Normal-only residual norm: {:.4e}".format(np.linalg.norm(R_normal)))
print("Normal-only static solution run time: {:.3e} s".format(t1 - t0))

if isinstance(sol_normal, dict) and (not sol_normal.get("success", True)):
    warnings.warn(
        "Normal-only prestress solve did not converge cleanly. "
        f"Residual norm: {np.linalg.norm(R_normal):.4e}. Continuing anyway."
    )

_, dRpredX_normal = normal_sys.static_res(Xpre_normal, Fv * prestress)
Kn = 0.5 * (dRpredX_normal + dRpredX_normal.T)

normal_eigvals, normal_eigvecs = normal_solver.eigs(Kn, M, subset_by_index=[0, 9])
normal_mode_ind = int(np.argmin(np.abs(np.sqrt(np.real(normal_eigvals)) - rough_freqs[0] * 2.0 * np.pi)))
wn = np.sqrt(np.real(normal_eigvals[normal_mode_ind]))

print(f"Normal-only prestressed natural frequency wn [Hz]: {wn / (2.0 * np.pi):.6f}")


###############################################################################
####### 3. Derive kt, b_A, s_A, b_i, s_i                               #######
###############################################################################

row = Uwxa_full[0, :]
omega_star = row[-3]
phi_star = extract_mode_shape(row, Ndof, M)

kt_pred = (
    omega_star**2
    - phi_star.T @ Kn @ phi_star
) / (
    (Qxy @ phi_star).T @ Axy @ (Qxy @ phi_star)
)

print("Identified low-amplitude tangential stiffness kt =", kt_pred)

if kt_pred <= 0.0:
    raise RuntimeError("Identified tangential stiffness is non-positive.")

kt_pred_array = np.zeros(Uwxa_full.shape[0])

for level in range(Uwxa_full.shape[0]):
    phi = extract_mode_shape(Uwxa_full[level, :], Ndof, M)

    kt_pred_array[level] = (
        Uwxa_full[level, -3]**2
        - phi.T @ Kn @ phi
    ) / (
        (Qxy @ phi).T @ Axy @ (Qxy @ phi)
    )

plt.figure()
plt.plot(Uwxa_full[:, -1], kt_pred_array)
plt.xlabel("Log Modal Amplitude")
plt.ylabel("Tangential Stiffness")
plt.title("Identified Tangential Stiffness")
plt.grid(True)
if show_plots:
    plt.show()
else:
    plt.close()

lA_vals = Uwxa_full[:, -1]
A_vals = 10 ** lA_vals

hardening_tol = 5e-3
fit_stop = len(A_vals)
for i in range(1, len(A_vals)):
    if (kt_pred_array[i] <= 0.0) or (kt_pred_array[i] > kt_pred_array[i - 1] * (1.0 + hardening_tol)):
        fit_stop = i
        break

fit_mask = ((np.arange(len(A_vals)) < fit_stop) & (kt_pred_array > 0.0))
if np.count_nonzero(fit_mask) < 2:
    raise RuntimeError("Insufficient positive stiffness points for the NormStiffness fit.")

rejected_start_lA = lA_vals[fit_stop] if fit_stop < len(lA_vals) else None

p0 = [2.7, gmean(A_vals[fit_mask])]
bounds = ([1e-3, A_vals.min() / 10.0], [100.0, A_vals.max() * 10.0])

popt, _ = curve_fit(
    lambda A, bA, sA: kt_eff_model_A(A, kt_pred, bA, sA),
    A_vals[fit_mask],
    kt_pred_array[fit_mask],
    p0=p0,
    bounds=bounds,
    maxfev=20000,
)

bA, sA = popt
kt_fit_plot = kt_eff_model_A(A_vals, kt_pred, bA, sA)

print(f"Fitted parameters: kt={kt_pred:.6e}, bA={bA:.6f}, sA={sA:.6e}")

plt.figure()
if rejected_start_lA is not None:
    plt.axvspan(
        rejected_start_lA,
        lA_vals[-1],
        color="red",
        alpha=0.15,
        label="Rejected backbone curve",
    )
plt.plot(lA_vals, kt_pred_array, "k.", alpha=0.4, label="Identified $k_t$")
plt.plot(lA_vals, kt_fit_plot, "r-", lw=2, label="NormStiffness fit")
plt.title("Predicted Tangential Stiffness Scale vs. Log Amplitude")
plt.xlabel(r"Log Amplitude, $\log_{10}(A)$")
plt.ylabel(r"Predicted Tangential Stiffness, $k_t$")
plt.legend()
plt.grid(True)
if show_plots:
    plt.show()
else:
    plt.close()

KT_patches = kt_pred * patch_areas

A_levels = np.power(10.0, lA_vals)
ui_levels_from_rough = np.zeros((Qxy.shape[0], Uwxa_full.shape[0]))

h0 = int(h[0] == 0)
for level in range(Uwxa_full.shape[0]):
    Uwxa = Uwxa_full[level, :]
    Uhc = Uwxa[:Nhc * Ndof].reshape(Nhc, Ndof).copy()

    A = 10.0 ** Uwxa[-1]
    Uhc[h0:, :] *= A

    ut = hutils.time_series_deriv(Nt, h, Uhc, order=0)
    unlt = ut @ Qxy.T

    ui_levels_from_rough[:, level] = 0.5 * (
        np.max(unlt, axis=0) - np.min(unlt, axis=0)
    )

    if level == 0:
        print(
            "Initial tangential local amplitudes: {}".format(
                0.5 * (np.max(unlt, axis=0) - np.min(unlt, axis=0))
            )
        )

A_sort_idx = np.argsort(A_levels)
A_sorted = A_levels[A_sort_idx]
ui_sorted = ui_levels_from_rough[:, A_sort_idx]
A_eval = np.clip(sA, A_sorted[0], A_sorted[-1])

s_i = np.array([
    np.interp(A_eval, A_sorted, ui_sorted[row, :])
    for row in range(ui_sorted.shape[0])
])
s_floor = max(1e-12, 1e-6 * np.max(s_i))
s_i = np.maximum(s_i, s_floor)
b_i = bA * np.ones(Qxy.shape[0])

fit_output_fname.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    fit_output_fname,
    KN_patches_gw=KN_patches_gw,
    gamma_patches=gamma_patches,
    kn_pred=kn_pred,
    KN_patches=KN_patches,
    patch_areas=patch_areas,
    KT_patches=KT_patches,
    kt_pred_array=kt_pred_array,
    kt_fit_plot=kt_fit_plot,
    fit_mask=fit_mask,
    kt_pred=kt_pred,
    bA=bA,
    sA=sA,
    b_i=b_i,
    s_i=s_i,
    meso_gap_quads=meso_gap_quads,
)


###############################################################################
####### 4. Create the NormStiffness / UnilateralSpring System          #######
###############################################################################

vib_sys = VibrationSystem(M, K, ab=damp_ab)

for patch in range(Npatches):
    x_row = 3 * patch
    y_row = 3 * patch + 1
    n_row = 3 * patch + 2

    vib_sys.add_nl_force(
        NormStiffness(
            Q[x_row:x_row + 1, :],
            T[:, x_row:x_row + 1],
            KT_patches[patch],
            b_i[2 * patch],
            s_i[2 * patch],
        )
    )
    vib_sys.add_nl_force(
        NormStiffness(
            Q[y_row:y_row + 1, :],
            T[:, y_row:y_row + 1],
            KT_patches[patch],
            b_i[2 * patch + 1],
            s_i[2 * patch + 1],
        )
    )
    vib_sys.add_nl_force(
        UnilateralSpring(
            Q[n_row:n_row + 1, :],
            T[:, n_row:n_row + 1],
            KN_patches[patch],
            delta=meso_gap_quads[patch],
        )
    )

norm_forces = vib_sys.nonlinear_forces


###############################################################################
####### 5. Run the NormStiffness System                                #######
###############################################################################

if not skip_continuation:
    ############################################################################
    ####### 5.1 Prestress Analysis of the Full (K, kn, kt) System         #######
    ############################################################################
    pre_fun = lambda U, calc_grad=True: vib_sys.static_res(U, Fv * prestress)

    Kstxy = Txy @ np.diag(np.repeat(KT_patches, 2)) @ Qxy
    Kstn = Tn @ np.diag(KN_patches) @ Qn
    K0_full = K + Kstxy + Kstn
    X0_full_linear = np.linalg.solve(K0_full, Fv * prestress).squeeze()

    R0_full_linear, _ = pre_fun(X0_full_linear)
    R0_full_normal, _ = pre_fun(Xpre_normal)

    print(
        "Full-system residual norm of linearized prestress guess: {:.4e}".format(
            np.linalg.norm(R0_full_linear)
        )
    )
    print(
        "Full-system residual norm at normal-only prestress state: {:.4e}".format(
            np.linalg.norm(R0_full_normal)
        )
    )

    if np.linalg.norm(R0_full_linear) <= np.linalg.norm(R0_full_normal):
        X0_full = X0_full_linear
        print("Using linearized full-system prestress guess.")
    else:
        X0_full = Xpre_normal.copy()
        print("Using normal-only prestress state as the full-system prestress guess.")

    static_solver = NonlinearSolver()

    t0 = time.time()
    Xpre, R_static, _, sol_static = static_solver.nsolve(
        pre_fun,
        X0_full,
        verbose=True,
        xtol=1e-13,
    )
    t1 = time.time()

    print("Full-system prestress residual norm: {:.4e}".format(np.linalg.norm(R_static)))
    print("Full-system static solution run time: {:.3e} s".format(t1 - t0))

    if isinstance(sol_static, dict) and (not sol_static.get("success", True)):
        warnings.warn(
            "Full NormStiffness prestress solve did not converge cleanly. "
            f"Residual norm: {np.linalg.norm(R_static):.4e}. Continuing anyway."
        )

    vib_sys.update_force_history(Xpre)
    vib_sys.reset_real_mu()

    ############################################################################
    ####### 5.2 Updated Eigenvalue Analysis After Prestress              #######
    ############################################################################

    Rpre, dRpredX = vib_sys.static_res(Xpre, Fv * prestress)

    sym_check = np.max(np.abs(dRpredX - dRpredX.T))
    print(
        "Symmetric matrix check (max asymmetry / max entry): {}".format(
            sym_check / np.abs(dRpredX).max()
        )
    )

    Kpre = 0.5 * (dRpredX + dRpredX.T)
    eigvals, eigvecs = static_solver.eigs(Kpre, M, subset_by_index=[0, 9])
    eigvecs /= np.sqrt(np.diag(eigvecs.T @ M @ eigvecs))

    print(f"Eigenvalues of NormStiffness prestress: {eigvals}")

    ############################################################################
    ####### 5.3 Updated Damping Matrix After Prestress                   #######
    ############################################################################

    desired_zeta = np.array([0.087e-2, 0.034e-2])
    omega_12 = np.array([np.sqrt(eigvals)[0:3:2]]).reshape(2, 1)
    prop_mat = np.hstack((1.0 / (2.0 * omega_12), omega_12 / 2.0))
    pre_ab = np.linalg.solve(prop_mat, desired_zeta)
    vib_sys.set_new_C(C=pre_ab[0] * vib_sys.M + pre_ab[1] * Kpre)

    ############################################################################
    ####### 5.4 EPMC Initial Guess                                       #######
    ############################################################################

    Fl = np.zeros(Nhc * Ndof)
    Fl[:Ndof] = prestress * Fv
    Fl[Ndof:2 * Ndof] = system_matrices["R"][2, :]

    Uwxa0 = np.zeros(Nhc * Ndof + 3)
    Uwxa0[:Ndof] = Xpre

    omega_target = rough_freqs[0] * 2.0 * np.pi
    mode_ind = int(np.argmin(np.abs(np.sqrt(np.real(eigvals)) - omega_target)))
    Uwxa0[2 * Ndof:3 * Ndof] = np.real(eigvecs[:, mode_ind])
    Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))
    Uwxa0[-2] = 2.0 * Uwxa0[-3] * desired_zeta[0]
    Uwxa0[-1] = Astart

    ref_freq_hz = rough_freqs[0]
    uwxa0_freq_hz = Uwxa0[-3] / (2.0 * np.pi)
    freq_rel_error = np.abs(uwxa0_freq_hz - ref_freq_hz) / np.abs(ref_freq_hz)

    print(f"Reference natural frequency [Hz]: {ref_freq_hz:.6f}")
    print(f"Constructed Uwxa0 natural frequency [Hz]: {uwxa0_freq_hz:.6f}")
    print(f"Relative frequency difference: {100.0 * freq_rel_error:.3f}%")

    if freq_rel_error > 0.01:
        raise RuntimeError(
            "Constructed Uwxa0 natural frequency differs from the rough-contact "
            f"reference by more than 1% ({100.0 * freq_rel_error:.3f}%). "
            "Canceling before continuation."
        )

    ############################################################################
    ####### 5.5 EPMC Continuation                                        #######
    ############################################################################

    epmc_fun = lambda Uwxa, calc_grad=True: vib_sys.epmc_res(
        Uwxa,
        Fl,
        h,
        Nt=Nt,
        calc_grad=calc_grad,
    )

    epmc_solver = NonlinearSolver()

    continue_config = {
        "DynamicCtoP": True,
        "TargetNfev": 4,
        "MaxSteps": 2500,
        "dsmin": dsmin,
        "dsmax": dsmax,
        "verbose": 1,
        "xtol": 1e-5 * np.sqrt(Uwxa0.shape[0]),
        "corrector": "Ortho",
        "nsolve_verbose": True,
        "FracLam": FracLam,
        "FracLamList": [0.9, 0.1, 1.0, 0.0],
        "backtrackStop": 0.05,
    }

    CtoPstatic = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-5)
    CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3)
    CtoP[:Ndof] = CtoPstatic[:Ndof]
    CtoP[-3:-1] = np.abs(Uwxa0[-3:-1])
    CtoP[-1] = np.abs(Aend - Astart)

    R0, _, _ = epmc_fun(Uwxa0)
    print("Norm of initial NormStiffness EPMC residual: {}".format(np.linalg.norm(R0)))

    cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP, config=continue_config)

    t0 = time.time()
    Uwxa_full_norm = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)
    t1 = time.time()

    print("NormStiffness continuation solve time: {: 8.3f} seconds".format(t1 - t0))

    norm_output_fname.parent.mkdir(parents=True, exist_ok=True)
    np.save(norm_output_fname, Uwxa_full_norm)

    norm_freqs = Uwxa_full_norm[:, -3] / (2.0 * np.pi)
    norm_xis = Uwxa_full_norm[:, -2]
    norm_amps = Uwxa_full_norm[:, -1]

    plt.figure()
    plt.plot(norm_amps, norm_freqs, label="NormStiffness")
    plt.plot(rough_amps, rough_freqs, label="Rough Contact")
    plt.legend()
    plt.xlabel("Logarithm of Modal Amplitude")
    plt.ylabel("Natural Frequency (Hz)")
    plt.title("Natural Frequency with Respect to Amplitude")
    plt.grid(True)

    plt.figure()
    plt.plot(norm_amps, norm_xis, label="NormStiffness")
    plt.plot(rough_amps, rough_xis, label="Rough Contact")
    plt.legend()
    plt.xlabel("Log Modal Amplitude")
    plt.ylabel("Self-Excitation Factor (1/s)")
    plt.title("Self Excitation Factor with Respect to Amplitude")
    plt.grid(True)

    if show_plots:
        plt.show()
    else:
        plt.close("all")
