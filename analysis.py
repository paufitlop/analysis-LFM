import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd


def gauss(x, sigma, mu, A,B,C):
    """ Gaussian fit with linear continuous component """
    y = A*np.exp(-0.5*(x-mu)**2/sigma**2) + B*x + C
    return y

def gauss_parab(x, sigma, mu, A,B,C,D):
    """ Gaussian fit with parabolic continuous component """
    y = A*np.exp(-0.5*(x-mu)**2/sigma**2) + B*x**2 + C*x + D
    return y

def gauss_solo(x, sigma, mu, A):
    """ Gaussian fit"""
    y = A*np.exp(-0.5*(x-mu)**2/sigma**2)
    return y


def smooth(y, box_pts):
    """ Smoothing of the net spectrum (optional) """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def recta(x, m, b):
    return m * x + b


def cargar_canales_separados(fitxer, asicid):
    #Reads the file
    df = pd.read_csv(fitxer, sep='\t')

    #Filtrates by ASIC (just 1 in this case)
    df_asic = df[df['ASIC'] == asicid]

    #Creates
    lista_de_vectores = []

    for i in range(16):
        nombre_columna = f'CH[{i}]'
        # Each column (channel) is extracted independently
        vector_canal = df_asic[nombre_columna].to_numpy()

        # Each form an element of a bigger column
        lista_de_vectores.append(vector_canal)

    return lista_de_vectores




def plot_acq_pedestal(dades, label, fit_bool, xl_1, xr_1, num_bins, peak_prom=50, ruta_txt='HG_espectres_pedestal.txt'):
    """ Plots the 16 partial spectra (used especially for the HG and LG pedestals) and plots the summed spectrum with 
    pedestal subtraction"""

    # Configuration of the layout
    n_files = len(dades)
    print(f"Processant {n_files} canals...")

    ncols = 4
    nrows = int(np.ceil(n_files / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharey=True)

    if n_files == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    pos_pedestal = []
    sigma_pedestal = []
    area_pedestal = []

    # Plot loop
    for idx, d in enumerate(dades):
        bins = int(0.5 * (np.max(d) - np.min(d)))
        bins = max(bins, 10)  # Evita col·lapse si hi ha valors plans

        counts, bin_edges, patches = axs[idx].hist(d, bins=bins, density=False, histtype='step',
                                                     log=True, color='navy', label=label[idx])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        mask = (bin_centers > xl_1) & (bin_centers < xr_1)
        y_peak1 = counts[mask]
        x_peak1 = bin_centers[mask]

        centro_final = None
        sigma = np.nan
        area = np.nan

        if fit_bool:
            # Finds the peak
            peak, _ = find_peaks(y_peak1, prominence=peak_prom)

        
            height_1 = np.array([y_peak1[i] for i in peak])
            index_max_1 = np.argmax(height_1)
            x_pic_1 = x_peak1[peak[index_max_1]].item()
            axs[idx].scatter(x_peak1[peak[index_max_1]], y_peak1[peak[index_max_1]],
                                color='red', zorder=4)

            centro_final = x_pic_1  # defect value if the fit fails

            if x_pic_1 is not None:
                mask_fit = (bin_centers > x_pic_1 - 35) & (bin_centers < x_pic_1 + 35)
                y_fit = counts[mask_fit]
                x_fit = bin_centers[mask_fit]

                try:
                    fit_1, cov = curve_fit(gauss_solo, x_fit, y_fit,
                                            p0=[20, x_pic_1, np.max(y_fit)])
                    y_gauss_1 = gauss_solo(x_fit, fit_1[0], fit_1[1], fit_1[2])
                    axs[idx].plot(x_fit, y_gauss_1, label='fit_1', linewidth=3, color='purple')

                    centro_final = fit_1[1]
                    sigma = np.abs(fit_1[0])
                    area = np.sum(y_peak1)
                except Exception as e:
                    print(f"[AVÍS] Fit fallit al canal idx={idx} (label={label[idx]}): {e}")

            axs[idx].axvline(x=xl_1, linestyle='--', color='red', alpha=0.5)
            axs[idx].axvline(x=xr_1, linestyle='--', color='red', alpha=0.5)

        pos_pedestal.append(centro_final)
        sigma_pedestal.append(sigma)
        area_pedestal.append(area)

        # Subplots 
        axs[idx].set_xlabel('ADC counts')
        axs[idx].set_ylabel('Events')
        axs[idx].set_xlim(.99 * np.min(d), 2 * np.mean(d))
        axs[idx].set_ylim(bottom=1)
        axs[idx].legend(loc='upper right', fontsize='small')
        axs[idx].grid(True, alpha=0.3)

    # Erase empty squares
    for k in range(idx + 1, len(axs)):
        axs[k].set_visible(False)

    plt.tight_layout()
    plt.show()

    if fit_bool:
        # Combined histogram

        rang = (-100, 300)
        bin_edges = np.linspace(rang[0], rang[1], num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        columnes = [bin_centers]
        noms = ['ADC']
        for d, p, lbl in zip(dades, pos_pedestal, label):
            if p is None or np.isnan(p):
                continue
            counts, _ = np.histogram(d - p, bins=bin_edges)
            columnes.append(counts)
            noms.append(str(lbl).replace(' ', '_'))

        np.savetxt(ruta_txt, np.column_stack(columnes), delimiter='\t',
                   header='\t'.join(noms), comments='',
                   fmt=['%.2f'] + ['%d'] * (len(columnes) - 1))
        print(f"Espectres guardats a: {ruta_txt}")

        fig, ax = plt.subplots(figsize=(10, 7))

        for d, p, lbl in zip(dades, pos_pedestal, label):
            if p is None or np.isnan(p):
                print(f"[AVÍS] Canal '{lbl}' sense pedestal vàlid: exclòs de l'histograma combinat.")
                continue
            ax.hist(d - p, bins=num_bins, range=(-100, 300), density=False,
                    histtype='step', log=True, linewidth=1.0, label=str(lbl))

        ax.set_ylim(bottom=1)
        ax.set_xlabel('ADC counts (pedestal-subtracted)')
        fig.tight_layout()
        plt.show()

        area_promig = np.nanmean(area_pedestal) if area_pedestal else 0
        return pos_pedestal, sigma_pedestal, area_promig






def fit_gaussiana(
    ruta,
    asicid,
    smoothing_points,
    xl,
    xr,
    prominence_peak,
    fit_bool,
    subtract_bool,
    background_removal,
    pos_pedestal=None,
    ruta_bg=None,
    tbg=1,
    tsrc=1,
    num_canals=16,
    num_bins=1024,
    fit_type="gauss",
):
    """ Construction of the net spectra (subtraction of the pedestal and of the background) and gaussian fitting (linear or parabolic) of a photopeak between xl and xr """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    # Data load
    dades = np.array(cargar_canales_separados(ruta, asicid))

    if subtract_bool:
        if pos_pedestal is None:
            raise ValueError(
                "There is no 'pos_pedestal'"
            )
        dades_corregides = dades - pos_pedestal.reshape(num_canals, 1)
        vector_total = np.sum(dades_corregides, axis=0)
    else:
        vector_total = np.sum(dades, axis=0)

    # Globat hist
    counts, bin_edges = np.histogram(vector_total, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Background subtraction (optional)
    if background_removal and subtract_bool:
        if ruta_bg is None:
            raise ValueError(
                "Cal proporcionar 'ruta_bg' quan background_removal=True."
            )

        dades_bg = np.array(cargar_canales_separados(ruta_bg, asicid))
        dades_corregides_bg = dades_bg - pos_pedestal.reshape(num_canals, 1)
        vector_total_bg = np.sum(dades_corregides_bg, axis=0)

        counts_bg, _ = np.histogram(
            vector_total_bg,
            bins=num_bins,
            range=(vector_total.min(), vector_total.max()),
        )

        time_bg = tbg
        time_source = tsrc
        counts_bg_time = counts_bg * (time_source / time_bg)
        counts_total = counts - counts_bg_time
    else:
        counts_total = counts

    # Smoothing (optional)
    smooth_y = smooth(counts_total, smoothing_points)

    # Global spectrum
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(bin_centers, counts_total, linewidth=2)
    plt.title(f"Histograma Global - Tots els canals ({len(vector_total)} events)")
    plt.xlabel("ADC counts")
    plt.ylabel("Events")
    plt.grid(True, alpha=0.3)

    magnituds = None
    delta_magnituds = None

    if fit_bool:
        # Peak detection between xl and xr
        mask = (bin_centers > xl) & (bin_centers < xr)
        y_peak = smooth_y[mask]
        x_peak = bin_centers[mask]

        peak, _ = find_peaks(y_peak, prominence=prominence_peak)
        if len(peak) == 0:
            raise RuntimeError(
                f"No s'ha detectat cap pic a la finestra [{xl}, {xr}]."
            )

        alcades = np.array([y_peak[i] for i in peak])
        idx_max = np.argmax(alcades)
        max_height = alcades[idx_max]
        x_pic = x_peak[peak[idx_max]].item()

        plt.axvline(x=xl, linestyle="--", color="red")
        plt.axvline(x=xr, linestyle="--", color="red")
        plt.scatter(
            x_peak[peak[idx_max]],
            y_peak[peak[idx_max]],
            color="red",
            zorder=4,
        )

        # Initial guess for the fit
        amp_guess = np.max(y_peak)
        mu_guess = x_pic
        sigma_guess = 20 

        # Choose the fitting function
        if fit_type == "gauss":
            fit_func = gauss
            p0 = [sigma_guess, mu_guess, amp_guess, 0, 0]
        elif fit_type == "gauss_parab":
            fit_func = gauss_parab
            p0 = [sigma_guess, mu_guess, amp_guess, 0, 0, 0]

        fit_1, cov = curve_fit(fit_func, x_peak, y_peak, p0)

        # Parameter extraction
        amp = fit_1[2]
        delta_amp = np.sqrt(cov[2][2])

        mu = fit_1[1]
        delta_mu = np.sqrt(cov[1][1])

        sigma = np.abs(fit_1[0])
        delta_sigma = np.sqrt(cov[0][0])

        fwhm = fwhm_factor * sigma
        delta_fwhm = fwhm_factor * delta_sigma

        bin_width = bin_edges[1] - bin_edges[0]
        area_gauss = sigma * amp * np.sqrt(2 * np.pi) / bin_width
        delta_area_gauss = (
            np.sqrt(2 * np.pi)
            * np.sqrt(sigma**2 * delta_amp**2 + amp**2 * delta_sigma**2)
            / bin_width
        )

        resol = 100 * fwhm / mu
        delta_resol = 100 * np.sqrt(
            delta_fwhm**2 / mu**2 + fwhm**2 / mu**4 * delta_mu**2
        )

        print(f"--- Ajust ({fit_type}) ---")
        print(f"mu = {mu} ± {delta_mu}")
        print(f"sigma = {sigma} ± {delta_sigma}")
        print(f"resolució = {resol} ± {delta_resol}")
        print(f"àrea = {area_gauss} ± {delta_area_gauss}")

        x_gauss = np.linspace(xl, xr, 150)
        y_gauss = fit_func(x_gauss, *fit_1)
        plt.plot(x_gauss, y_gauss, label=f"fit ({fit_type})", linewidth=3, color="green")

        magnituds = [mu, sigma, amp, fwhm, area_gauss]
        delta_magnituds = [
            delta_mu,
            delta_sigma,
            delta_amp,
            delta_fwhm,
            delta_area_gauss,
        ]

    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.xlim(left=-200)

    if not fit_bool:
        plt.axvline(x=xl, color="red")
        plt.axvline(x=xr, color="red")

    plt.show()

    # Second graphic: zoom to the photopeak
    if fit_bool:
        plt.figure(figsize=(15, 8), dpi=80)
        plt.scatter(
            mu,
            fit_func(mu, *fit_1),
            color="purple",
            s=100,
            zorder=4,
        )
        plt.plot(bin_centers, counts_total, label="Dades")
        plt.plot(bin_centers, smooth_y, label="Suavitzat")
        plt.plot(x_gauss, y_gauss, label=f"Fit ({fit_type})", linewidth=4)
        plt.xlabel("Canal ADC")
        plt.ylabel("Comptes")
        plt.xlim(xl, xr)
        plt.ylim(0, 1.15 * max_height)
        plt.legend()
        plt.show()

    return bin_centers, counts_total, magnituds, delta_magnituds



def plot_espectres_bg(
    ruta,
    ruta_bg,
    asicid=1,
    pos_pedestal=None,
    tbg=1,
    tsrc=1,
    num_canals=16,
    num_bins=1024,
    subtract_bool=True,
):
    """ Plots the raw spectrum and the reescaled background spectrum for comparison """

    # Loads and corrects data
    dades = np.array(cargar_canales_separados(ruta, asicid))

    if subtract_bool:
        if pos_pedestal is None:
            raise ValueError(
                "Cal proporcionar 'pos_pedestal' quan subtract_bool=True."
            )
        dades_corregides = dades - pos_pedestal.reshape(num_canals, 1)
        vector_total = np.sum(dades_corregides, axis=0)
    else:
        vector_total = np.sum(dades, axis=0)

    # Source hist
    counts_raw, bin_edges = np.histogram(vector_total, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Background hist
    dades_bg = np.array(cargar_canales_separados(ruta_bg, asicid))

    if subtract_bool:
        dades_corregides_bg = dades_bg - pos_pedestal.reshape(num_canals, 1)
        vector_total_bg = np.sum(dades_corregides_bg, axis=0)
    else:
        vector_total_bg = np.sum(dades_bg, axis=0)

    counts_bg, _ = np.histogram(
        vector_total_bg,
        bins=num_bins,
        range=(vector_total.min(), vector_total.max()),
    )

    # Reescaling
    counts_bg_scaled = counts_bg * (tsrc / tbg)
    counts_net = counts_raw - counts_bg_scaled

    # Raw and bg spectra
    plt.figure(figsize=(10, 6), dpi=80)
    plt.plot(
        bin_centers,
        counts_raw,
        label="Espectre brut (Raw)",
        color="royalblue",
        linewidth=1.5,
    )
    plt.plot(
        bin_centers,
        counts_bg_scaled,
        label=f"Fons reescalat (t_src/t_bg = {tsrc/tbg:.2f})",
        color="crimson",
        linewidth=1.5,
        linestyle="--",
    )
    plt.title("Comparativa: Espectre Brut vs Fons Reescalat")
    plt.xlabel("ADC counts")
    plt.ylabel("Comptes")
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

    # Net spectrum
    plt.figure(figsize=(10, 6), dpi=80)
    plt.plot(
        bin_centers, counts_net, label="Espectre Net", color="forestgreen", linewidth=1.5
    )
    plt.axhline(0, color="black", linestyle=":", alpha=0.6)  # Línia de zero
    plt.title("Espectre Net (Resta de Fons)")
    plt.xlabel("ADC counts")
    plt.ylabel("Comptes")
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return bin_centers, counts_raw, counts_bg_scaled, counts_net



def assign_markers(energies, shared=None, tol=5):
    """ Assigns the markers to each experiemntal point """
    if shared is None:
        shared = {'o': [80, 356], 's': [1173, 1332]}
    energies = np.asarray(energies, dtype=float)
    others = iter(['^', 'D', 'v', 'p', 'h', '*', 'X', '<', '>'])
    markers = [None] * len(energies)
    for i in np.argsort(energies):
        E = energies[i]
        for m, E_list in shared.items():
            if any(abs(E - E0) < tol for E0 in E_list):
                markers[i] = m
                break
        else:
            markers[i] = next(others)
    return markers


def energy_calib(adc, ener_teor, delta_adc,
                 save_fig_bool=False, output_filename='energy_calib.pdf'):
    """ Performs the energy calibration """
    adc = np.asarray(adc, dtype=float)
    delta_adc = np.asarray(delta_adc, dtype=float)
    ener_teor = np.asarray(ener_teor, dtype=float)

    # ---- Linear fit E = m*ch + b ----
    popt, pcov = curve_fit(recta, adc, ener_teor)
    m_ener, b_ener = popt
    dm_ener, db_ener = np.sqrt(np.diag(pcov))
    r = np.corrcoef(adc, ener_teor)[0, 1]

    print(f"Pendent (m): {m_ener} pm {dm_ener}")
    print(f"Ordenada a l'origen (b): {b_ener} pm {db_ener}")
    print(f"Coeficient de correlació lineal (r): {r}")

    # Conversion from ADC to energy
    ener = m_ener * adc + b_ener
    delta_ener = np.sqrt(adc**2 * dm_ener**2 + m_ener**2 * delta_adc**2 + db_ener**2)
    print(ener)
    print(delta_ener)

    # Residuals
    residuals = (ener_teor - ener) / delta_ener

    # Markers
    markers = assign_markers(ener_teor)
    open_style = dict(markerfacecolor='none', markeredgewidth=1.2, markersize=9,
                      capsize=4, linestyle='none', elinewidth=1)

    x_range = np.linspace(adc.min() * 0.9, adc.max() * 1.1, 100)
    y_fit = m_ener * x_range + b_ener

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        gridspec_kw={'height_ratios': [3, 1]},
    )
    fig.set_layout_engine('constrained', hspace=0.02)

    # Top panel: calibration
    for x, y, dx, m in zip(adc, ener_teor, delta_adc, markers):
        ax1.errorbar(x, y, fmt=m, color='black', zorder=3, **open_style)

    fit_line, = ax1.plot(x_range, y_fit, '-', color='crimson', linewidth=2, zorder=2)

    order = np.argsort(ener_teor)
    unique_markers = list(dict.fromkeys(markers[i] for i in order))
    exp_handles = tuple(
        Line2D([], [], marker=m, color='black', markerfacecolor='none',
               markeredgewidth=1.2, markersize=5, linestyle='none')
        for m in unique_markers
    )

    ax1.legend(
        handles=[exp_handles, fit_line],
        labels=['Experimental values', 'Linear regression'],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        handlelength=len(unique_markers) * 1.0,
        loc='upper left',
    )
    ax1.set_ylabel(r'$E_\mathrm{ref}$ (keV)')
    ax1.set_ylim(0,1400)
    ax1.set_xlim(500,12000)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: residuals in units of sigma
    ax2.axhspan(-2, 2, color='#F0F0F0', zorder=0)   # pm2sigma band
    ax2.axhspan(-1, 1, color='#D9D9D9', zorder=0)   # pm1sigma band
    ax2.axhline(0, color='black', linewidth=0.8)

    for x, p, m in zip(adc, residuals, markers):
        ax2.plot(x, p, marker=m, color='black', markerfacecolor='none',
                 markeredgewidth=1.2, markersize=9, linestyle='none')

    ax2.set_xlabel('ADC channel')
    ax2.set_ylabel(r'$(E_\mathrm{ref}-E_\mathrm{fit})/\delta E_\mathrm{fit}$')

    lim = max(3, np.ceil(np.abs(residuals).max()) + 0.5)   # symmetric, at least pm3sigma
    ax2.set_ylim(-lim, lim)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, symmetric=True))

    ax2.set_xlim(0,12000)
    fig.align_ylabels([ax1, ax2])

    if save_fig_bool:
        fig.savefig(output_filename, bbox_inches='tight')

    plt.show()

    for E_ref, E_fit, dE, p in zip(ener_teor, ener, delta_ener, residuals):
        print(f"{E_ref:9.2f} keV   E_fit = {E_fit:9.2f} ± {dE:.2f} keV   residuals = {p:+6.2f} sigma")

    return m_ener, dm_ener, b_ener, db_ener, ener, delta_ener






def energy_res(fwhm, delta_fwhm, adc, delta_adc, ener, delta_ener,
               m_ener, recta,
               individual_ylim_res=None, show=True):
    """ Computes FWHM and resolution in ADC and energy units 
        Fits FWHM^2 = m_e * E + b_e and makes plots """

    # Array inputs
    fwhm = np.asarray(fwhm, dtype=float)
    delta_fwhm = np.asarray(delta_fwhm, dtype=float)
    adc = np.asarray(adc, dtype=float)
    delta_adc = np.asarray(delta_adc, dtype=float)
    ener = np.asarray(ener, dtype=float)
    delta_ener = np.asarray(delta_ener, dtype=float)

    # FWHM in keV
    fwhm_ener = m_ener * fwhm
    delta_fwhm_ener = m_ener * delta_fwhm

    # FWHM^2
    fwhm_2_adc = fwhm**2
    delta_fwhm_2_adc = 2 * fwhm * delta_fwhm

    fwhm_2_ener = fwhm_ener**2
    delta_fwhm_2_ener = 2 * fwhm_ener * delta_fwhm_ener

    # RESOLUTIONS
    # In ADC units
    resol_adc = 100 * fwhm / adc
    delta_resol_adc = 100 * np.sqrt(
        delta_fwhm**2 / adc**2 + fwhm**2 / adc**4 * delta_adc**2
    )
    # In energy units
    resol_ener = 100 * fwhm_ener / ener
    delta_resol_ener = 100 * np.sqrt(
        delta_fwhm_ener**2 / ener**2 + fwhm_ener**2 / ener**4 * delta_ener**2
    )

    # Fit FWHM^2 = m_e * E + b_e
    popt_e, pcov_e = curve_fit(
        recta, ener, fwhm_2_ener,
        sigma=delta_fwhm_2_ener, absolute_sigma=True,
    )
    m_e, b_e = popt_e
    delta_m_e, delta_b_e = np.sqrt(np.diag(pcov_e))

    print(f"m_e = {m_e:.4f} ± {delta_m_e:.4f} keV")
    print(f"b_e = {b_e:.4f} ± {delta_b_e:.4f} keV^2")
    r = np.corrcoef(ener, fwhm_2_ener)[0, 1]
    print(f"r = {r:.6f}   (r^2 = {r**2:.6f})")
    print(f"R(E) = 100 * sqrt({m_e:.4f} * E + {b_e:.4f}) / E")

    x_fit_ener = np.linspace(ener.min() * 0.3, ener.max() * 1.2, 300)
    arg = m_e * x_fit_ener + b_e
    res_fit_curve = np.where(arg > 0, 100 * np.sqrt(np.clip(arg, 0, None)) / x_fit_ener, np.nan)

    # FWHM^2 vs ADC
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(adc, fwhm_2_adc, xerr=delta_adc, yerr=delta_fwhm_2_adc,
                 fmt='o', color='royalblue', capsize=3)
    ax1.set_title(r'$\mathrm{FWHM}^2$ vs ADC')
    ax1.set_xlabel('ADC channel')
    ax1.set_ylabel(r'$\mathrm{FWHM}^2$ (ADC channels$^2$)')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # FWHM^2 vs Energy
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.errorbar(ener, fwhm_2_ener, xerr=delta_ener, yerr=delta_fwhm_2_ener,
                 fmt='o', color='crimson', capsize=3, label='Experimental')
    ax2.set_title(r'$\mathrm{FWHM}^2$ vs Energy')
    ax2.set_xlabel('Energy (keV)')
    ax2.set_ylabel(r'$\mathrm{FWHM}^2$ (keV$^2$)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # Resolution (%) vs ADC
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.errorbar(adc, resol_adc, xerr=delta_adc, yerr=delta_resol_adc,
                 fmt='o', color='forestgreen', capsize=3)
    ax3.set_title('Resolution vs ADC')
    ax3.set_xlabel('ADC channel')
    ax3.set_ylabel(r'$\mathrm{FWHM}_{\mathrm{ADC}}/\mathrm{ADC}$ (%)')
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    # Resolution (%) vs Energy
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.errorbar(ener, resol_ener, xerr=delta_ener, yerr=delta_resol_ener,
                 fmt='o', color='darkorange', capsize=3, label='Experimental')
    ax4.plot(x_fit_ener, res_fit_curve, '-', color='black',
             label=r'Model: $100\sqrt{mE+b}/E$')
    ax4.set_title('Resolution vs Energy')
    ax4.set_xlabel('Energy (keV)')
    ax4.set_ylabel(r'$R = \mathrm{FWHM}_E/E$ (%)')
    ax4.legend()
    if individual_ylim_res is not None:
        ax4.set_ylim(*individual_ylim_res)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()

    if show:
        plt.show()

    return resol_ener, delta_resol_ener, fwhm_ener, delta_fwhm_ener, resol_adc, delta_resol_adc, m_e, delta_m_e, b_e, delta_b_e




def energy_eff(area, delta_area, activity, delta_activity, time, delta_time, branching_ratio, delta_branching_ratio, distance,delta_distance):
    """ Calculates absolute and intrinsic efficiency.
        Distance: distance in cm """
    # Convert all inputs to NumPy arrays
    area = np.array(area)
    delta_area = np.array(delta_area)
    activity = np.array(activity)
    delta_activity = np.array(delta_activity)
    time = np.array(time)
    delta_time = np.array(delta_time)
    branching_ratio = np.array(branching_ratio)
    delta_branching_ratio = np.array(delta_branching_ratio)
    distance = np.array(distance)
    delta_distance = np.array(delta_distance)

    # Absolute Efficiency
    abs_eff = area / (activity * time * branching_ratio)
    delta_abs_eff = (1/(activity*branching_ratio*time)) * np.sqrt(delta_area**2 +
                                                                  (area**2/activity**2)* delta_activity**2 +
                                                                  (area**2/time**2)* delta_time**2 +
                                                                  (area**2/branching_ratio**2)* delta_branching_ratio**2)


    # Square face (cm)
    a = 1.2

    # Geometric factor
    geom_fact_rect = (1.0 / np.pi) * np.arcsin(a**2 / (a**2 + 4.0 * distance**2))

    # Uncertainty
    dG_dd = -(4.0 * a**2) / (
            np.pi
            * (a**2 + 4.0 * distance**2)
            * np.sqrt(2.0 * a**2 + 4.0 * distance**2)
    )
   

    print(f"Derivative:{dG_dd}")

    delta_geom_fact_rect = np.abs(dG_dd) * delta_distance

    # Intrinsic eff
    int_eff = abs_eff / geom_fact_rect

    delta_int_eff = np.sqrt(
        (delta_abs_eff / geom_fact_rect) ** 2
        + (abs_eff * delta_geom_fact_rect / (geom_fact_rect**2)) ** 2
    )

    return abs_eff, delta_abs_eff, int_eff, delta_int_eff





def area_photopeak(bin_centers, counts, xl, xr, to_x=lambda x: x):
    """ Calculates net photopeak area """

    mask = (bin_centers >= xl) & (bin_centers <= xr)
    x_peak = bin_centers[mask]
    y_peak = counts[mask]

    if len(x_peak) < 2:
        return 0, 0

    x_l, x_r = x_peak[0], x_peak[-1]
    y_l, y_r = y_peak[0], y_peak[-1]

    m = (y_r - y_l) / (x_r - x_l)
    c = y_l - m * x_l
    baseline = m * x_peak + c

    net_heights = y_peak - baseline
    net_area_counts = np.sum(net_heights)

    bin_width = x_peak[1] - x_peak[0]
    continuum_area = (0.5 * m * (x_r**2 - x_l**2) + c * (x_r - x_l)) / bin_width
    raw_area = np.trapezoid(y_peak, x_peak) / bin_width

    error_net_area = np.sqrt(continuum_area + raw_area)

    xp = to_x(x_peak)
    plt.fill_between(xp, baseline, y_peak, color='orange', alpha=0.4,
                     label=f'Net area: {net_area_counts:.1f} counts')
    plt.plot(xp, baseline, color='red', linestyle='--', linewidth=2,
             label='Continuum (linear interp.)')
    plt.scatter(to_x(np.array([x_l, x_r])), [y_l, y_r], color='red', marker='s',
                s=90, zorder=10, label='Reference points')

    return net_area_counts, error_net_area


def save_pdf(fig, filename):
    """ Save the figure as a pdf """
    root, _ = os.path.splitext(filename)
    fig.savefig(root + '.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)


def fit_gaussiana_and_area(
    ruta,
    asicid,
    smoothing_points,
    xl,
    xr,
    prominence_peak,
    fit_bool,
    subtract_bool,
    background_removal,
    pos_pedestal=None,
    ruta_bg=None,
    tbg=1,
    tsrc=1,
    num_canals=16,
    num_bins=1200,
    fit_type="gauss",
    calc_area_bool=True,
    area_xl=4170,
    area_xr=4650,
    plot_label="",
    save_fig_bool=False,
    output_filename="areaNa_fit.pdf",
    calib=None,
    plot_xlim=None,
    plot_ylim=None,
    label_pos_x=1,
    label_pos_y=1,
):
    """ Gaussian fit and photopeak net area calculation  """


    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    # Plots ADC channel if no calibration parameters is provided or plots E
    if calib is not None:
        def to_x(ch):
            return np.polyval(calib, ch)
        dcalib = np.polyder(np.asarray(calib, dtype=float))
        xlabel = r'$E$ (keV)'
    else:
        def to_x(ch):
            return ch
        xlabel = 'ADC channel'
    ylabel = 'Counts'

    # Load Data
    dades = np.array(cargar_canales_separados(ruta, asicid))

    if subtract_bool:
        dades_corregides = dades - pos_pedestal.reshape(num_canals, 1)
        vector_total = np.sum(dades_corregides, axis=0)
    else:
        vector_total = np.sum(dades, axis=0)

    # Compute Histogram
    counts, bin_edges = np.histogram(vector_total, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_plot = to_x(bin_centers)

    # Normalised background removal
    if background_removal and subtract_bool:
        dades_bg = np.array(cargar_canales_separados(ruta_bg, asicid))
        dades_corregides_bg = dades_bg - pos_pedestal.reshape(num_canals, 1)
        vector_total_bg = np.sum(dades_corregides_bg, axis=0)

        counts_bg, _ = np.histogram(
            vector_total_bg,
            bins=num_bins,
            range=(vector_total.min(), vector_total.max()),
        )

        counts_bg_time = counts_bg * (tsrc / tbg)
        counts_total = counts - counts_bg_time
    else:
        counts_total = counts

    # Smooth spectrum
    smooth_y = smooth(counts_total, smoothing_points)

    magnituds = None
    delta_magnituds = None
    net_area, error_net_area = None, None

    # Gaussian fit
    if fit_bool == True:
        mask = (bin_centers > xl) & (bin_centers < xr)
        y_peak = smooth_y[mask]
        x_peak = bin_centers[mask]

        peak, _ = find_peaks(y_peak, prominence=prominence_peak)
        if len(peak) == 0:
            raise RuntimeError(f"No s'ha detectat cap pic a la finestra [{xl}, {xr}].")

        alcades = np.array([y_peak[i] for i in peak])
        idx_max = np.argmax(alcades)
        max_height = alcades[idx_max]
        x_pic = x_peak[peak[idx_max]].item()

        amp_guess = np.max(y_peak)
        mu_guess = x_pic
        sigma_guess = 20

        if fit_type == "gauss":
            fit_func = gauss
            p0 = [sigma_guess, mu_guess, amp_guess, 0, 0]
        elif fit_type == "gauss_parab":
            fit_func = gauss_parab
            p0 = [sigma_guess, mu_guess, amp_guess, 0, 0, 0]
        else:
            raise ValueError(f"Model '{fit_type}' no vàlid. Utilitza 'gauss' o 'gauss_parab'.")

        fit_1, cov = curve_fit(fit_func, x_peak, y_peak, p0)

        # Parameter Calculations (in ADC channels)
        amp, delta_amp = fit_1[2], np.sqrt(cov[2][2])
        mu, delta_mu = fit_1[1], np.sqrt(cov[1][1])
        sigma, delta_sigma = np.abs(fit_1[0]), np.sqrt(cov[0][0])

        fwhm = fwhm_factor * sigma
        delta_fwhm = fwhm_factor * delta_sigma

        bin_width = bin_edges[1] - bin_edges[0]
        area_gauss = sigma * amp * np.sqrt(2 * np.pi) / bin_width
        delta_area_gauss = (
            np.sqrt(2 * np.pi)
            * np.sqrt(sigma**2 * delta_amp**2 + amp**2 * delta_sigma**2)
            / bin_width
        )

        resol = 100 * fwhm / mu
        delta_resol = 100 * np.sqrt(delta_fwhm**2 / mu**2 + fwhm**2 / mu**4 * delta_mu**2)

        print(f"--- Fit ({fit_type}) ---")
        print(f"mu = {mu:.3f} ± {delta_mu:.3f} ch")
        print(f"sigma = {sigma:.3f} ± {delta_sigma:.3f} ch")
        print(f"R (%) [canals] = {resol:.2f} ± {delta_resol:.2f}")
        print(f"area (gauss) = {area_gauss:.2f} ± {delta_area_gauss:.2f}")

        # Same quantities in energy (calibration uncertainty not included)
        if calib is not None:
            slope = abs(np.polyval(dcalib, mu))
            mu_E, delta_mu_E = to_x(mu), slope * delta_mu
            fwhm_E, delta_fwhm_E = slope * fwhm, slope * delta_fwhm
            resol_E = 100 * fwhm_E / mu_E
            delta_resol_E = 100 * np.sqrt(delta_fwhm_E**2 / mu_E**2
                                          + fwhm_E**2 / mu_E**4 * delta_mu_E**2)
            print(f"E(mu) = {mu_E:.2f} ± {delta_mu_E:.2f} keV")
            print(f"FWHM = {fwhm_E:.2f} ± {delta_fwhm_E:.2f} keV")
            print(f"R (%) [energy] = {resol_E:.2f} ± {delta_resol_E:.2f}")

        # Zoom plot
        fig, ax = plt.subplots(figsize=(15, 8))

        if calc_area_bool:
            net_area, error_net_area = area_photopeak(bin_centers, counts_total,
                                                      area_xl, area_xr, to_x=to_x)
            print("--- Linear Subtraction Calculation ---")
            print(f"Net Peak Area = {net_area:.2f} ± {error_net_area:.2f} counts\n")

        x_gauss = np.linspace(xl, xr, 300)
        y_gauss = fit_func(x_gauss, *fit_1)

        ax.scatter(to_x(mu), fit_func(mu, *fit_1), color="purple", s=100, zorder=4,
                   label="Centroid (Gaussian)")
        ax.scatter(x_plot[mask], counts_total[mask], color="#051282", s=80, zorder=5,
                   label="Fitting points")
        ax.plot(x_plot, counts_total, color="#FF570A", label="Net spectrum")
        ax.plot(to_x(x_gauss), y_gauss, linewidth=4, color="green",
                label="Gaussian fit")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(plot_xlim if plot_xlim is not None else (to_x(xl), to_x(xr)))
        ax.set_ylim(plot_ylim if plot_ylim is not None else (0, 1.15 * max_height))
        ax.legend(loc="upper right")

        if plot_label:
            ax.text(label_pos_x, label_pos_y, plot_label, transform=ax.transAxes,
                    va="top", ha="left", fontsize=25)

        if save_fig_bool:
            save_pdf(fig, output_filename)

        plt.show()

        magnituds = [mu, sigma, amp, fwhm, area_gauss, net_area]
        delta_magnituds = [delta_mu, delta_sigma, delta_amp, delta_fwhm, delta_area_gauss, error_net_area]

    # No Fit
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(x_plot, counts_total, color='black', linewidth=1)
        ax.axvline(x=to_x(xl), color='red')
        ax.axvline(x=to_x(xr), color='red')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1)
        if plot_xlim is not None:
            ax.set_xlim(plot_xlim)
        else:
            ax.set_xlim(left=to_x(-200))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if plot_label:
            ax.text(label_pos_x, label_pos_y, plot_label, transform=ax.transAxes,
                    va="top", ha="left", fontsize=20, fontweight='bold')

        if save_fig_bool:
            save_pdf(fig, output_filename)

        plt.show()

    return bin_centers, counts_total, net_area, error_net_area, magnituds, delta_magnituds
