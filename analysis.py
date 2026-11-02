import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd

def gauss(x, sigma, mu, A,B,C):
    y = A*np.exp(-0.5*(x-mu)**2/sigma**2) + B*x + C
    return y

def gauss_parab(x, sigma, mu, A,B,C,D):
    y = A*np.exp(-0.5*(x-mu)**2/sigma**2) + B*x**2 + C*x + D
    return y

def gauss_solo(x, sigma, mu, A):
    y = A*np.exp(-0.5*(x-mu)**2/sigma**2)
    return y


def smooth(y, box_pts):
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
        # Extraemos la columna de 50,000 valores como un array independiente
        vector_canal = df_asic[nombre_columna].to_numpy()

        # Lo añadimos a nuestra lista
        lista_de_vectores.append(vector_canal)

    return lista_de_vectores




def plot_acq_pedestal(dades, label, fit_bool, xl_1, xr_1, num_bins, peak_prom=50,ruta_txt='HG_espectres_pedestal.txt'):

    # 1. Configuració del layout
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

    # 2. Bucle de ploteig
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
            # --- Cerca de pics, amb prominència fixa o dinàmica ---
            if peak_prom == 'din':
                prom_dinamica = max(10, np.max(counts) * 0.1)
                peak, _ = find_peaks(y_peak1, prominence=prom_dinamica)
                #y_log = np.log10(np.maximum(y_peak1, 1))
                #peak, _ = find_peaks(y_log, prominence=0.1, height=np.log10(50))
                #peak = peak[1:2]   # només el segon pic (el primer és el pedestal)
            else:
                peak, _ = find_peaks(y_peak1, prominence=peak_prom)

            if len(peak) == 0:
                # Alternativa: si no es detecta cap pic, fem servir el màxim pur de la zona
                if len(y_peak1) > 0:
                    x_pic_1 = x_peak1[np.argmax(y_peak1)]
                else:
                    print(f"[AVÍS] Canal idx={idx} (label={label[idx]}): "
                          f"finestra xl_1={xl_1}, xr_1={xr_1} buida. S'omet el fit.")
                    x_pic_1 = None
            else:
                height_1 = np.array([y_peak1[i] for i in peak])
                index_max_1 = np.argmax(height_1)
                x_pic_1 = x_peak1[peak[index_max_1]].item()
                axs[idx].scatter(x_peak1[peak[index_max_1]], y_peak1[peak[index_max_1]],
                                  color='red', zorder=4)

            centro_final = x_pic_1  # valor per defecte si el fit gaussià falla

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

        # --- Guardem sempre, encara que hagi fallat, per no desalinear els canals ---
        pos_pedestal.append(centro_final)
        sigma_pedestal.append(sigma)
        area_pedestal.append(area)

        # Ajustos de format per cada subplot
        axs[idx].set_xlabel('ADC counts')
        axs[idx].set_ylabel('Events')
        axs[idx].set_xlim(.99 * np.min(d), 2 * np.mean(d))
        axs[idx].set_ylim(bottom=1)
        axs[idx].legend(loc='upper right', fontsize='small')
        axs[idx].grid(True, alpha=0.3)

    # 3. Neteja: Amagar quadres buits si n'hi ha
    for k in range(idx + 1, len(axs)):
        axs[k].set_visible(False)

    plt.tight_layout()
    plt.show()

    if fit_bool:
        # --- Histograma combinat, restant el pedestal a cada canal ---

        # --- Guardar els 16 espectres (pedestal restat) en un txt ---
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

        #np.savetxt(ruta_txt, np.column_stack(columnes), delimiter='\t',
        #           header='\t'.join(noms), comments='',
        #           fmt=['%.2f'] + ['%d'] * (len(columnes) - 1))
        #print(f"Espectres guardats a: {ruta_txt}")

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
    """Ajusta una gaussiana o gaussiana + fons parabòlic a l'espectre global.

    Paràmetres addicionals ---------- fit_type : str, opcional Models disponibles:
    'gauss' (default) o 'gauss_parab'.
    """

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    # --- Càrrega de dades ---
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

    # --- Histograma global ---
    counts, bin_edges = np.histogram(vector_total, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- Resta de fons (opcional) ---
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

    # --- Suavitzat de l'histograma ---
    smooth_y = smooth(counts_total, smoothing_points)

    # --- Gràfic principal: espectre global ---
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(bin_centers, counts_total, linewidth=2)
    plt.title(f"Histograma Global - Tots els canals ({len(vector_total)} events)")
    plt.xlabel("ADC counts")
    plt.ylabel("Events")
    plt.grid(True, alpha=0.3)

    magnituds = None
    delta_magnituds = None

    if fit_bool:
        # --- Detecció del pic dins la finestra [xl, xr] ---
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

        # --- Selecció del model i paràmetres inicials ---
        amp_guess = np.max(y_peak)
        mu_guess = x_pic
        sigma_guess = 20  # basat en la resolució típica del detector

        if fit_type == "gauss":
            fit_func = gauss
            p0 = [sigma_guess, mu_guess, amp_guess, 0, 0]
        elif fit_type == "gauss_parab":
            fit_func = gauss_parab
            p0 = [sigma_guess, mu_guess, amp_guess, 0, 0, 0]
        else:
            raise ValueError(
                f"Model '{fit_type}' no vàlid. Utilitza 'gauss' o 'gauss_parab'."
            )

        fit_1, cov = curve_fit(fit_func, x_peak, y_peak, p0)

        # Extracció de paràmetres (sigma=index 0, mu=index 1, amp=index 2)
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

    # --- Segon gràfic: zoom lineal sobre el pic ajustat ---
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


import matplotlib.pyplot as plt
import numpy as np


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
    """Calcula i ploteja l'espectre brut (raw) i de fons (background) reescalat

    en una gràfica, i l'espectre net (net spectrum) en una altra.
    """
    # --- 1. Càrrega i correcció de dades de la font (Raw) ---
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

    # Histograma de la font (Raw)
    counts_raw, bin_edges = np.histogram(vector_total, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # --- 2. Càrrega i correcció de dades de fons (Background) ---
    dades_bg = np.array(cargar_canales_separados(ruta_bg, asicid))

    if subtract_bool:
        dades_corregides_bg = dades_bg - pos_pedestal.reshape(num_canals, 1)
        vector_total_bg = np.sum(dades_corregides_bg, axis=0)
    else:
        vector_total_bg = np.sum(dades_bg, axis=0)

    # Histograma de fons en el mateix rang
    counts_bg, _ = np.histogram(
        vector_total_bg,
        bins=num_bins,
        range=(vector_total.min(), vector_total.max()),
    )

    # --- 3. Reescalat temporal i càlcul de l'espectre net ---
    counts_bg_scaled = counts_bg * (tsrc / tbg)
    counts_net = counts_raw - counts_bg_scaled

    # --- 4. Gràfica 1: Raw Spectrum vs Background Spectrum ---
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

    # --- 5. Gràfica 2: Net Spectrum ---
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



# def energy_calib(adc,ener_teor,delta_adc):
#     adc = np.asarray(adc)
#     delta_adc = np.asarray(delta_adc)

#     popt, pcov = curve_fit(recta, adc, ener_teor)

#     m_ener, b_ener = popt
#     perr = np.sqrt(np.diag(pcov)) # Desviación estándar de los parámetros
#     dm_ener, db_ener = perr # Errores de m y b
#     r_matrix = np.corrcoef(adc, ener_teor)
#     r = r_matrix[0, 1]

#     # 3. Crear la línea del ajuste
#     x_range = np.linspace(min(adc)*0.9, max(adc)*1.1, 100)
#     y_fit = m_ener * x_range + b_ener

#     # 4. Graficar
#     plt.figure(figsize=(8, 5))

#     # Puntos de datos con barras de error en X
#     plt.errorbar(adc, ener_teor, xerr=delta_adc, fmt='o', color='red', label='Experimental values', capsize=5)

#     # Línea del ajuste
#     plt.plot(x_range, y_fit, color='blue', linestyle='--', label=f'Ajuste: y = {m_ener:.2f}x + {b_ener:.2f}')

#     plt.xlabel('ADC channel')
#     plt.ylabel(r'$E_\mathrm{ref}$ (keV)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     #plt.scatter(pic6[0],511)




#     plt.show()


#     print(f"Pendent (m): {m_ener} pm {dm_ener}")
#     print(f"Ordenada a l'origen (b): {b_ener} pm {db_ener}")
#     print(f"Coeficient de correlació lineal (r): {r}")



#     #Calculem energies centroides (amb ajust)
#     ener = m_ener*adc + b_ener
#     delta_ener = np.sqrt(adc**2 * dm_ener**2 + m_ener**2 * delta_adc**2 + db_ener**2)

#     print(ener)
#     print(delta_ener)

#     residuals = 100 * (ener_teor - ener) / ener
#     delta_residuals = 100 * np.abs(ener_teor - 2 * ener) / ener **2 * delta_ener

#     plt.figure(figsize=(8, 5))

#     plt.errorbar(adc, residuals, xerr=delta_adc,yerr=delta_residuals ,fmt='o', color='red', label='Experimental values', capsize=5)


#     plt.xlabel('Canal ADC')
#     plt.ylabel('Residuals (%)')
#     plt.axhline(y=0, linestyle='--', color='black')


#     plt.legend()
#     plt.grid(True, alpha=0.3)




#     plt.show()


#     return m_ener, dm_ener, b_ener, db_ener, ener, delta_ener


def assign_markers(energies, shared=None, tol=5):
    """
    Same marker scheme as the efficiency plot: 80 & 356 keV share 'o',
    1173 & 1332 keV share 's', the rest get their own marker in order of
    increasing energy. Returns markers aligned with the input order.
    """
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

    # ---- Centroid energies from the fit ----
    ener = m_ener * adc + b_ener
    delta_ener = np.sqrt(adc**2 * dm_ener**2 + m_ener**2 * delta_adc**2 + db_ener**2)
    print(ener)
    print(delta_ener)

    # ---- Pulls: difference in units of the uncertainty of E_fit ----
    pulls = (ener_teor - ener) / delta_ener

    # ---- Markers & styles ----
    markers = assign_markers(ener_teor)
    open_style = dict(markerfacecolor='none', markeredgewidth=1.2, markersize=9,
                      capsize=4, linestyle='none', elinewidth=1)

    x_range = np.linspace(adc.min() * 0.9, adc.max() * 1.1, 100)
    y_fit = m_ener * x_range + b_ener

    # ---- Figure: two panels, shared x axis ----
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

    # Bottom panel: pulls in units of sigma
    ax2.axhspan(-2, 2, color='#F0F0F0', zorder=0)   # ±2σ band
    ax2.axhspan(-1, 1, color='#D9D9D9', zorder=0)   # ±1σ band
    ax2.axhline(0, color='black', linewidth=0.8)

    for x, p, m in zip(adc, pulls, markers):
        ax2.plot(x, p, marker=m, color='black', markerfacecolor='none',
                 markeredgewidth=1.2, markersize=9, linestyle='none')

    ax2.set_xlabel('ADC channel')
    ax2.set_ylabel(r'$(E_\mathrm{ref}-E_\mathrm{fit})/\delta E_\mathrm{fit}$')

    lim = max(3, np.ceil(np.abs(pulls).max()) + 0.5)   # symmetric, at least ±3σ
    ax2.set_ylim(-lim, lim)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, symmetric=True))

    ax2.set_xlim(0,12000)
    fig.align_ylabels([ax1, ax2])

    if save_fig_bool:
        fig.savefig(output_filename, bbox_inches='tight')

    plt.show()

    for E_ref, E_fit, dE, p in zip(ener_teor, ener, delta_ener, pulls):
        print(f"{E_ref:9.2f} keV   E_fit = {E_fit:9.2f} ± {dE:.2f} keV   pull = {p:+6.2f} sigma")

    return m_ener, dm_ener, b_ener, db_ener, ener, delta_ener






def energy_res(fwhm, delta_fwhm, adc, delta_adc, ener, delta_ener,
               m_ener, dm_ener, recta,
               individual_ylim_res=None, show=True):
    """
    Energy-resolution analysis: fit FWHM^2 vs Energy and make four plots.

    The ADC -> energy calibration is E = m_ener * ADC + b_ener, so a width
    converts as FWHM_E = m_ener * FWHM_ADC (the intercept cancels).

    Parameters
    ----------
    fwhm, delta_fwhm : array-like
        FWHM of each peak and its uncertainty, in ADC channels.
    adc, delta_adc : array-like
        Peak positions (ADC channels) and their uncertainties.
    ener, delta_ener : array-like
        Reference energies of the peaks (keV) and their uncertainties.
    m_ener, dm_ener : float
        Slope of the ADC -> energy calibration (keV/channel) and its error.
    recta : callable
        Linear model for curve_fit, recta(x, m, b) = m*x + b.
    individual_ylim_res : tuple or None
        Optional (bottom, top) y-limits for the Resolution vs Energy plot.
    show : bool
        Call plt.show() at the end. Set False if you want to save the
        returned figures afterwards.

    Returns
    -------
    dict with resolution vectors, FWHM in energy units, fit parameters
    (statistical and total errors), chi2/ndf and the figures.
    """

    # --- 0. Inputs as arrays ---
    fwhm = np.asarray(fwhm, dtype=float)
    delta_fwhm = np.asarray(delta_fwhm, dtype=float)
    adc = np.asarray(adc, dtype=float)
    delta_adc = np.asarray(delta_adc, dtype=float)
    ener = np.asarray(ener, dtype=float)
    delta_ener = np.asarray(delta_ener, dtype=float)

    # --- 1. FWHM in energy units ---
    fwhm_ener = m_ener * fwhm
    # Statistical (point-to-point) error only: used in the fit
    delta_fwhm_ener_stat = m_ener * delta_fwhm
    # Total error including the calibration slope (fully correlated): for display
    delta_fwhm_ener = np.sqrt(fwhm**2 * dm_ener**2 + m_ener**2 * delta_fwhm**2)

    fwhm_2_adc = fwhm**2
    delta_fwhm_2_adc = 2 * fwhm * delta_fwhm

    fwhm_2_ener = fwhm_ener**2
    delta_fwhm_2_ener_stat = 2 * fwhm_ener * delta_fwhm_ener_stat
    delta_fwhm_2_ener = 2*fwhm_ener*delta_fwhm_ener

    # --- 2. Resolutions ---
    # In ADC units (only meaningful vs ADC)
    resol_adc = 100 * fwhm / adc
    delta_resol_adc = 100 * np.sqrt(
        delta_fwhm**2 / adc**2 + fwhm**2 / adc**4 * delta_adc**2
    )
    # In energy units (the physical resolution, consistent with R(E) model)
    resol_ener = 100 * fwhm_ener / ener
    delta_resol_ener = 100 * np.sqrt(
        delta_fwhm_ener**2 / ener**2 + fwhm_ener**2 / ener**4 * delta_ener**2
    )

    # --- 3. Fit FWHM^2 = m_e * E + b_e (statistical errors only) ---
    popt_e, pcov_e = curve_fit(
        recta, ener, fwhm_2_ener,
        sigma=delta_fwhm_2_ener_stat, absolute_sigma=True,
    )
    m_e, b_e = popt_e
    perr_stat = np.sqrt(np.diag(pcov_e))

    # Calibration-slope systematic: FWHM_E^2 scales as m_ener^2, so both
    # fit parameters carry a relative error of 2 * dm_ener / m_ener
    rel_syst = 2 * dm_ener / m_ener
    perr_syst = np.abs(popt_e) * rel_syst
    perr = np.sqrt(perr_stat**2 + perr_syst**2)

    residuals = fwhm_2_ener - recta(ener, *popt_e)
    chi2 = np.sum((residuals / delta_fwhm_2_ener_stat)**2)
    ndf = len(ener) - len(popt_e)

    print(f"m_e = {m_e:.4f} ± {perr_stat[0]:.4f} (stat) ± {perr_syst[0]:.4f} (cal) keV")
    print(f"b_e = {b_e:.4f} ± {perr_stat[1]:.4f} (stat) ± {perr_syst[1]:.4f} (cal) keV^2")
    # Pearson linear correlation coefficient
    r = np.corrcoef(ener, fwhm_2_ener)[0, 1]
    print(f"r = {r:.6f}   (r^2 = {r**2:.6f})")
    #print(f"chi2/ndf = {chi2:.2f}/{ndf}")
    print(f"R(E) = 100 * sqrt({m_e:.4f} * E + {b_e:.4f}) / E")

    x_fit_ener = np.linspace(ener.min() * 0.3, ener.max() * 1.2, 300)
    y_fit_f2_ener = recta(x_fit_ener, m_e, b_e)
    arg = m_e * x_fit_ener + b_e
    res_fit_curve = np.where(arg > 0, 100 * np.sqrt(np.clip(arg, 0, None)) / x_fit_ener, np.nan)

    # --- 4. Plots ---
    figs = {}

    # Plot 1: FWHM^2 vs ADC
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(adc, fwhm_2_adc, xerr=delta_adc, yerr=delta_fwhm_2_adc,
                 fmt='o', color='royalblue', capsize=3)
    ax1.set_title(r'$\mathrm{FWHM}^2$ vs ADC')
    ax1.set_xlabel('ADC channel')
    ax1.set_ylabel(r'$\mathrm{FWHM}^2$ (ADC channels$^2$)')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    figs['fwhm2_vs_adc'] = fig1

    # Plot 2: FWHM^2 vs Energy (error bars = those used in the fit)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.errorbar(ener, fwhm_2_ener, xerr=delta_ener, yerr=delta_fwhm_2_ener,
                 fmt='o', color='crimson', capsize=3, label='Experimental')
    ax2.plot(x_fit_ener, y_fit_f2_ener, '--', color='black',
             label=rf'Fit: $mE+b$, $\chi^2/\nu = {chi2:.1f}/{ndf}$')
    ax2.set_title(r'$\mathrm{FWHM}^2$ vs Energy')
    ax2.set_xlabel('Energy (keV)')
    ax2.set_ylabel(r'$\mathrm{FWHM}^2$ (keV$^2$)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    figs['fwhm2_vs_energy'] = fig2

    # Plot 3: Resolution (%) vs ADC (ADC-based resolution)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.errorbar(adc, resol_adc, xerr=delta_adc, yerr=delta_resol_adc,
                 fmt='o', color='forestgreen', capsize=3)
    ax3.set_title('Resolution vs ADC')
    ax3.set_xlabel('ADC channel')
    ax3.set_ylabel(r'$\mathrm{FWHM}_{\mathrm{ADC}}/\mathrm{ADC}$ (%)')
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    figs['resolution_vs_adc'] = fig3

    # Plot 4: Resolution (%) vs Energy (energy-based resolution + model)
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
    figs['resolution_vs_energy'] = fig4

    if show:
        plt.show()

    return {
        "fwhm_ener": fwhm_ener,
        "delta_fwhm_ener": delta_fwhm_ener,
        "delta_fwhm_ener_stat": delta_fwhm_ener_stat,
        "resol_adc": resol_adc,
        "delta_resol_adc": delta_resol_adc,
        "resol_ener": resol_ener,
        "delta_resol_ener": delta_resol_ener,
        "m_e": m_e,
        "b_e": b_e,
        "perr_stat": perr_stat,
        "perr_syst": perr_syst,
        "perr": perr,
        "chi2": chi2,
        "ndf": ndf,
        "figs": figs,
    }




def energy_eff(area, delta_area, activity, delta_activity, time, delta_time, branching_ratio, delta_branching_ratio, distance,delta_distance):
    """
    Calculates absolute and intrinsic efficiency, avoiding division by zero when distance is 0.
    Distance: distance in cm
    """
    # 1. Convert all inputs to NumPy arrays for element-wise arithmetic
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

    # 2. Calculate Absolute Efficiency
    abs_eff = area / (activity * time * branching_ratio)
    delta_abs_eff = (1/(activity*branching_ratio*time)) * np.sqrt(delta_area**2 +
                                                                  (area**2/activity**2)* delta_activity**2 +
                                                                  (area**2/time**2)* delta_time**2 +
                                                                  (area**2/branching_ratio**2)* delta_branching_ratio**2) # + ( (2/distance)*(area)*delta_distance)**2    )

    # 3. Geometric Efficiency Calculation (Solid Angle Fraction)
    radius_eff = 1.2 / np.sqrt(np.pi)

    # Avoid ZeroDivisionError when distance == 0 (Source directly on detector surface)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = np.where(
            distance == 0,
            0.0,  # cos(pi/2) = 0 for 2pi coverage
            1.0 / np.sqrt(1.0 + (radius_eff / distance)**2)
        )

    omega = 2.0 * np.pi * (1.0 - cos_theta)
    solid_angle_fraction = omega / (4.0 * np.pi)  # Equals 0.5 when distance == 0

    # Mida de la cara rectangular del detector (cm)
    a = 1.2

    # 3. Factor geomètric
    geom_fact_rect = (1.0 / np.pi) * np.arcsin(a**2 / (a**2 + 4.0 * distance**2))

    # Incertesa de G(d) respecte a la distància (dG/dd * delta_distance)
    with np.errstate(divide="ignore", invalid="ignore"):
        dG_dd = -(4.0 * a**2) / (
            np.pi
            * (a**2 + 4.0 * distance**2)
            * np.sqrt(2.0 * a**2 + 4.0 * distance**2)
        )
    # Gestió de quan la distància és 0 per evitar NaNs
    dG_dd = np.where(distance == 0, 0.0, dG_dd)

    print(f"Derivative:{dG_dd}")

    delta_geom_fact_rect = np.abs(dG_dd) * delta_distance

    # 4. Eficiència Intrínseca i la teva incertesa
    int_eff = abs_eff / geom_fact_rect

    delta_int_eff = np.sqrt(
        (delta_abs_eff / geom_fact_rect) ** 2
        + (abs_eff * delta_geom_fact_rect / (geom_fact_rect**2)) ** 2
    )

    return abs_eff, delta_abs_eff, int_eff, delta_int_eff


def area_photopeak(bin_centers, counts, xl, xr):
    """
    Calculates the net peak area by summing bin heights minus a linear baseline
    and shades the region on the active matplotlib plot.
    """
    mask = (bin_centers >= xl) & (bin_centers <= xr)
    x_peak = bin_centers[mask]
    y_peak = counts[mask]

    if len(x_peak) < 2:
        print("Warning: Window xl to xr is too narrow or contains no data points.")
        return 0, 0

    x_l, x_r = x_peak[0], x_peak[-1]
    y_l, y_r = y_peak[0], y_peak[-1]

    # Linear baseline calculation: y = m * x + c
    m = (y_r - y_l) / (x_r - x_l)
    c = y_l - m * x_l
    baseline = m * x_peak + c

    # Bin summation & integration
    net_heights = y_peak - baseline
    net_area_counts = np.sum(net_heights)

    bin_width = x_peak[1] - x_peak[0]
    continuum_area = (0.5 * m * (x_r**2 - x_l**2) + c * (x_r - x_l)) / bin_width
    raw_area = np.trapezoid(y_peak, x_peak) / bin_width

    error_net_area = np.sqrt(continuum_area + raw_area)

    # Shading and reference baseline points on current figure
    plt.fill_between(x_peak, baseline, y_peak, color='orange', alpha=0.4,
                     label=f'Net Area: {net_area_counts:.1f} cts')
    plt.plot(x_peak, baseline, color='red', linestyle='--', linewidth=2,
             label='Linear Baseline (continuum fit)')
    plt.scatter([x_l, x_r], [y_l, y_r], color='red', marker='s', s=90, zorder=10,
                label='Reference points (continuum fit)')

    return net_area_counts, error_net_area