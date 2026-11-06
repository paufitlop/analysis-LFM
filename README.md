# Characterization of a 16-channel LaBr₃ detector

Python analysis code for characterizing a LaBr₃ scintillation detector read out by 16 channels of an ASIC. It covers pedestal and background subtraction, photopeak fitting, **energy calibration**, **energy resolution**, and **absolute and intrinsic efficiency**, compared against a Geant4 simulation.

## Repository contents

| File | Description |
|---|---|
| `analysis.py` | Module with all the analysis functions: data loading, pedestals, net spectra, fits, calibration, resolution, peak areas and efficiency. |
| `energy_calib.ipynb` | Notebook for energy calibration and energy resolution. |
| `area_photopeak_plots.ipynb` | Notebook for net photopeak areas and detector efficiency. |

## Requirements

- Python ≥ 3.9
- `numpy` ≥ 2.0 (the code uses `np.trapezoid`)
- `scipy`
- `matplotlib`
- `pandas`
- `jupyter`

```bash
pip install numpy scipy matplotlib pandas jupyter
```

## Data format

The notebooks expect the data files in a `Data/` folder next to the code. Each file is tab-separated, with an `ASIC` column and one column per channel, `CH[0]` … `CH[15]`. Each row is one event.

```
TRG_ID	CH[0]	CH[1]	...	CH[15]
1	2412	2398	...	2405
```

Files used (LG12 gain, 36.5 V):

| Measurement | File | Time (s) |
|---|---|---|
| Pedestal (LG) | `pedestal_LaBr_fpga_LG12_36.5V_30sec.txt` | 30 |
| Pedestal (HG) | `pedestal_LaBr_fpga_HG1_36.5V_30sec.txt` | 30 |
| Background | `fons_LaBr_maj_nmaj4_LG12_36.5V_10800s_merged.txt` | 10800 |
| ⁶⁰Co | `60Co_LaBr_maj_nmaj4_LG12_36.5V_900s_merged.txt` | 900 |
| ¹³⁷Cs | `137Cs_LaBr_maj_nmaj4_LG12_36.5V_cgain15_1200s_merged.txt` | 1200 |
| ²²Na | `22Na_LaBr_maj_nmaj4_LG12_36.5V_3900s_merged.txt` | 3900 |
| ¹³³Ba | `133Ba_LaBr_maj_nmaj4_LG12_36.5V_1500s_merged.txt` | 1500 |
| ²⁴¹Am | `241Am_LaBr_maj_nmaj4_LG12_36.5V_cgain15_1500s_merged.txt` | 1500 |

## Workflow

Run the notebooks from the same folder as `analysis.py`, since they load it with `import analysis`.

### 1. `energy_calib.ipynb`: calibration and resolution

1. **Pedestals**: a Gaussian is fitted to the pedestal of each of the 16 channels (HG and LG) and its position is stored.
2. **Net spectra**: for each source the pedestal is subtracted channel by channel, the 16 channels are summed event by event, and the background is subtracted after rescaling it by the live-time ratio `t_src / t_bg`.
3. **Photopeak fits**: a Gaussian on a linear (`gauss`) or parabolic (`gauss_parab`) background. The fit gives the centroid, σ, FWHM and Gaussian area with their uncertainties.
4. **Energy calibration**: linear fit `E = m·ADC + b` using seven reference lines, with a residuals panel in units of σ.
5. **Calibrated spectra**: saved as text files (`*_spec.txt` in keV and `*_spec_ADC.txt` in ADC channels) for each source and for the background.
6. **Energy resolution**: fit of `FWHM² = m_e·E + b_e`, which gives the model `R(E) = 100·√(m_e·E + b_e) / E`.

Calibration lines:

| Source | Energy (keV) |
|---|---|
| ²⁴¹Am | 59.5409 |
| ¹³³Ba | 80.9979, 356.0129 |
| ¹³⁷Cs | 661.655 |
| ⁶⁰Co | 1173.228, 1332.492 |
| ²²Na | 1274.537 |

### 2. `area_photopeak_plots.ipynb`: peak areas and efficiency

1. The pedestal is recomputed and the net spectra are built.
2. Each photopeak is fitted with a Gaussian, and its **net area** is obtained by subtracting a linear continuum drawn between the edges of the `[area_xl, area_xr]` window. Figures can be saved as PDF with an energy axis, using the calibration coefficients (`calib=[m, b]`).
3. **Current activity** of each source, from its reference activity, certification date and half-life, with uncertainty propagation.
4. **Absolute efficiency**: `ε_abs = N / (A · t · BR)`.
5. **Intrinsic efficiency**: `ε_int = ε_abs / G`, where `G` is the geometric factor for a square detector face of side `a = 1.2 cm` at an on-axis distance `d`:

   $$G = \frac{1}{\pi}\arcsin\left(\frac{a^2}{a^2 + 4d^2}\right)$$

6. Weighted fit `ln ε = a·(ln E)² + b·ln E + c` and comparison with Geant4 simulated values, with residuals in units of the experimental σ.

> The calibration coefficients are hard-coded in `area_photopeak_plots.ipynb`. If you redo the calibration, update them with the `m_ener` and `b_ener` values from `energy_calib.ipynb`.

## Main functions in `analysis.py`

| Function | What it does |
|---|---|
| `cargar_canales_separados(fitxer, asicid)` | Reads a data file and returns a list with the 16 channel vectors for the given ASIC. |
| `plot_acq_pedestal(...)` | Plots the pedestal spectra of the 16 channels, fits their positions and saves the combined pedestal-subtracted spectrum. |
| `fit_gaussiana(...)` | Builds the net spectrum (pedestal and background subtracted) and fits a photopeak between `xl` and `xr`. |
| `fit_gaussiana_and_area(...)` | Same as above, plus the net peak area, an optional energy-calibrated axis and figure saving. |
| `plot_espectres_bg(...)` | Compares the raw spectrum with the rescaled background and shows the net spectrum. |
| `energy_calib(adc, ener_teor, delta_adc, ...)` | Linear energy calibration with a residuals plot. |
| `energy_res(...)` | FWHM and resolution in ADC channels and in keV, and the FWHM² vs E fit. |
| `area_photopeak(...)` | Net area of a peak over a linear continuum, with its statistical uncertainty. |
| `energy_eff(...)` | Absolute and intrinsic efficiency with uncertainty propagation. |
| `gauss`, `gauss_parab`, `gauss_solo`, `recta` | Fit models. |
| `smooth`, `assign_markers`, `save_pdf` | Helpers for smoothing, plot markers and PDF export. |

## Outputs

- PDF figures: `energy_calib_plot.pdf`, `areaCo1173.pdf`, `areaCo1332.pdf`, `areaCs662.pdf`, `areaNa1275.pdf`, `areaBa356.pdf`, `areaBa81.pdf` and `areaAm60.pdf`.
- Text spectra: `Co_spec.txt`, `Na_spec.txt`, `Cs_spec.txt`, `Ba_spec.txt`, `Am_spec.txt` and `background_spec.txt`, each with an `*_ADC.txt` version, plus `HG_espectres_pedestal.txt`.

## Author

Pau Fité
