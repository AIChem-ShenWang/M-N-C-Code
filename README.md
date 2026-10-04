## Introduction

Code and data set for the paper "Data-driven Insight into M−N₄−C Stability: Guidance towards Large-scale Material Screening".

Runs on Python 3.9. `environment.yml` creates the conda environment `material`; `pkgs.txt` lists the same direct dependencies. `shap` and `TorchSisso` come from PyPI (see the file headers), and `stable table generator.py` also needs `pdftotext` from **poppler-utils**.

### `script`

Run from inside the folder, in the order below; each imports from `utils/` and writes its output beside itself or into `figures/`.

* `dataset generator.py`: builds the data set — VASP inputs from `data/template` and `data/5.4.4 VASP POTCAR`, then `data/M-N-C data set.xlsx`.
* `machine learning.py`: LOOCV comparison of six regressors, feature elimination, SHAP and SISSO for `E_b`, `E_f`, `U_diss`. `--redraw` redraws the figures from cached tables without repeating the search.
* `results plot.py`: periodic-table stability maps, descriptor heatmaps and per-target factor figures, read from the retained features of `* ML report.txt`.
* `structure analysis.py`: geometry–stability correlations (`angle threshold.png`, `stability vs geometry.png`) and the lanthanide bonding descriptors (`Lanthanide bonding.png`); `--lobster-setup` writes the LOBSTER inputs into the VASP directories, `--redraw` redraws from the CSV.
* `ablation study.py`: feature ablation (`.csv` tables and report).
* `stable table generator.py`: the stability map and its data sources — USGS 2026 price/production chapters, `mendeleev` supply risk and price, PubChem hazard classification. Writes `stable table.csv` and `data/stable M-N4-C report.txt`; `--refresh-usgs` re-reads the chapters.

### `utils`

* `vaspfile.py`: VASP file I/O and material descriptors.
* `doscar.py`: `DOSCAR` splitting and processing.
* `ml.py`: shared model list, metrics, symbol dictionary, LOOCV driver and plotting for the ML scripts.

### `data`

`5.4.4 VASP POTCAR` (PBE library), `template` (INCAR/KPOINTS/POSCAR), `atom-table` (reference data), `vasp-file` (the runs, `M`, `N-C`, `M-N-C/{opt,energy,dos}`; not shipped), `M-N-C data set.xlsx`.

### `figures`

Every figure, grouped by the script that produces it: `data analysis` and `dos`, `ML`, and `structure analysis`.

## Computational details

### Core parameters

| Parameter | Optimization | Energyᵍ | DOS | Isolated atom |
| :---: | :---: | :---: | :---: | :---: |
| `ISTART` | 0 | 1 | 1 | 0 |
| `ICHARG` | 2 | 1 | 11 | 2 |
| `ALGO`ᵃ | Normal | All | All | All |
| `NELM` | 80 | 500 | 200 / 1ᵇ | 200 |
| `ENCUT` | 520 eV | 520 eV | 520 eV | 520 eV |
| `EDIFF` | 1E-5 eV | 1E-5 eV | — | 1E-5 eV |
| `EDIFFG` | -0.03 eV/Å | — | — | — |
| `IBRION` | 2 | -1 | -1 | -1 |
| `NSW` | 300 | 0 | 0 | 0 |
| `ISIF` | 2 | 2 | — | 0 |
| `ISMEAR` / `SIGMA` | 0 / 0.1 eV | 0 / 0.1 eV | 0 / 0.05 eV | 0 / 0.05 eV |
| k-points | Γ-centered 2×2×1 | Γ-centered 5×5×1 | Γ-centered 5×5×1 | Γ-centered 1×1×1 |
| `IVDW` | 11 | 11 | 11 | — |
| `PREC` / `LASPH` / `ISYM` | Accurate / .TRUE. / 0 | Accurate / .TRUE. / 0 | Accurate / .TRUE. / 0 | Accurate / .TRUE. / 0 |
| `ISPIN`ᶜ | 2 | 2 | 2 | 2 |
| `MAGMOM` | `44*0.0 4*0.0 n`ᵈ | — | — | `n`ᵉ |
| `NUPDOWN` | — | — | — | `n`ᵉ |
| `LDAU`ᶠ | `.TRUE.` | `.TRUE.` | `.TRUE.` | `.TRUE.` |
| `LORBIT` / `NEDOS` | — | — | 11 / 2000 | — |

ᵃ `ALGO = All` from the relaxed `WAVECAR`/`CHGCAR`; exceptions Sm (`VeryFast`, cold, `AMIX`/`BMIX` set) and Yb (`Normal`). ᵇ `ICHARG` = 11 from the energy `CHGCAR`, so `NELM` = 1 for read-only restarts. ᶜ 1 for the closed-shell Be, Mg, Ca, Sr, Ba, Zn, Pd, Cd, Hg, Yb. ᵈ unpaired electrons in M−N₄−C; Ho = 4 (4f¹⁰ of Ho³⁺), none for Sm. ᵉ unpaired electrons of the free atom; Sm carries 6. ᶠ `LDAUTYPE = 2` for the 19 metals with a partly filled d or f shell: 3d Sc–Ni `2`/`4.0`/`0`, `LMAXMIX` 4; 4f Ce–Tm `3`/`5.0`/`0`, `LMAXMIX` 6. La, Yb, Lu carry none. ᵍ Sm's slab, DOS, atom and LOBSTER runs were all redone with the +U recipe.

### Metal pseudopotentials

VASP, PBE, `potpaw_PBE` (5.4.4) from `data/5.4.4 VASP POTCAR/PBE/potpaw_PBE`; C and N always standard. Potentials differing from the element name:

| Metal | POTCAR | Metal | POTCAR | Metal | POTCAR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Na | Na_pv | K | K_sv | Rb | Rb_sv |
| Cs | Cs_sv | Ca | Ca_sv | Sr | Sr_sv |
| Ba | Ba_sv | Sc | Sc_sv | Ti | Ti_sv |
| V | V_sv | Cr | Cr_pv | Mn | Mn_pv |
| Y | Y_sv | Zr | Zr_sv | Nb | Nb_sv |
| Mo | Mo_sv | Ru | Ru_pv | Rh | Rh_pv |
| Hf | Hf_pv | Ta | Ta_pv | W | W_pv |
| Ga | Ga_d | Ge | Ge_d | In | In_d |
| Sn | Sn_d | Tl | Tl_d | Pb | Pb_d |
| Bi | Bi_d | | | | |

All others use the element name.