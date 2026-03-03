This project is the code for the paper "Machine Learning insight into the stability of M−N4−C: Guidance towards High-throughput Computation", including the dataset used and the code for dataset construction, processing & analysis, and ML technique.

The project uses Python to build and train the model. The required Python version and Python libraries are listed in the pkgs.txt file.

The project consists of 4 folders: code, data, figures, utils. Each of which is described below.

* code folder: 

  * data generator.py: The Python coder for VASP input file construction and reading of VASP output files to construct the data set.

  * machine learning.py: The codes for model training & testing, with SHAP for feature importance analysis and SISSO prediction.

  * data analysis.py: The Python codes analysis of data distribution, summary of machine learning results, and fitting $E_{f}$ with $CM1$.

* data folder:

  * 5.4.4 VASP POTCAR: Atomic pseudopotential library for constructing the input files required by the VSAP software.

  * atom-table folder: The reference data used in this work.

  * template folder: A template for generating the POSCAR file in VASP input files.

  * vasp-file folder: The directory for VASP files. Due to the large number of files, they are not provided on this website but can be obtained by contacting the corresponding author.

  * M-N-C data set.xlsx: The $M-N_{4}-C$ data set.


* figures folder:

    The figures can be obtained by running Python files from code folder.

* utils folder: A toolkit for convenient file conversion and material feature generation.

  * vaspfile.py: Functions related to handling VASP files and processing material information.

  * doscar.py: Functions to process DOSCAR file. Ref: https://vasppy.readthedocs.io/en/latest/_modules/vasppy/doscar.html.

