# Open-Source Tools for Raman and Infrared
Python Jupyter notebooks for importing, processing, and analysing Raman &amp; FTIR spectra

Designed for ease of use by non-experts, but some Python coding experience is needed if you want to modify the scripts.
These notebooks use the Jupyter environment to run. To get Jupyter or access its documentation, go to https://jupyter.org

More details to follow

Current dependencies and minimum versions:
- numpy 1.19.5
- pandas 0.22.0
- matplotlib 2.1.2
- lmfit 1.0.3
- scipy 1.5.4
- scikit-learn 0.24.2


Key functions available for use:
- subtract_baseline() - fits the spectrum with a polynomial function then subtracts that from the spectrum, good for removing background signal
- subtract_reference() - fits the spectrum with a scaled reference spectrum then subtracts that from the spectrum, good for removing unwanted known signal
- detect_peaks() - finds peaks in the spectrum by looking for local maxima, good for a first pass at getting peak positions
- fit_peaks() - fits peaks in the spectrum using mathematical functions, good for measuring peak positions, height, and shape
- get_noise() - tells you the magnitude of background noise in the spectrum, across a given frequency range
- do_pca() - does Principal Component Analysis for dimensional reduction of multiple spectra
- prepare_data_for_classification() - collates spectra and prepares them for training/testing a machine-learning classification model to categorise spectra
- train_and_test_classification_model() - trains and tests a machine-learning classification model on collated data, reports efficacy of model at categorising spectra
