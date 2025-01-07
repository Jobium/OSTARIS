"""
# ==================================================

This template script contains all the code needed to import and process Raman spectra, then save them in a standardised file format. Processing includes:
- Averaging for multi-spec measurements
- Baseline (background) subtraction
- Reference spectrum subtraction

Requires input data files to be organised as follows:
- Location: {Data_dir}/{Sample}/{Measurement Date}/
- Sample/Measurement metadata must be recorded in either file {spec filename}_MetaData.csv, or within spec filename
    - Spec filename format:
    
        {Spec ID number}_{Sample}_{Subsample}_{notes}_{Magnification}x_{Laser wavelength}nm_{Laser power}mW_{Accumulations}x{Exposure time}.txt
    
    - Each spectrum file in a given project must have a unique identifier (preferable a sequential ID number) so that it can always be distinguished from other measurements, even when they are otherwise identical. These ID numbers can be used to refer to particular spectra when manually specifying settings to use for outliers/exceptional cases.
    - any pre-processing steps applied to the data should be included at the end of the filename
        - C: cosmic ray removal
        - O: outlier removal
        - N: normalisation
        
How to use this script:
1) 

# ==================================================
"""

# ==================================================
# import python modules

import os
import math
import glob
import datetime
import numpy as np
import pandas as pd
import lmfit as lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

from functions_v01 import *

# ==================================================
# set initial variables and enable processing steps

Technique = 'Raman'         # 'Raman' or 'FTIR'

# filter data import by sample / subsample
Sample = '*'                # name of sample, or '*' for all
Subsample = '*'             # name of subsample, or '*'
Measurement_ID = '*'        # unique ID of measurement, or '*'
Measurement_Date = '*'      # Date in YYYY-MM-DD format as string, or '*'

# list spectral IDs to skip when importing data, as strings
Do_Not_Import = []

# filter by measurement settings
Metadata_File = False       # set to True to import metadata file, or False to extract data from spec filename
Laser_Wavelength = 785      # wavelength (in nm) as integer, or '*'
Laser_Power = '*'           # power (in mW or %) as int/float, or '*'
Exposure_Time = '*'         # exposure time (in sec) as int/float, or '*'
Accumulations = '*'         # accumulations as int, or '*'
Magnification = '*'         # objective magnification as int, or '*'
Preprocessing = '*'         # specify required preprocessing, or '*' for best available

# processes to run (see each section for more details)
Baseline_Subtraction = True     # fit and subtract baseline
Subtract_Reference = True      # scale and subtract reference spectra from specified spectra

# list directories for input data, figures and output files
Data_dir = '../test data/'
Fig_dir = '../figures/'
Out_dir = '../output/'

# Colourblind-friendly palette developed by Paul Tol ( https://personal.sron.nl/~pault/#sec:qualitative )
Color_list_dark = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
Color_list_light = ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD']
Color_list = Color_list_dark + Color_list_light

"""
# ==================================================
# FILE SEARCH
# - this section searches for spectrum files that match the specified settings
# ==================================================
"""

print()
print()
print("SEARCHING FOR SPECTRUM DATA FILES...")

# find files
#   - requires files to be Data_dir/{Sample}/{mMasurement_Date}/{Measurement_ID}_{Sample}_{Subsample}_{Measurement Parameters}.txt
text = "%s%s/%s/%s_%s_%s*_%sX_%snm_%smW_%sx%ss*txt" % (Data_dir, Sample, Measurement_Date, Measurement_ID, Sample, Subsample, Magnification, Laser_Wavelength, Laser_Power, Accumulations, Exposure_Time)
spec_dirs = sorted(glob.glob(text))

# trim to measurement ID numbers
Spec_IDs = np.unique([spec.split("/")[-1].split("_")[0] for spec in spec_dirs])
print()
print("spectrum IDs found:", Spec_IDs)

# find appropriate pre-processed file for each measurement
spec_dirs = []
for ID in Spec_IDs:
    if Preprocessing != '*':
        # use specified preprocessing
        text = "%s%s/%s/%s_*_%s.txt" % (Data_dir, Sample, Measurement_Date, ID, Preprocessing)
        temp = sorted(glob.glob(text))
        if len(temp) > 0:
            lengths = [len(s) for s in temp]
            spec_dirs.append(temp[np.argmax(lengths)])
    else:
        # use best available
        text = "%s%s/%s/%s_*.txt" % (Data_dir, Sample, Measurement_Date, ID)
        temp = sorted(glob.glob(text))
        if len(temp) > 0:
            lengths = [len(s) for s in temp]
            spec_dirs.append(temp[np.argmax(lengths)])

print()
print("data files found:", len(spec_dirs))
for file in spec_dirs:
    print("    ", file.split("/")[-1])
            
"""
# ==================================================
# DATA IMPORT
# - this section actually imports data files and extracts their metadata
# - metadata can be in a separate CSV file with the same name + "_metadata", or written into the spec filename
# - measurements are imported file by file, and added sequentially to the data storage dict 'data'
# - to access a specific measurement, you call data[measurement ID][key], where 'key' is the type of data you want from it
# - files that cannot be completely imported will not be added to the storage array, if a file doesn't get processed that usually means there's something wrong with the filename which means it got filtered out or is missing some key info
# ==================================================
"""

print()
print()
print("IMPORTING MEASUREMENTS...")

# set up data storage dictionary
data = {}

# ==================================================
# each measurement imported will be added to this dictionary as a Measurement object. To access a particular measurement, use data[measurement ID].

# for each detected file
for spec_dir in spec_dirs:
    while True:
        try:
            filename = spec_dir.split("/")[-1][:-4]
            ID = filename.split("_")[0]
            if ID in Do_Not_Import:
                print()
                print(" measurement %s is in Do_Not_Import list, skipping" % filename)
                break
            else:
                print()
                print("importing %s" % filename)
                # extract sample/measurement metadata
                metadata = False
                if Metadata_File == True:
                    # search for metadata file with matching name
                    metadata_dir = glob.glob("%s_metadata.csv" % spec_dir[:-4])
                    print("    metadata files found:", len(metadata_dir))
                    if len(metadata_dir) > 0:
                        metadata = True
                if metadata == True:
                    # get metadata from metadata file
                    metadata = pd.read_csv(metadata_dir)
                else:
                    # get metadata from filename instead
                    date = datetime.datetime.strptime(spec_dir.split("/")[-2], "%Y-%m-%d")
                    filename_split = filename.split("_")
                    # get sample info
                    ID, sample, subsample = filename_split[:3]
                    # check for preprocessing
                    if len(filename_split[-1].split("x")) == 2 and filename_split[-1][-1] == 's':
                        # final item in filename is {Accumulations}x{Exposure_Time}s, no preprocessing
                        preprocessing = 'none'
                        # check for additional sample notes
                        notes = ''
                        if filename_split[2] != filename_split[-5]:
                            notes = "_".join(filename_split[3:-4])
                        mag, laser_wavelength, laser_power, accumxexp = filename_split[-4:]
                    else:
                        # final item in filename assumed to be preprocessing steps
                        preprocessing = filename_split[-1]
                        # check for additional sample notes
                        notes = ''
                        if filename_split[2] != filename_split[-6]:
                            notes = "_".join(filename_split[3:-5])
                            metadata_start = 4
                        mag, laser_wavelength, laser_power, accumxexp = filename_split[-5:-1]
                    mag = mag[:-1]  # remove 'X' from magnification
                    laser_wavelength = int(laser_wavelength[:-2])   # remove 'nm'
                    if laser_power[-2:] == 'mW':
                        # laser power in mW
                        laser_power = float(laser_power[:-2])   # remove 'mW'
                        power_unit = 'mW'
                    else:
                        # assume power in %
                        laser_power = float(laser_power[:-1])   # remove '%'
                        power_unit = "%%"
                    accumulations, exposure_time = accumxexp.split("x")
                    accumulations = int(accumulations)
                    exposure_time = float(exposure_time[:-1])   # remove 's'
                # report metadata
                print("    measurement ID:", ID)
                print("              sample:", sample)
                print("           subsample:", subsample)
                print("         measured on:", date.strftime("%Y-%m-%d"))
                print("               notes:", notes)
                print("    measurement settings:")
                print("         magnification: %s X" % (mag))
                print("            wavelength: %s nm" % (laser_wavelength))
                print("                 power: %s %s" % (laser_power, power_unit))
                print("              exposure: %s x %s seconds" % (accumulations, exposure_time))
                print("    preprocessing: %s" % (preprocessing))
                # import spectrum file (assumes Renishaw file formatting with either 2 or 4 columns)
                spec = np.genfromtxt(spec_dir).transpose()
                print("    spec array:", np.shape(spec))
                distances = []
                xy_coords = []
                if np.size(spec, axis=0) == 4:
                    # map or line, columns=(x_position, y_position, raman_shift, intensity)
                    spec_type = 'map'
                    # determine splits between stacked spectra
                    splits = np.ravel(np.where(np.roll(spec[2], 1) < spec[2]))
                    points = len(splits)
                    print("        spectral map, %s points" % (points))
                    # get X,Y coordinates (in um) for each point spectrum
                    x_pos = np.asarray(spec[0][splits])
                    y_pos = np.asarray(spec[1][splits])
                    xy_coords = np.asarray([x_pos, y_pos])
                    # determine if line, or map by looking at signs of dX, dY
                    dx = np.roll(x_pos, 1)[1:] - x_pos[1:]
                    dy = np.roll(y_pos, 1)[1:] - y_pos[1:]
                    dx_sign = np.sign(dx)
                    dy_sign = np.sign(dy)
                    if np.all(dx_sign == np.sign(np.mean(dx))) & np.all(dy_sign == np.sign(np.mean(dy))):
                        print("    measurement is a 1D line scan")
                        spec_type = 'line'
                        distances = np.cumsum(np.sqrt(dx**2 + dy**2))
                        distances = np.insert(distances, 0, 0)
                    else:
                        print("    measurement is a 2D map scan")
                    # extract shift, intensity spectrum for each point
                    x = np.asarray(np.split(spec[2], splits[1:]))[0]
                    y = np.asarray(np.split(spec[3], splits[1:]))
                else:
                    # single point, columns=(raman_shift, intensity)
                    spec_type = 'point'
                    points = 1
                    xy_coords = np.asarray([[0],[0]])
                    print("        single point measurement")
                    x = spec[0]
                    y = spec[1:]
                sort = np.argsort(x)
                x = x[sort]
                y = y.transpose()[sort,:]
                print("              shift:", np.shape(x))
                print("          intensity:", np.shape(y))
                print("        shift range: %0.f - %0.f cm-1" % (np.amin(x), np.amax(x)))
                print("          inf check:", np.any(np.isinf(y)))
                print("          nan check:", np.any(np.isnan(y)))
                # get average spectrum (for single points, just use spec)
                y_av = np.mean(y, axis=0)
                # generate sample title for consistent naming
                title = "%s_%s_%s" % (ID, sample, subsample)
                if notes != '':
                    title += "_" + notes
                # create Measurement instance from imported data
                data[str(ID)] = Measurement(
                    ID=str(ID),
                    title=title,
                    filename=filename,
                    sample=sample,
                    subsample=subsample,
                    notes=notes,
                    x=x,
                    y=y,
                    technique='Raman',
                    laser_wavelength=str(laser_wavelength),
                    spec_type=spec_type,
                    points=points,
                    magnification=mag,
                    laser_power=laser_power,
                    accumulations=accumulations,
                    exposure_time=exposure_time,
                    x_coords=xy_coords[0],
                    y_coords=xy_coords[1],
                    Fig_dir = Fig_dir,
                    Out_dir = Out_dir
                )
                
                print("    imported successfully!")
                break
        except Exception as e:
            print("    something went wrong! Exception:", e)
            break
        
print()
print("%s/%s files imported" % (len(data.keys()), len(spec_dirs)))

# report which files were imported and which were not
print()
for num in Spec_IDs:
    if num in data.keys():
        print(num, u'\u2713', data[num]['filename'])
    else:
        text = ''
        if num in Do_Not_Import:
            text = 'ID in Do_Not_Import list'
        print(num, 'X', text)

# update list of Spec IDs to only include imported spectra
Spec_IDs = list(data.keys())

samples = np.unique([data[num]['sample'] for num in Spec_IDs])
print()
print("samples in dataset:", samples)

lasers = np.unique([measurement.laser_wavelength for ID, measurement in data.items()])
print()
print("laser wavelengths in dataset:", lasers)
        
"""
# ==================================================
# BASELINE SUBTRACTION
# - baseline is a polynomial fitted to specified points along the x axis
# - you can specify the points and polynomial order used for each wavelength, and add additional logic for specific samples if needed
# - subtract_baseline() function finds the local average for each point ±25 cm-1, this value can be changed using the window argument
# - automatically baselines each spectrum in a multi-spec measurement separately, individual spectra are stored in y_sub[measurement index][spec index]
# - y_av_sub is the baselined average spectrum
# - y_sub_av is the average of the baselined spectra
# ==================================================
"""

# set whether to print debug messages in this section
debug = True

if Baseline_Subtraction == True:
    print()
    print()
    print("DOING BASELINE SUBTRACTION")

    if len(Spec_IDs) == 0:
        print("    no spectra imported, skipping")

    for ID, measurement in data.items():
        title = measurement.title
        print()
        print(ID, title)
        sample = measurement.sample

        # decide which set of baseline points to use
        if measurement.laser_wavelength == '532':
            base_list = [100, 200, 300, 400, 415, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1600, 1700, 1800]
            order = 11
        elif measurement.laser_wavelength in ['633', '638']:
            base_list = [100, 200, 300, 400, 415, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1500]
            order = 21
        elif measurement.laser_wavelength == '785':
            if ID in ['R0001', 'R0002']:
                # these spectra earmarked for having glass slide PL peak, requires reference subtraction 
                base_list = [250, 475, 525, 660, 790, 900, 1000, 2100, 2250, 2500, 2600, 2900, 3100, 3500, 3600, 3800]
                order = 11
            else:
                base_list = [250, 475, 525, 660, 790, 900, 1200, 1300, 1700, 1900, 2000, 2250, 2500, 2600, 2900, 3100, 3500, 3600, 3800]
                order = 11
        else:
            base_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1500]
            order = 11
        base_list = np.asarray(base_list)

        # subtract baseline from average spectrum
        subtract_baseline(measurement, 'raman_shift', 'y_av', 'y_av_sub', base_list, base='polynomial', order=order, fixed_ends=True, plot=True, show_plot=True, debug=debug)

        # baseline individual spectra if necessary
        if measurement.points > 1:
            if debug == True:
                print("%s points:" % title, measurement.points)
                print("    map x,y arrays:", np.shape(measurement['raman_shift']), np.shape(measurement['y']))
            # subtract baseline from individual spectra
            subtract_baseline(measurement, 'raman_shift', 'y', 'y_sub', base_list, base='poly', order=order, fixed_ends=True, show_plot=False, debug=debug)

            # plot average of baselines vs baseline of average
            plt.figure(figsize=(8,4))
            plt.title("%s:\nAverage of %s baselined spectra" % (title, measurement.points))
            ### need to add st.dev. handling to make this work
            ### plt.fill_between(spec['raman_shift'], np.mean(spec['y_sub'], axis=0)-np.std(spec['y_sub'], axis=0), np.mean(spec['y_sub'], axis=0)+np.std(spec['y_sub'], axis=0), color='k', alpha=0.1, linewidth=0.)
            print(np.shape(measurement['y_sub'].y))
            plt.plot(measurement['raman_shift'], measurement['y_sub'].mean(), 'k')
            plt.xlim(measurement.x_start, measurement.x_end)
            plt.show()

            # get average of baselined spectra
            measurement.add_spectrum(key='y_sub_av', y=measurement['y_sub'].mean(), label='Averaged Baselined Intensity (counts)', log=measurement['y_sub'].log + ['averaged over %s spectra' % measurement.points])

"""
# ==================================================
# REFERENCE SPECTRUM SUBTRACTION
# - for specified spectra, subtract named reference spectrum
# - works by interpolating reference spectrum to match target spectrum, rescaling reference spectrum to fit target spectrum at key positions, then subtracting reference from target
# - requires processed reference spectrum to be in ./{Output_dir}/
# - y_av_rsub is the baselined average spectrum
# - y_sub_av is the average of the baselined spectra
# ==================================================
"""

# for each reference spectrum being used, add its spec ID to this dict as follows:
# ref ID: {
#   'name': 'blah', # name of reference,
#   'targets': [],  # list of spec IDs to subtract this reference from,
#   'x_list': []    # list of x positions to use when rescaling reference to match y values of target,
# }
subtraction_refs = {
    'R0009': { # this is for glass PL peak (785nm only)
        'name': 'glass',
        'targets': ['0004', '0007', '0012', '0014', '0015', '0016', '0017', '0018', '0019', '0023', '0024', '0025', '0026', '0028', '0029', '0030', '0031', 'R0001', 'R0002', 'R0003', 'R0004', 'R0005', 'R0008'],
        'x_list': [1200, 1300, 1380, 1480, 1650, 1750, 1900]
    }
}

# set whether to print debug messages in this section
debug = True

# set whether to rerun baseline subtraction after subtracting reference
rerun_baseline = True

# data storage array for reference spectra
ref_data = {}

# import processed reference spectra
if Subtract_Reference == True:
    print()
    print()
    print("DOING REFERENCE SPECTRUM SUBTRACTION")
    
    # check which references need to be imported
    refs_for_import = []
    for ref, ref_info in subtraction_refs.items():
        targets = ref_info['targets']
        check = np.any(np.asarray([target in Spec_IDs for target in targets]))
        if check == True:
            refs_for_import.append(ref)
            
    # import processed reference spectra
    if len(refs_for_import) == 0:
        # no references found for import
        print()
        print("no references specified!")
    else:
        # proceed with data import
        
        for ref_ID in refs_for_import:
            # get info from dict
            ref_name = subtraction_refs[ref_ID]['name']
            targets = [target for target in subtraction_refs[ref_ID]['targets'] if target in Spec_IDs]
            x_list = subtraction_refs[ref_ID]['x_list']
            print("reference:", ref_ID, ref_name)
            print("    to be subtracted from:", targets)
            print("    x_list for fitting:", x_list)
            # check if ref is in imported data
            success = False
            if ref_ID in Spec_IDs:
                # use currently-imported spectrum if possible
                if hasattr(data[str(ref_ID)], 'y_av_sub'):
                    print("    using currently-imported data for %s" % ref_ID)
                    # add link to data storage array
                    ref_data[str(ref_ID)] = data[str(ref_ID)]
                    ref_data[str(ref_ID)].name = ref_name
                    ref_data[str(ref_ID)].targets = targets
                    ref_data[str(ref_ID)].x_list = x_list
                    success = True
            if success == False:
                # find output spectrum file instead
                print("    searching for processed data file...")
                spec_dirs = glob.glob("%s*/*/*/%s_*_av-spectrum.csv" % (Out_dir, ref_ID))
                print("        files found:", len(spec_dirs))

                if len(spec_dirs) > 0:
                    # import spectrum file
                    spec = pd.read_csv(spec_dirs[0])
                    print("    imported spec array:", np.shape(spec))
                    print(spec.info())
                    filename_split = spec_dirs[0].split("/")[-1][:-4].split("_")
                
                    # add to ref_data storage array
                    ref_data[str(ref_ID)] = Measurement(
                        ID = ref_ID,
                        title = "_".join(filename_split[:3]),
                        sample = title[1],
                        subsample = title[2],
                        name = ref_name,
                        targets = targets,
                        x_list = x_list,
                        laser_wavelength = filename_split[-4][:-2],
                        x_type = 'raman_shift',
                        x = spec['Raman Shift (cm-1)'],
                        y = spec['Raw Intensity'],
                        Fig_dir = Fig_dir,
                        Out_dir = Out_dir
                    )
                    ref_data[str(ref_ID)].add_spectrum('y_av_sub', spec['Baselined Intensity'], label='Baselined Intensity (counts)', log=['imported from file', 'baselined'])
            
    # proceed with subtraction, ref by ref
    for ref_ID, ref in ref_data.items():
        # get ref info from dict
        ref_name = ref.name
        targets = ref.targets
        x_list = ref.x_list
        print()
        print("subtracting %s reference %s..." % (ref_name, ref_ID))
        
        # for each measurement in target list
        for target_ID in targets:
            measurement = data[target_ID]
            print()
            print("    doing subtraction for %s %s" % (target_ID, measurement.title))
            subtract_reference(ref, measurement, 'raman_shift', 'y_av_sub', 'y_av_sub_refsub', x_list, plot=True, show_plot=True, debug=debug)
            
            if rerun_baseline == True:
                # decide which set of baseline points to use
                if measurement.laser_wavelength == '532':
                    base_list = [100, 200, 300, 400, 415, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1500]
                    order = 11
                elif measurement.laser_wavelength in ['633', '638']:
                    base_list = [100, 200, 300, 400, 415, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1500]
                    order = 21
                elif measurement.laser_wavelength == '785':
                    base_list = [250, 475, 525, 660, 790, 900, 1200, 1300, 1380, 1720, 1900, 2000, 2250, 2500, 2600, 2800, 3050, 3500, 3600, 3800]
                    order = 11
                else:
                    base_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1050, 1150, 1400, 1500]
                    order = 11
                base_list = np.asarray(base_list)
                
                # subtract baseline from average spectrum
                subtract_baseline(measurement, 'raman_shift', 'y_av_sub_refsub', 'y_av_sub_refsub_sub', base_list, base='poly', order=order, fixed_ends=True, plot=True, show_plot=True, plot_name='baselined2', debug=debug)
    
"""
# ==================================================
# SAVE PROCESSED SPECTRA
- for all measurements, the average spectrum is saved to _av-spectrum.csv
    - includes columns for raw intensity, baselined intensity, and normalised values
- for multi-spec measurements (e.g. maps and line-scans), all point-spectra are saved to _all-spectra_baselined.csv
    - each spectrum is saved as a column along with its X,Y coordinates
# ==================================================
"""

print()
print()
print("SAVING PROCESSED SPECTRA TO OUTPUT FOLDER")

# set whether to print debug messages in this section
debug = True

# set whether to output metadata files
Save_metadata = True

if len(Spec_IDs) == 0:
    print("    no spectra imported, skipping")

for ID, measurement in data.items():
    title = measurement.title
    print()
    print(title)
    
    # one column for each modification of spectrum
    headers = ['Wavelength (nm)', 'Raman Shift (cm-1)', 'Raw Intensity', 'Normalised Intensity', 'Baselined Intensity', 'Normalised Baselined Intensity']
    keys = ['wavelength', 'raman_shift', 'y_av', 'y_av_norm', 'y_av_sub', 'y_av_sub_norm']
    save_measurement(measurement, keys=keys, headers=headers, save_name='av-spectrum', debug=debug)
    
    # save all point-spectra to file (maps & multi-spec files only)
    if measurement.points > 1 and hasattr(measurement, 'y_sub'):
        # one column per baselined point-spectrum
        headers = ['Wavelength (nm)', 'Raman Shift (cm-1)', 'Baselined Intensity']
        keys = ['wavelength', 'raman_shift', 'y_sub']
        save_measurement(measurement, keys=keys, headers=headers, save_name='all-spectra-baselined', debug=debug)
    print("    %s saved!" % ID)

"""
# ==================================================
# end of script
# ==================================================
"""

print()
print()
print("DONE")