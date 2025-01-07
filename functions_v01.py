"""
# ==================================================

This script contains all functions to be used by other scripts.

# ==================================================
"""

import os
import numpy as np
import pandas as pd
import lmfit as lmfit
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

# Colourblind-friendly palette developed by Paul Tol ( https://personal.sron.nl/~pault/#sec:qualitative )
Color_list_dark = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
Color_list_light = ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD']
Color_list = Color_list_dark + Color_list_light

"""
functions to add:
1) outlier detection
2) basic spectral plot generator
3) map generator
"""



"""
# ==================================================
# classes for consistently storing and handling measurement, spectrum data
# ==================================================
"""

class Measurement:
    # class containing all recorded data for a measurement object
    def __init__(self, **kwargs):
        # method called when creating instance of measurement class
        # check that essential parameters included in kwargs
        check = ['ID', 'title', 'sample', 'technique', 'x', 'y', 'Fig_dir', 'Out_dir']
        check = [s for s in check if s not in kwargs.keys()]
        if 'technique' in kwargs.keys():
            if kwargs['technique'] == 'Raman' and 'laser_wavelength' not in kwargs.keys():
                raise Exception("Error! Cannot create a Measurement instance for a Raman spectrum without specifying laser_wavelength!")
        if len(check) > 0:
            raise Exception("Error! Cannot create a Measurement instance without the following arguments: %s" % ",".join(check))
        del check
        # add parameters to instance
        for k in kwargs:
            # dynamically add properties based on what was passed to measurement()
            setattr(self, k, kwargs[k])
            
        # handle input x properties
        self.x_start = np.amin(self.x)
        self.x_end = np.amax(self.x)
        if kwargs['technique'].lower() in ['f', 'fluor', 'fluorescence', 'p', 'pl', 'photolum', 'photoluminescence']:
            # fluorescence, wavelength domain
            self.technique = 'Photoluminescence'
            self.xlabel = 'Wavelength (nm)'
            self.frequency = wavelength2frequency(kwargs['x'])
            self.wavelength = kwargs['x']
        elif kwargs['technique'].lower() in ['r', 'raman']:
            # Raman measurements, raman shift domain
            self.technique = 'Raman'
            self.xlabel = 'Raman Shift (cm$^{-1}$)'
            self.raman_shift = kwargs['x']
            self.wavelength = shift2wavelength(kwargs['x'], kwargs['laser_wavelength'])
        else:
            # default to Infrared, frequency domain
            self.technique = 'Infrared'
            self.xlabel = 'Frequency (cm$^{-1}$)'
            self.frequency = x
            self.wavelength = frequency2wavelength(kwargs['x'])
            
        # check for info on nature of input spectrum
        if 'ykey' in kwargs:
            ykey = kwargs['ykey']
            del self.ykey
        else:
            ykey = 'y'
        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
            del self.ylabel
        else:
            ylabel = 'Intensity (counts)'
        if 'log' in kwargs:
            log = kwargs['log']
            del self.log
        else:
            log = ['imported from file']
        # convert input y into a Spectrum instance
        self.add_spectrum(key=ykey, y=kwargs['y'], label=ylabel, log=log, debug=False)
        if kwargs['y'].ndim > 1 and ykey == 'y':
            # add Spectrum instance for average spectrum as well
            self.add_spectrum('y_av', np.mean(kwargs['y'], axis=1), label='Average Intensity (counts)', log=['imported from file', 'averaged %s spectra' % np.shape(kwargs['y'])[1]])
        
        # create figure, output directories for this measurement
        self.fig_dir = '%s%s/%snm/%s/' % (kwargs['Fig_dir'], self.sample, self.laser_wavelength, self.title)
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.out_dir = '%s%s/%snm/%s/' % (kwargs['Out_dir'], self.sample, self.laser_wavelength, self.title)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        # remove unneeded parameters
        delattr(self, 'Fig_dir')
        delattr(self, 'Out_dir')
        
    def __getitem__(self, item):
        # method called when subscripting the instance, e.g. measurement1[property]
        return getattr(self, item)  # return specified property
        
    def __call__(self, i, j):
        # method called when referencing the instance like a function, e.g. measurement1(x_key, y_key)
        if j[-5:] == '_norm' and hasattr(self, j) == False and hasattr(self, j[:-5]) == True:
            # return specified x and normalised y properties
            j = j[:-5]
            if hasattr(self[j], i):
                return self[j][i], self[j].norm()
            else:
                return self[i], self[j].norm()
        else:
            # return specified x and y properties
            if hasattr(self[j], i):
                return self[j][i], self[j].norm()
            else:
                return self[i], self[j].y
    
    def get_plot_data(self, ykey='y', xkey='x', start=None, end=None):
        # returns all necessary information for plotting
        if start == None:
            start = self.start
        if end == None:
            end = self.end
        if hasattr(self[ykey], xkey):
            x = getattr(self[ykey], xkey, self.x)
        else:
            x = getattr(self, xkey, self.x)
        sliced = np.where((start <= x) & (x <= end))
        return x[sliced], self.y[sliced], self.label
        
    def add_spectrum(self, key, y=None, label=None, x=None, log=[], indices=None, debug=False):
        # function for creating a new spectrum instance inside measurement instance
        if hasattr(self, str(key)) == True and debug == True:
            print("CAUTION: overwriting %s" % str(key))
        parent = self
        
        class Spectrum:
            # class containing all relevant info for a spectrum
            def __init__(self, key=key, y=y, label=label, log=log, indices=indices, debug=debug):
                # method called when creating instance of spectrum class
                self.key = key                  # key for referencing this spectrum (string)
                self.parent = parent            # parent measurement instance
                self.label = label              # label for y axis (string)
                self.log = log                  # records all processing steps applied so far (list of strings)
                y = np.asarray(y)
                if y.ndim == 1:
                    # received a single spectrum in 1D array
                    self.type = 'single'
                    self.y = y[:,np.newaxis]    # y values forced into 2D array
                elif np.shape(y)[1] == 1:
                    # received a single spectrum in 2D array
                    self.type = 'single'
                    self.y = y                  # y values as 2D array
                else:
                    # received multiple spectra in 2D array
                    self.type = 'multi'
                    self.y = y                  # y values as 2D array
                if np.all(np.asarray(indices) != None):
                    if len(indices) != np.shape(y)[1]:
                        raise Exception("add_spectrum() method must receive index list matching length of y array!")
                    else:
                        self.indices = indices      # track indices of spectra in measurement, 0-indexed
                else:
                    self.indices = range(np.shape(self.y)[1])
                if np.all(x != None):
                    self.x = x

            def __iter__(self):
                # returns spectrum as x,y tuple
                yield parent.x
                yield self.y

            def get_plot_data(self, xlabel='x', start=parent.x_start, end=parent.x_end):
                # returns all necessary information for plotting a spectrum (x values, y values, y axis label)
                x = getattr(parent, xlabel, parent.x)
                sliced = np.ravel(np.where((start <= x) & (x <= end)))
                return x[sliced], self.y[sliced], self.label
                
            # class methods for handling spectra
            def start(self):
                # get first x value
                return parent.x_start
            def end(self):
                # get last x value
                return parent.x_end
            def y_min(self, start=parent.x_start, end=parent.x_end):
                # get lowest y value in range
                sliced = np.where((start <= parent.x) & (parent.x <= end))
                return np.amin(self.y[sliced])
            def ymax(self, start=parent.x_start, end=parent.x_end):
                # get highest y value in range
                sliced = np.where((start <= parent.x) & (parent.x <= end))
                return np.amax(self.y[sliced])
            def norm(self, amax=None, amin=None, start=parent.x_start, end=parent.x_end):
                # get normalised spectrum
                if np.all(amax == None):
                    amax = np.amax(self.y)
                if np.all(amin == None):
                    amin = np.amin(self.y)
                sliced = np.where((start <= parent.x) & (parent.x <= end))
                return (self.y[sliced] - amin) / (amax - amin)
            def mean(self, start=parent.x_start, end=parent.x_end):
                # get mean spectrum (or mean value)
                sliced = np.where((start <= parent.x) & (parent.x <= end))
                return np.mean(self.y[sliced], axis=1)
            def median(self, start=parent.x_start, end=parent.x_end):
                # get median spectrum (or median value)
                sliced = np.where((start <= parent.x) & (parent.x <= end))
                return np.median(self.y[sliced], axis=1)
            
        # finish add_spectrum() method by adding Spectrum instance to parent Measurement instance
        setattr(self, str(key), Spectrum(key, y, label=label, log=log, debug=debug))
        if debug == True:
            print('    end of add_spectrum method')
    
    # end of Measurement class

"""
# ==================================================
# functions for converting between raman shift, wavelength, frequency, and energy
# ==================================================
"""

def wavelength2shift(wavelength, excitation=785.):
    # convert wavelength (in nm) to raman shift (in cm-1)
    if type(excitation) in [str, int, np.str_]:
        excitation = float(excitation)
    shift = ((1./excitation) - (1./wavelength)) * (10**7)
    return shift

def shift2wavelength(shift, excitation=785.):
    # convert raman shift (in cm-1) to wavelength (in nm)
    if type(excitation) in [str, int, np.str_]:
        excitation = float(excitation)
    wavelength = 1./((1./excitation) - shift/(10**7))
    return wavelength

def wavelength2frequency(wavelength, excitation=785.):
    # convert wavelength (in nm) to absolute frequency (in cm-1)
    if type(excitation) in [str, int, np.str_]:
        excitation = float(excitation)
    return (1./wavelength) * (10**7)

def frequency2wavelength(frequency, excitation=785.):
    # convert absolute frequency (in cm-1) to wavelength (in nm)
    if type(excitation) in [str, int, np.str_]:
        excitation = float(excitation)
    return (1./frequency) / (10**7)

def shift2frequency(shift, excitation=785.):
    # convert raman shift (in cm-1) to absolute frequency (in cm-1)
    if type(excitation) in [str, int, np.str_]:
        excitation = float(excitation)
    wavelength = 1./((1./excitation) - shift/(10**7))
    return wavelength2frequency(wavelength)
    
def frequency2shift(frequency, excitation=785.):
    # convert absolute frequency (in cm-1) to raman shift (in cm-1)
    if type(excitation) in [str, int, np.str_]:
        excitation = float(excitation)
    wavelength = frequency2wavelength(frequency)
    shift = ((1./excitation) - (1./wavelength)) * (10**7)
    return shift

def wavelength2energy(wavelength):
    # convert wavelength (in nm) to energy (in J)
    hc = 1.98644586*10**-25 # in J meters
    return hc / (np.asarray(wavelength)/10**9)

def energy2wavelength(energy):
    # convert photon energy (in J) to wavelength (in nm)
    hc = 1.98644586*10**-25 # in J meters
    return (hc / np.asarray(wavelength)) * 10**9

def photon_count(wavelength, energy):
    # convert a total laser energy (in J) at given wavelength (in nm) to photon count
    hc = 1.98644586*10**-25 # in J meters
    photon_energy = hc / (np.asarray(wavelength)/10**9)
    return np.asarray(energy) / photon_energy

def intensity2snr(intensity, noise):
    return intensity / noise

def snr2intensity(snr, noise):
    return snr * noise

"""
# ==================================================
# functions for smoothing, trimming and normalising spectra
# ==================================================
"""

def smooth_spectrum(y, window_length, polyorder):
    # function for smoothing data based on Savitsky-Golay filtering
    if window_length % 2 != 1:
        window_length += 1
    y_smooth = savgol_filter(y, window_length, polyorder)
    return y_smooth

def slice_spectrum(x, y, std=None, start=400, end=4000, norm=False, inset=0, debug=False):
    # function for slicing a spectrum to a specific range and normalising if needed
    # argument summary:
    #   - x:        x values (1D or 2D array)
    #   - y:        y values (1D or 2D array)
    #   - std:      st. dev. of y (optional, 1D or 2D array)
    #   - start:    start of x axis range to process (int or float)
    #   - end:      end of x axis range to process (int or float)
    #   - norm:     normalise data to maximum (boolean)
    #   - debug:    print debug messages (boolean)
    if debug == True:
        print("    "*(inset) + "slicing spectrum...")
        print("    "*(inset+1) + "input arrays:", np.shape(x), np.shape(y))
        print("    "*(inset+1) + "y array ndim:", y.ndim)
    if start > np.amax(x) or end < np.amin(x):
        print("    "*(inset+1) + "slice range:", start, end)
        print("    "*(inset+1) + "spectrum x range:", np.amin(x), np.amax(x))
        raise Exception("attempting to slice spectrum beyond x range!")
    if np.all(std != None):
        std_check = True
    else:
        std_check = False
    if debug == True and std_check == True:
        print("    "*(inset+1) + "stdev array:", np.shape(std))
    y_max = []
    if x.ndim == y.ndim and y.ndim > 1:
        # multiple spectra, each with own x and y
        x_slice = []
        y_slice = []
        std_slice = []
        for i in range(np.shape(y)[1]):
            sliced = np.ravel(np.where((start <= x[:,i]) & (x[:,i] <= end)))
            x_slice.append(x[sliced,i])
            y_max.append(np.amax(y[sliced,i]))
            if norm == True:
                y_slice.append(y[sliced,i] / y_max[-1])
                if std_check:
                    std_slice.append(std[sliced,i] / y_max[-1])
            else:
                y_slice.append(y[sliced,i])
                if std_check:
                    std_slice.append(std[sliced,i])
    elif y.ndim > 1:
        # multiple spectra with common x axis
        sliced = np.ravel(np.where((start <= x) & (x <= end)))
        x_slice = x[sliced]
        y_slice = []
        for i in range(np.shape(y)[1]):
            y_max.append(np.amax(y[sliced,i]))
            if norm == True:
                y_slice.append(y[sliced,i] / y_max[-1])
                if std_check:
                    std_slice.append(std[sliced,i] / y_max[-1])
            else:
                y_slice.append(y[sliced,i])
                if std_check:
                    std_slice.append(std[sliced,i])
    else:
        # single spectrum
        sliced = np.ravel(np.where((start <= x) & (x <= end)))
        x_slice = x[sliced]
        y_max.append(np.amax(y[sliced]))
        if norm == True:
            y_slice = y[sliced] / y_max[-1]
            if std_check:
                std_slice = std[sliced] / y_max[-1]
        else:
            y_slice = y[sliced]
            if std_check:
                std_slice = std[sliced]
    x_slice = np.asarray(x_slice)
    y_slice = np.asarray(y_slice)
    if std_check == True:
        std_slice = np.asarray(std_slice)
        if debug == True:
            print("    "*(inset+1) + "sliced arrays:", np.shape(x_slice), np.shape(y_slice), np.shape(std_slice))
            print("    "*(inset+1) + "x range:", np.amin(x_slice), np.amax(x_slice))
            print("    "*(inset+1) + "y range:", np.amin(y_slice), np.amax(y_slice), np.amax(y_max))
        return x_slice, y_slice, std_slice
    else:
        if debug == True:
            print("    "*(inset+1) + "sliced arrays:", np.shape(x_slice), np.shape(y_slice))
            print("    "*(inset+1) + "x range:", np.amin(x_slice), np.amax(x_slice))
            print("    "*(inset+1) + "y range:", np.amin(y_slice), np.amax(y_slice), np.amax(y_max))
        return x_slice, y_slice

def normalise_spectrum(x, y, std=None, y_max=None, max_start=400, max_end=4000, debug=False):
    # function for normalising a spectrum
    # argument summary:
    #   - x:        x values (1D or 2D array)
    #   - y:        y values (1D or 2D array)
    #   - std:      st. dev. of y (optional, 1D or 2D array)
    #   - start:    start of x axis range to process (int or float)
    #   - end:      end of x axis range to process (int or float)
    #   - debug:    print debug messages (boolean)
    # check if stdev included
    if np.all(std != None):
        std_check = True
    else:
        std_check = False
    # check y_max
    if y_max != None:
        # use input y_max
        pass
    else:
        # instead use max intensity from range max_start - max_end
        y_max = find_max(x, y, max_start, max_end)[1]
    if std_check == True:
        return y / y_max, std / y_max
    else:
        return y / y_max

"""
# ==================================================
# functions for fitting and subtracting a baseline
# ==================================================
"""

def average_list(x, y, point_list, window, debug=False):
    # function for taking a set of user-defined points and creating arrays of their average x and y values
    if debug == True:
        print("        ", point_list)
    x_averages = np.zeros_like(point_list, dtype=float)
    y_averages = np.zeros_like(point_list, dtype=float)
    point_num = 0
    for i in range(np.size(point_list)):
        point_num += 1
        x_averages[i], y_averages[i] = local_average(x, y, point_list[i], window)
        if debug == True:
            print("        point", str(point_num), ": ", x_averages[i], y_averages[i])
    return x_averages, y_averages

def local_average(x, y, center, window):
    # function for finding the average position for a set of points with a given center +/- window
    center_ind = np.argmin(np.absolute(x - center))
    start_ind = center_ind - window
    end_ind = center_ind + window
    if start_ind < 0:
        start_ind = 0
    if end_ind > len(y)-1:
        end_ind = len(y)-1
    x_temp = x[start_ind:end_ind]
    y_temp = y[start_ind:end_ind]
    x_average = (np.average(x_temp))
    y_average = (np.average(y_temp))
    return x_average, y_average

def f_polynomial(x, *params):
    # function for generating an exponential baseline
    y = params[0]
    for i in range(1, len(params)):
        y += params[i] * x**i
    return y

def polynomial_fit(x_averages, y_averages, sigma, order=15, debug=False):
    # function for fitting selected average data-points using a polynominal function
    if len(x_averages) > int(order):
        guess = np.zeros((int(order)))
    else:
        guess = np.zeros_like(x_averages)
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_polynomial, x_averages, y_averages, sigma=sigma, p0=guess)
    if debug == True:
        print("        fitted parameters:", fit_coeffs)
    return fit_coeffs, fit_covar

def subtract_baseline(measurement, x_key='raman_shift', y_key='y', output_key='y_sub', x_list=[], base='polynomial', order=15, find_minima=True, window=35, fixed_ends=True, baseline_by_section=False, splits=[], plot=False, show_plot=True, plot_name=None, log=None, inset=0, debug=False):
    # script for subtracting a baseline by fitting it with a polynomial
    # argument summary:
    #   - measurement:      Measurement object containing data to be processed
    #   - x_key:            key for x axis values to get from measurement (string)
    #   - y_key:            key for y axis values to get from measurement (string)
    #   - output_key:       key for recording results in measurement (string)
    #   - x_list:           list of x positions to evaluate when fitting reference to target (list or 1D array)
    #   - base:             function for generating baseline curve (string)
    #   - order:            order for polynomial (int)
    #   - find_minima:      find minimum value within window around each point? (boolean)
    #   - window:           window for finding minima around each point, in x axis units (int or float)
    #   - fixed_ends:       force baseline to pass through ends of spectrum? (boolean)
    #   - baseline_by_section:  divide spectrum into user-defined sections for baselining separately? (boolean)
    #   - splits:           points on x axis for dividing spectrum into sections (list of ints or floats)
    #   - plot:             generate summary figure for process? (boolean) - only applies to single spectra!
    #   - show_plot:        show plot in console window? (boolean)
    #   - plot_name:        suffix to use when naming figure file (string)
    #   - debug:            print debug messages? (boolean)
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    if debug == True:
        print()
        print("    "*(inset) + "running baseline subtraction on measurement %s" % measurement.title)
    if len(x_list) == 0:
        raise Exception("    "*(inset+1) + "no x positions passed to subtract_baseline function!")
    if isinstance(measurement, Measurement) == False:
        raise Exception("    "*(inset+1) + "first argument passed to subtract_baseline function must be a Measurement instance!")
    elif hasattr(measurement, x_key) == False:
        raise Exception("    "*(inset+1) + "%s not in measurement data!" % x_key)
    elif hasattr(measurement, y_key) == False:
        raise Exception("    "*(inset+1) + "%s not in measurement data!" % y_key)
    if baseline_by_section == True and len(splits) == 0:
        raise Exception("    "*(inset+1) + "if baseline_by_section argument is true, splits argument must contain at least one value!")
    if plot == True and plot_name == None:
        # use default suffix for baseline
        plot_name = "baselined"
    if type(log) != str:
        log = 'baselined using %s' % base
        
    # ==================================================
    # get data from measurement object
    x, y = measurement(x_key, y_key)
    old_log = measurement[y_key].log
    if debug == True:
        print("    "*(inset+1) + "input arrays:", x_key, np.shape(x), y_key, np.shape(y))
        print("    "*(inset+1) + "previous y processing:", "\n".join(old_log))
        
    # ==================================================
    # smooth data for fitting
    if y.ndim == 1:
        # convert single-spec y to 2D array for handling
        single_spec = True
        y_s = smooth_spectrum(y, 5, 3)[:,np.newaxis]
    elif np.shape(y)[1] == 1:
        # y array is 2D but contains single spectrum
        single_spec = True
        y_s = np.zeros_like(y)
        for i in range(np.shape(y)[1]):
            y_s[:,i] = smooth_spectrum(y[:,i], 5, 3)
    else:
        # y array is 2D contains multiple spectra to process individually
        single_spec = False
        y_s = np.zeros_like(y)
        for i in range(np.shape(y)[1]):
            y_s[:,i] = smooth_spectrum(y[:,i], 5, 3)
        if plot == True:
            print("    "*(inset+1) + "cannot plot results of peak subtraction for multi-spec measurement!")
            plot = False
    
    # ==================================================
    # begin figure if necessary
    if plot == True:
        # prepare figure
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot(211)  # ax1: original target spec overlaid with rescaled reference spec
        ax2 = plt.subplot(212, sharex=ax1)  # ax2: target spec after subtraction
        ax1.set_title("%s\nBaseline Subtraction" % (measurement.title))
        ax1.set_ylabel("Average Intensity")
        ax2.set_ylabel("Average Intensity")
        ax2.set_xlabel("Raman shift (cm$^{-1}$)")
        ax1.set_xlim(measurement.x_start, measurement.x_end)
        # plot original spectrum in ax1
        ax1.plot(x, y, 'k', label='input spectrum')
    
    # ==================================================
    # trim x_list to points that fall within spectrum x range
    x_list = np.asarray(x_list)
    x_list = np.sort(x_list[np.where((np.amin(x)+10 <= x_list) & (x_list <= np.amax(x)-10))])
    # reduce polynomial order if it exceeds number of points-1
    if len(x_list)-1 < order:
        order = len(x_list)-1
    if debug == True:
        print("    "*(inset+1) + "x range: %0.f - %0.f" % (np.amin(x), np.amax(x)))
        print("    "*(inset+1) + "x_list for subtraction:", x_list)
        
    # ==================================================
    # divide up data into sections (default: 1)
    if baseline_by_section == True and len(splits) > 0:
        fixed_ends = True
        # convert input splits (in x axis units) into a sorted list of indices for use by np.split()
        section_splits = np.asarray([np.argmin(np.abs(x - float(point))) for point in splits if point > np.amin(x) and point < np.amax(x)])
        if debug == True:
            print("    "*(inset+1) + "spectrum split at x=", ", ".join([str(split) for split in splits]))
    else:
        section_splits = 1
        if debug == True:
            print("    "*(inset+1) + "no spectrum splitting")
    x_splits = np.split(x, section_splits, axis=0)
    y_splits = np.split(y_s, section_splits, axis=0)
    
    # ==================================================
    # do baseline fitting, section by section
    fitted_baseline = [np.zeros_like(split) for split in y_splits]
    for i2, x_slice, y_slice in zip(range(0, len(x_splits)), x_splits, y_splits):
        # for each section
        if debug == True:
            print("    "*(inset+1) + "section %s x, y arrays:" % i2, np.shape(x_slice), np.shape(y_slice))
            print("    "*(inset+2) + "section %s x range: %0.1f - %0.1f cm-1" % (i2, np.amin(x_slice), np.amax(x_slice)))
        for i in range(np.shape(y_s)[1]):
            # for each spectrum in slice
            if debug == True:
                print("    "*(inset+2) + "spectrum %s" % i)
            
            # ==================================================
            # trim x_list to section
            local_x_list = np.asarray(x_list)[np.where((np.amin(x_slice) <= np.asarray(x_list)) & (np.asarray(x_list) < np.amax(x_slice)))]
            if find_minima == True:
                # find minimum value within window around each point in local_x_list 
                points = []
                for point in local_x_list:
                    points.append(find_min(x_slice, y_slice[:,i], point-window, point+window)[0])
            else:
                points = local_x_list
            
            # ==================================================
            # create arrays of average values for each point +/-5 pixels
            x_averages, y_averages = average_list(x_slice, y_slice[:,i], points, 5, debug=debug)
            # add fixed first and last points if applicable, and create sigma array for point weighting
            sigma = np.ones_like(y_averages)
            if fixed_ends == True:
                for index in [5, -6]:
                    x_0, y_0 = local_average(x_slice, y_slice[:,i], x_slice[index], 5)
                    x_averages = np.append(x_averages, x_0)
                    y_averages = np.append(y_averages, y_0)
                    sigma = np.append(sigma, 0.1)
            # sort x_list into ascending order
            sort = np.argsort(x_averages)
            x_averages = x_averages[sort]
            y_averages = y_averages[sort]
            sigma = sigma[sort]

            # ==================================================
            # attempt to fit section data using specified base function
            while True:
                try:
                    # first attempt (original order)
                    local_order = order
                    if local_order > len(y_averages)-1:
                        local_order = len(y_averages)-1
                    fit_coeffs, fit_covar = polynomial_fit(x_averages, y_averages, sigma, order=local_order, debug=debug)
                    basefit = f_polynomial(x_slice, *fit_coeffs)
                    if debug == True:
                        print("    "*(inset+1) + "basefit:", np.shape(basefit))
                        print("    "*(inset+1) + "baseline section:", np.shape(fitted_baseline[i2][:,i]))
                    fitted_baseline[i2][:,i] = basefit
                    break
                except Exception as e:
                    if debug == True:
                        print("    "*(inset+1) + "something went wrong! Exception:", e)
                        print("    "*(inset+2) + "attempting another fit with reduced polynomial order...")
                    try:
                        # second attempt (reduced order)
                        if local_order - 1 > 1:
                            local_order -= 1
                        fit_coeffs, fit_covar = polynomial_fit(x_averages, y_averages, sigma, order=local_order, debug=debug)
                        basefit = f_polynomial(x_slice, *fit_coeffs)
                        if debug == True:
                            print("    "*(inset+2) + "basefit:", np.shape(basefit))
                            print("    "*(inset+2) + "baseline section:", np.shape(fitted_baseline[i2][:,i]))
                        fitted_baseline[i2][:,i] = basefit
                        if plot == True:
                            ax1.scatter(x_averages, y_averages, c=Color_list_light[i % len(Color_list_light)], zorder=3)
                            ax1.plot(x_slice, basefit, Color_list_light[i % len(Color_list_light)], zorder=2)
                        break
                    except Exception as e:
                        if debug == True:
                            print("    "*(inset+2) + "something went wrong! Exception:", e)
                            print("    "*(inset+3) + "attempting third fit with reduced polynomial order...")
                        try:
                            # second attempt (reduced order)
                            if local_order - 1 > 1:
                                local_order -= 1
                            fit_coeffs, fit_covar = polynomial_fit(x_averages, y_averages, sigma, order=local_order, debug=debug)
                            basefit = f_polynomial(x_slice, *fit_coeffs)
                            if debug == True:
                                print("    "*(inset+3) + "basefit:", np.shape(basefit))
                                print("    "*(inset+3) + "baseline section:", np.shape(fitted_baseline[i2][:,i]))
                            fitted_baseline[i2][:,i] = basefit
                            if plot == True:
                                ax1.scatter(x_averages, y_averages, c=Color_list_light[i % len(Color_list_light)], zorder=3)
                                ax1.plot(x_slice, basefit, Color_list_light[i % len(Color_list_light)], zorder=2)
                            break
                        except Exception as e:
                            if debug == True:
                                print("    "*(inset+3) + "something went wrong again! Exception:", e)
                                print("    "*(inset+4) + "3 attempts failed. Taking minimum for flat baseline instead.")
                            # both attempts failed, use flat baseline instead
                            fitted_baseline[i2][:,i] = np.full_like(y_slice, np.amin(y_slice))
                            break
            # add section baseline to plot
            if plot == True and single_spec == True:
                ax1.scatter(x_averages, y_averages, c=Color_list_light[i2 % len(Color_list_light)], zorder=3)
                ax1.plot(x_slice, fitted_baseline[i2][:,0], c=Color_list_light[i2 % len(Color_list_light)], zorder=2, label='section %s' % i2)
    
    if debug == True:
        print("    "*(inset+1) + "split baseline arrays:", [np.shape(split) for split in fitted_baseline])
    
    # ==================================================
    # concatenate sections into a single baseline and subtract from data
    fitted_baseline = np.concatenate(fitted_baseline, axis=0)
    y_sub = y - fitted_baseline
    
    # ==================================================
    # save subtracted spectrum to measurement
    measurement.add_spectrum(output_key, y_sub, label="Baselined Intensity (counts)", log=old_log + [log])
    if debug == True:
        print("    "*(inset+1) + "output %s array:" % output_key, np.shape(y_sub))
        print("    "*(inset+1) + "output log:", old_log + [log])
    
    # ==================================================
    # complete figure if required
    if plot == True and single_spec == True:
        # plot subtracted spectrum in ax2
        ax2.plot(x, y_sub, label='baselined')
        # finish figure
        ax1.legend()
        ax2.legend()
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig("%s%s_%s.png" % (measurement.fig_dir, measurement.title, plot_name), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
    # ==================================================
    # end of function
    if debug == True:
        print("    "*(inset+1) + "end of subtract_baseline function")

"""
# ==================================================
# functions for fitting and subtracting a reference spectrum
# ==================================================
"""

def reference_fit(params, ref_x, ref_y, target_x, target_y):
    # function for use with LMFIT minimize(), provides difference between target data and rescaled reference data
    # rescale ref_y using scale_factor parameter
    ref_y_rescaled = ref_y * params['scale_factor']
    # get ref - target difference in x,y
    dx = ref_x - target_x
    dy = ref_y_rescaled - target_y
    # return euclidean distance between ref and target
    return np.sqrt(dx**2 + dy**2)
            
def subtract_reference(ref, target, x_key='raman_shift', y_key='y_sub', output_key='y_refsub', x_list=[], plot=False, show_plot=True, plot_name=None, log=None, inset=0, debug=False):
    # script for subtracting a scaled reference spectrum by fitting it to a set of points
    # argument summary:
    #   - ref:              Measurement object containing data for reference spectrum
    #   - target:           Measurement object containing data for target spectrum
    #   - x_key:            list of x positions to evaluate when fitting reference to target (list or 1D array)
    #   - y_key:            key for x axis values in spec dict (string)
    #   - y_type:           key for y axis values in spec dict (string)
    #   - output_key:       key for outputting results into spec dict (string)
    #   - debug:            print debug messages? (boolean)
    #   - plot:             generate summary figure for process? (boolean) - only applies to single spectra!
    #   - show_plot:        show plot in console window? (boolean)
    #   - plot_name:        suffix to use when naming figure file (string)
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    if debug == True:
        print()
        print("    "*(inset) + "running reference subtraction on measurement %s using %s" % (target.title, ref.name))
    if type(x_list) != list or len(x_list) == 0:
        raise Exception("    "*(inset+1) + "no x positions passed to subtract_reference function!")
    if isinstance(ref, Measurement) == False:
        raise Exception("    "*(inset+1) + "ref argument passed to subtract_reference function must be a Measurement object!")
    elif hasattr(ref, x_key) == False:
        raise Exception("    "*(inset+1) + "%s not in ref data!" % x_key)
    elif hasattr(ref, 'y_av_sub') == False:
        raise Exception("    "*(inset+1) + "y_av_sub not in ref data!")
    if isinstance(target, Measurement) == False:
        raise Exception("    "*(inset+1) + "target argument passed to subtract_reference function must be a Measurement object!")
    elif hasattr(target, x_key) == False:
        raise Exception("    "*(inset+1) + "%s not in measurement data!" % x_key)
    elif hasattr(target, y_key) == False:
        raise Exception("    "*(inset+1) + "%s not in measurement data!" % y_key)
    if plot == True and plot_name == None:
        plot_name = ref['name']+"-sub"
    if type(log) != str:
        log = '%s fitted and subtracted' % ref.name
        
    # ==================================================
    # get spectrum data for both ref, target
    ref_x, ref_y = ref(x_key, 'y_av_sub')
    ref_y /= np.amax(ref_y)
    target_x, target_y = target(x_key, y_key)
    indices = target[y_key].indices
    old_log = target[y_key].log
    if debug == True:
        print("    "*(inset+1) + "input ref arrays:", np.shape(ref_x), np.shape(ref_y))
        print("    "*(inset+1) + "input target arrays:", np.shape(target_x), np.shape(target_y))
        print("    "*(inset+1) + "previous y processing:", "\n".join(old_log))
        
    # ==================================================
    # smooth data for fitting
    if target_y.ndim == 1:
        # convert single-spec y to 2D array for handling
        single_spec = True
        target_y_s = smooth_spectrum(target_y, 5, 3)[:,np.newaxis]
    elif np.shape(target_y)[1] == 1:
        # y array is 2D but contains single spectrum
        single_spec = True
        target_y_s = np.zeros_like(target_y)
        for i in range(np.shape(target_y)[1]):
            target_y_s[:,i] = smooth_spectrum(target_y[:,i], 5, 3)
    else:
        # y array is 2D contains multiple spectra to process individually
        single_spec = False
        target_y_s = np.zeros_like(y)
        for i in range(np.shape(target_y)[1]):
            target_y_s[:,i] = smooth_spectrum(target_y[:,i], 5, 3)
        if plot == True:
            print("    "*(inset+1) + "cannot plot results of subtraction for multi-spec measurement!")
            plot = False
    if debug == True:
        print("    "*(inset+1) + "smoothed y array:", np.shape(target_y_s))
            
    # ==================================================
    # trim x_list to points that fall within spectrum x range
    x_list = np.asarray(x_list)
    x_list = np.sort(x_list[np.where((np.amin(ref_x)+10 <= x_list) & (x_list <= np.amax(ref_x)-10) & (np.amin(target_x)+10 <= x_list) & (x_list <= np.amax(target_x)-10))])
    if debug == True:
        print("    "*(inset+1) + "x_list for fitting:" % x_list)
        
    # ==================================================
    # get x,y values using x_list and fit reference to match
    target_y_refsub = np.zeros_like(target_y_s)
    # for each input target spectrum
    for i in range(np.shape(target_y)[1]):
        # create arrays of average values for each point +/-5 pixels
        ref_x_averages, ref_y_averages = average_list(ref_x, ref_y, x_list, 5, debug=debug)
        target_x_averages, target_y_averages = average_list(target_x, target_y[:,i], x_list, 5, debug=debug)
    
        # create LMFIT parameters object
        params = lmfit.Parameters()
        params.add('scale_factor', value=1., min=0.)
        
        # fit target values using reference values
        fit_output = lmfit.minimize(reference_fit, params, args=(ref_x_averages, ref_y_averages, target_x_averages, target_y_averages))
        
        if debug == True:
            print("    "*(inset+1) + "spec %s: rescale factor = %0.2f" % (indices[i], fit_output.params['scale_factor'].value))
        
        # interpolate rescaled reference to match target x values and subtract
        ref_y_rescaled = ref_y[:,0] * fit_output.params['scale_factor']
        ref_y_rescaled_interp = np.interp(target_x, ref_x, ref_y_rescaled)
        target_y_refsub[:,i] = target_y[:,i] - ref_y_rescaled_interp
    
    # ==================================================
    # save subtracted spectrum to measurement object
    target.add_spectrum(output_key, target_y_refsub, label="Intensity After %s Subtraction (counts)" % ref.name, log=old_log + [log])
    if debug == True:
        print("    "*(inset+1) + "output %s array:" % output_key, np.shape(target_y_refsub))
        print("    "*(inset+1) + "output log:", old_log + [log])
    
    # ==================================================
    # create figure if required
    if plot == True:
        # prepare figure
        plt.figure(figsize=(8,8))
        ax1 = plt.subplot(211)  # ax1: original target spec overlaid with rescaled reference spec
        ax2 = plt.subplot(212, sharex=ax1)  # ax2: target spec after subtraction
        ax1.set_title("%s\n%s Subtraction" % (target.title, ref.name))
        ax1.set_ylabel("Average Intensity")
        ax2.set_ylabel("Average Intensity")
        ax2.set_xlabel("Raman shift (cm$^{-1}$)")
        ax1.set_xlim(target.x_start, target.x_end)
        # plot original spectrum in ax1
        ax1.plot(target_x, target_y, 'k', label='input spectrum')
        # plot points used for fitting in ax1
        ax1.plot(target_x_averages, target_y_averages, 'ro')
        # plot rescaled reference spectrum in ax1
        ax1.plot(ref_x, ref_y_rescaled, label='rescaled ref')
        # plot refsub spectrum in ax2
        ax2.plot(target_x, target_y_refsub, label='after sub')
        # finish figure
        ax1.legend()
        ax2.legend()
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig("%s%s_%s.png" % (target.fig_dir, target.title, plot_name), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
    # ==================================================
    # end of function
    if debug == True:
        print("    "*(inset+1) + "end of subtract_reference function")

"""
# ==================================================
# functions for finding peaks
# ==================================================
"""

def find_max(x, y, start, end):
    # function for finding the maximum in a slice of input data
    x_slice = x[np.where((start <= x) & (x <= end))]        # create slice
    y_slice = y[np.where((start <= x) & (x <= end))]
    i = np.argmax(y_slice)
    return np.array([x_slice[i], y_slice[i]]) # return x,y position of the maximum

def find_min(x, y, start, end):
    # function for finding the minimum in a slice of input data
    x_slice = x[np.where((start <= x) & (x <= end))]        # create slice
    y_slice = y[np.where((start <= x) & (x <= end))]
    i = np.argmin(y_slice)
    return np.array([x_slice[i], y_slice[i]]) # return x,y position of the minimum

def find_maxima(x, y, min_sep, threshold, inset=0, debug=False):
    # function for finding the maxima of input data. Each maximum will have the largest value within ±/- min_sep.
    index_list = argrelextrema(y, np.greater, order=min_sep)    # determines indices of all maxima
    all_maxima = np.asarray([x[index_list], y[index_list]]) # creates an array of x and y values for all maxima
    y_limit = threshold * np.amax(y)                        # set the minimum threshold for defining a 'peak'
    x_maxima = all_maxima[0, all_maxima[1] >= y_limit]      # records the x values for all valid maxima
    y_maxima = all_maxima[1, all_maxima[1] >= y_limit]      # records the y values for all valid maxima
    maxima = np.asarray([x_maxima, y_maxima])               # creates an array for all valid maxima
    if debug == True:
        print("    "*(inset) + "maxima found:")
        print("    "*(inset+1) + "x: ", maxima[0])
        print("    "*(inset+1) + "y: ", maxima[1])
    return maxima

def get_noise(x, y, start=2000, end=2100, inset=0, debug=False):
    # function for estimating background noise level
    if end > np.amax(x):
        start, end = (1800, 1900)
    if end > np.amax(x):
        start, end = (600, 700)
    if debug == True:
        print("    "*(inset) + "estimating noise")
        print("    "*(inset+1) + "noise region: %0.1f - %0.1f" % (start, end))
    # noise is calculated as st. dev. of a 100 cm-1 wide peak-free region, defaults to 2400-2500
    noise_x, noise_y = slice_spectrum(x, y, start=start, end=end, inset=inset+1, debug=debug)
    noise = np.std(noise_y)
    noise = np.std(noise_y)
    if debug == True:
        print("    "*(inset+1) + "noise level: %0.1f" % noise)
    return noise

def detect_peaks(measurement, x_key='raman_shift', y_key='y_av_sub', output_key='detected_peaks', SNR_threshold=10, norm_threshold=0.05, min_sep=20, x_start=None, x_end=None, noise=None, noise_region=(2000,2100), plot=False, show_plot=True, plot_name=None, save_to_file=True, log=None, debug=False, inset=0):
    # script for finding peaks in a spectrum
    # argument summary:
    #   - measurement:      Measurement object containing all data
    #   - SNR_threshold:    minimum signal:noise ratio for valid peaks (int or float)
    #   - norm_threshold:   minimum intensity for valid peaks relative to spectrum maximum (float, 0-1)
    #   - min_sep:          minimum separation between valid peaks, in x axis units (int or float)
    #   - x_start:          start of x axis range to be processed (int or float)
    #   - x_end:            end of x axis range to be processed (int or float)
    #   - noise:            user-defined noise level (int or float), or None for automatic noise estimation
    #   - noise_region:     user-defined region to calculate noise from (tuple of 2 ints or 2 floats)
    #   - x_key:            key for x axis values in spec dict (string)
    #   - y_key:            key for y axis values in spec dict (string)
    #   - output_key:       key for outputting results into spec dict (string)
    #   - debug:            print debug messages? (boolean)
    #   - plot:             generate summary figure for process? (boolean) - only applies to single spectra!
    #   - show_plot:        show plot in console window? (boolean)
    #   - plot_name:        suffix to use when naming figure file (string)
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    print()
    print("    "*inset + "running peak detection on measurement %s" % measurement.title)
    if isinstance(measurement, Measurement) == False:
        raise Exception("    "*(inset+1) + "first argument passed to subtract_baseline function must be a Measurement instance!")
    elif hasattr(measurement, x_key) == False:
        raise Exception("    "*(inset+1) + "%s not in measurement data!" % x_key)
    elif hasattr(measurement, y_key) == False:
        raise Exception("    "*(inset+1) + "%s not in measurement data!" % y_key)
    if noise != None and type(noise) not in [int, float, np.int16, np.int32, np.float32, np.float64]:
        raise Exception("    "*(inset+1) + "noise argument passed to find_peaks must be integer or float!")
    elif type(noise) in [int, float, np.int16, np.int32, np.float32, np.float64]:
        if noise <= 0:
            raise Exception("    "*(inset+1) + "noise argument passed to find_peaks must be greater than 0!")
    if plot == True and plot_name == None:
        plot_name = "peak-detection"
    if type(log) != str:
        log = 'automatic peak detection'
    
    # ==================================================
    # get data from measurement object
    x, y = measurement(x_key, y_key)
    indices = measurement[y_key].indices
    old_log = measurement[y_key].log
    if debug == True:
        print("    "*(inset+1) + "input arrays:", x_key, np.shape(x), y_key, np.shape(y))
        print("    "*(inset+1) + "previous y processing:", "\n".join(old_log))
        
    # ==================================================
    # smooth data for fitting
    if y.ndim == 1:
        # convert single-spec y to 2D array for handling
        single_spec = True
        y_s = smooth_spectrum(y, 5, 3)[:,np.newaxis]
    elif np.shape(y)[1] == 1:
        # y array is 2D but contains single spectrum
        single_spec = True
        y_s = np.zeros_like(y)
        for i in range(np.shape(y)[1]):
            y_s[:,i] = smooth_spectrum(y[:,i], 5, 3)
    else:
        # y array is 2D contains multiple spectra to process individually
        single_spec = False
        y_s = np.zeros_like(y)
        for i in range(np.shape(y)[1]):
            y_s[:,i] = smooth_spectrum(y[:,i], 5, 3)
        if plot == True:
            print("    "*(inset+1) + "cannot plot results of peak detection for multi-spec measurement!")
            plot = False
    if x_start == None:
        x_start = np.amin(x)
    if x_end == None:
        x_end = np.amax(x)
    
    # ==================================================
    # search for maxima in spectra
    peak_data = []
    peak_data = {'spec_index': [], 'centers': [], 'heights': []}
    for i in range(np.shape(y)[1]):
        # for each input target spectrum
        
        noise = get_noise(x, y_s[:,i], start=noise_region[0], end=noise_region[1], inset=inset+1, debug=debug)

        # slice the data for peak detection
        x_slice, y_slice = slice_spectrum(x, y_s[:,i], start=x_start, end=x_end, inset=inset+1)
        
        # find local maxima above norm_threshold, produces two lists: x positions and y intensity values
        y_min = np.amin(y_slice)
        y_max = np.amax(y_slice)
        maxima = find_maxima(x_slice, y_slice, min_sep, norm_threshold)
        if debug == True:
            print("    "*(inset+1) + "spec %s:" % indices[i])
            print("    "*(inset+2) + "%s maxima found:" % len(maxima[0]), ", ".join(["%0.1f" % peak for peak in maxima[0]]))

        # only pass maxima that are above SNR threshold
        maxima_pass = [[],[]]
        for i2 in range(0, len(maxima[0])):
            if maxima[1,i2] > float(SNR_threshold) * noise:
                maxima_pass[0].append(maxima[0,i2])
                maxima_pass[1].append(maxima[1,i2])
        maxima_pass = np.asarray(maxima_pass)   # same structure as maxima
        if debug == True:
            print("    "*(inset+2) + "%s pass SNR threshold:" % len(maxima_pass[0]), ", ".join(["%0.1f" % peak for peak in maxima_pass[0]]))
            print("    "*(inset+2) + "%s did not pass:" % (len(maxima[0])-len(maxima_pass[0])), ", ".join(["%0.1f" % peak for peak in maxima[0] if peak not in maxima_pass[0]]))
        
        # add data to temp storage array
        peak_data['spec_index'] += list(np.full_like(maxima_pass[0], indices[i]))
        peak_data['centers'] += list(maxima_pass[0])
        peak_data['heights'] += list(maxima_pass[1])
    
    peak_data = pd.DataFrame(peak_data)
    print("    "*(inset+1) + "total %s potential peaks found across %s spectra" % (len(peak_data['spec_index']), len(np.unique(peak_data['spec_index']))))
    
    # ==================================================
    # save detected peaks to spec dict
    measurement.detected_peaks = peak_data
    if debug == True:
        print("    "*(inset+1) + "output %s array:" % output_key, np.shape(peak_data))
        print("    "*(inset+1) + "output log:", old_log + [log])
        
    # save detected peaks to file
    if len(peak_data['centers']) > 0 and save_to_file == True:
        print()
        print("    "*(inset) + "saving peak fit data to file")
        # save to file
        outdir = "%s%s_detected-peaks.csv" % (measurement.out_dir, measurement.title)
        if debug == True:
            print("    "*(inset+1) + "detected peak properties dataframe:")
            print(peak_data.info())
            print("    "*(inset+1) + "saving to %s" % outdir)
        # save data to output folder
        peak_data.to_csv(outdir)
    
    # ==================================================
    # create figure if required
    if plot == True:
        # prepare figure
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)
        ax1.set_title("%s\nAutomatic Peak Detection" % (measurement.title))
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax1.set_ylabel("Intensity (counts)")
        ax1.set_xlim(x_start, x_end)
        ax1.set_ylim(-0.2*y_max, 1.2*y_max)
        # add horizontal line for relative intensity threshold (blue)
        ax1.axhline(norm_threshold*y_max, color='b', linestyle=':', alpha=0.5, label='min. int.')
        # add horizontal line for SNR threshold (red)
        ax1.axhline(float(SNR_threshold)*noise, color='r', linestyle=':', alpha=0.5, label='min. SNR')
        # plot spectrum
        plt.plot(x_slice, y_slice, 'k', label='data')
        # plot maxima that fail
        ax1.plot(maxima[0], maxima[1], 'ro', label='fail (%d)' % (len(maxima[0])-len(maxima_pass[0])))
        # plot maxima that pass
        ax1.plot(maxima_pass[0], maxima_pass[1], 'bo', label='pass (%d)' % (len(maxima_pass[0])))
        if debug == True:
            print("    "*(inset+1) + "detected peaks:")
        for i2 in range(len(maxima_pass[0])):
            # report detected positions
            ax1.text(maxima_pass[0,i2], maxima_pass[1,i2]+0.05*y_max, "%0.f" % maxima_pass[0,i2], rotation=90, va='bottom', ha='left')
            if debug == True:
                print("    "*(inset+2) + "%0.f cm: %0.2f" % (maxima_pass[0,i2], maxima_pass[1,i2]/y_max))
        
        # create second y axis for SNR values
        ax2 = ax1.twinx()
        ax2.set_ylim(intensity2snr(ax1.get_ylim(), noise))
        ax2.set_ylabel("SNR")
        ax1.legend()
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig("%s%s_%s.png" % (measurement.fig_dir, measurement.title, plot_name), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
            
    # ==================================================
    # end of function
    if debug == True:
        print("    "*(inset) + "end of detect_peaks() function")

"""
# ==================================================
# functions for fitting peaks
# ==================================================
"""

def multiG_curve(x, params, maxima):
    # function for generating a model gaussian curve using defined parameters
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiG_fit(params, x, y, maxima):
    # function for use with LMFIT minimize(), provides difference between input data and gaussian model
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        model += A * np.exp(-0.5*(x - mu)**2/(sigma**2))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiL_curve(x, params, maxima):
    # function for generating a model lorentzian curve using defined parameters
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        gamma = params['gamma_%s' % i]
        model += A * (gamma**2)/((x - mu)**2 + gamma**2)
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiL_fit(params, x, y, maxima):
    # function for use with LMFIT minimize(), provides difference between input data and lorentzian model
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        gamma = params['gamma_%s' % i]
        model += A * (gamma**2)/((x - mu)**2 + gamma**2)
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def multiPV_curve(x, params, maxima):
    # function for generating a model pseudo-voigt curve using defined parameters
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        gamma = np.sqrt(2.*np.log(2.)) * sigma
        eta = params['eta_%s' % i]
        model += A * (eta * (gamma**2)/((x - mu)**2 + gamma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return model

def multiPV_fit(params, x, y, maxima):
    # function for use with LMFIT minimize(), provides difference between input data and pseudo-voigt model
    model = np.zeros_like(x)
    for i in range(0, len(maxima)):
        A = params['amplitude_%s' % i]
        mu = params['center_%s' % i]
        sigma = params['sigma_%s' % i]
        gamma = np.sqrt(2.*np.log(2.)) * sigma
        eta = params['eta_%s' % i]
        model += A * (eta * (gamma**2)/((x - mu)**2 + gamma**2) + (1.-eta) * np.exp(-0.5*(x - mu)**2/(sigma**2)))
    gradient = params['gradient']
    intercept = params['intercept']
    model += gradient*x + intercept
    return (y - model)

def peak_fit_script(x, y, maxima, max_shift=10., min_fwhm=5., max_fwhm=150., function='g', vary_baseline=True, debug=False):
    # script for fitting a set of maxima with pre-defined peak functions
    if function.lower() in ['pv', 'pseudo-voigt', 'pseudovoigt', 'psuedo-voigt', 'psuedovoigt']:
        function = 'pv'
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac', 'fermidirac']:
        function = 'fd'
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        function = 'l'
    elif function.lower() in ['g', 'gauss', 'gaussian']:
        function = 'g'
    else:
        if debug == True:
            print("    specified peak function is not recognised, defaults to gaussian")
    # create LMFIT parameters object
    params = lmfit.Parameters()
    # add linear baseline parameters
    params.add('gradient', value=0., vary=vary_baseline)
    params.add('intercept', value=np.amin(y), vary=vary_baseline)
    for i in range(0, len(maxima)):
        # for each input peak, add necessary peak function parameters
        y_max = x[np.argmin(np.absolute(y - maxima[i]))]
        params.add('center_%s' % i, value=maxima[i], min=maxima[i]-max_shift, max=maxima[i]+max_shift)
        params.add('amplitude_%s' % i, value=y_max, min=0.)
        if function == 'pv':
            # pseudo-voigt functions use sigma parameter for width, eta for lorentzian factor
            params.add('sigma_%s' % i, value=10., min=min_fwhm/(2.*np.sqrt(2.*np.log(2))), max=max_fwhm/(2.*np.sqrt(2.*np.log(2))))
            params.add('eta_%s' % i, value=0.5, min=0., max=1.)
        elif function == 'fd':
            # symmetric fermi-dirac functions use width parameter for half-width, rounding for overall roundness
            params.add('width_%s' % i, value=10., min=min_fwhm/2., max=max_fwhm/2.)
            params.add('rounding_%s' % i, value=5., min=0.)
        elif function == 'l':
            # lorentzian functions use gamma for width
            params.add('gamma_%s' % i, value=10., min=2., max=max_fwhm/2.)
        else:
            # gaussian functions use sigma for width
            params.add('sigma_%s' % i, value=10., min=min_fwhm/(2.*np.sqrt(2.*np.log(2))), max=max_fwhm/(2.*np.sqrt(2.*np.log(2))))
    if debug == True:
        print("        initial parameters:")
        print(params.pretty_print())
    # run fit and generate fitted curve
    if function == 'pv':
        fit_output = lmfit.minimize(multiPV_fit, params, args=(x, y, maxima))
        fit_curve = multiPV_curve(x, fit_output.params, maxima)
    elif function == 'fd':
        fit_output = lmfit.minimize(multiFD_fit, params, args=(x, y, maxima))
        fit_curve = multiFD_curve(x, fit_output.params, maxima)
    elif function == 'l':
        fit_output = lmfit.minimize(multiL_fit, params, args=(x, y, maxima))
        fit_curve = multiL_curve(x, fit_output.params, maxima)
    else:
        fit_output = lmfit.minimize(multiG_fit, params, args=(x, y, maxima))
        fit_curve = multiG_curve(x, fit_output.params, maxima)
    if debug == True:
        print("        fit status: ", fit_output.message)
        print("        fitted parameters:")
        print(fit_output.params.pretty_print())
    return fit_output, fit_curve

def fwhm(params, i=None, function='g'):
    # returns the FWHM for any function using the appropriate calculation
    if function.lower() in ['v', 'voigt']:
        # voigt width is approximated by combination of lorentzian gamma and gaussian sigma parameters
        if i == None:
            fL = 2. * params['gamma']
            fG = np.sqrt(2. * np.log(2)) * params['sigma']
            return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
        else:
            fL = 2. * params['gamma_%s' % i]
            fG = np.sqrt(2. * np.log(2)) * params['sigma_%s' % i]
            return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
    elif function.lower() in ['pv', 'pseudo-voigt', 'pseudovoigt', 'psuedo-voigt', 'psuedovoigt']:
        # pseudo-voigt width is 2*sqrt(2*ln(2)) times the sigma parameter
        if i == None:
            return 2. * np.sqrt(2.*np.log(2.)) * params['sigma']
        else:
            return 2. * np.sqrt(2.*np.log(2.)) * params['sigma_%s' % i]
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac', 'fermidirac']:
        # fermi-dirac width is twice the width parameter
        if i == None:
            return 2. * params['width']
        else:
            return 2. * params['width_%s' % i]
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        # lorentzian width is twice the gamma parameter
        if i == None:
            return 2. * params['gamma']
        else:
            return 2. * params['gamma_%s' % i]
    else:
        # gaussian width is 2*sqrt(2*ln(2)) times the sigma parameter
        if i == None:
            return 2. * np.sqrt(2.*np.log(2.)) * params['sigma']
        else:
            return 2. * np.sqrt(2.*np.log(2.)) * params['sigma_%s' % i]
        
def fwhm_err(params, i=None, function='g'):
    # returns the FWHM standard error for any function using the appropriate calculation
    if function.lower() in ['v', 'voigt']:
        # voigt width is approximated by combination of lorentzian gamma and gaussian sigma parameters
        if i == None:
            if params['gamma'].stderr != None and params['sigma'].stderr != None:
                fL = 2. * params['gamma'].stderr
                fG = np.sqrt(2. * np.log(2)) * params['sigma'].stderr
                return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
            else:
                return 0.
        else:
            if params['gamma_%s' % i].stderr != None and params['sigma_%s' % i].stderr != None:
                fL = 2. * params['gamma_%s' % i].stderr
                fG = np.sqrt(2. * np.log(2)) * params['sigma_%s' % i].stderr
                return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
            else:
                return 0.
    elif function.lower() in ['pv', 'pseudo-voigt', 'pseudovoigt', 'psuedo-voigt', 'psuedovoigt']:
        # pseudo-voigt width is 2*sqrt(2*ln(2)) times the sigma parameter
        if i == None:
            if params['sigma'].stderr != None:
                return 2. * np.sqrt(2.*np.log(2.)) * params['sigma'].stderr
            else:
                return 0.
        else:
            if params['sigma_%s' % i].stderr != None:
                return 2. * np.sqrt(2.*np.log(2.)) * params['sigma_%s' % i].stderr
            else:
                return 0.
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac', 'fermidirac']:
        # fermi-dirac width is twice the width parameter
        if i == None:
            if params['width'].stderr != None:
                return 2. * params['width'].stderr
            else:
                return 0.
        else:
            if params['width_%s' % i].stderr != None:
                return 2. * params['width_%s' % i].stderr
            else:
                return 0.
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        # lorentzian width is twice the gamma parameter
        if i == None:
            if params['gamma'].stderr != None:
                return 2. * params['gamma'].stderr
            else:
                return 0.
        else:
            if params['gamma_%s' % i].stderr != None:
                return 2. * params['gamma_%s' % i].stderr
            else:
                return 0.
    else:
        # gaussian width is 2*sqrt(2*ln(2)) times the sigma parameter
        if i == None:
            if params['sigma'].stderr != None:
                return 2. * np.sqrt(2.*np.log(2.)) * params['sigma'].stderr
            else:
                return 0.
        else:
            if params['sigma_%s' % i].stderr != None:
                return 2. * np.sqrt(2.*np.log(2.)) * params['sigma_%s' % i].stderr
            else:
                return 0.
            
def fitting_regions(peak_list, start=400, end=1800, window=150., rounding=10, inset=0, debug=False):
    # function for dividing a list of peak positions into separate regions for fitting
    regions = []
    peak_list = np.sort(peak_list)
    if len(peak_list) == 0:
        raise Exception("    "*(inset) + "no peaks passed to fitting_regions() function!")
    else:
        peak_list_trim = peak_list[np.where((start <= peak_list) & (peak_list <= end))]
    if len(peak_list_trim) == 1:
        # single region centred around single peak, rounded to nearest 10.
        regions = [[rounding*np.floor((peak_list_trim[0] - window)/rounding), rounding*np.ceil((peak_list_trim[0] + window)/rounding)]]
    else:
        # at least two peaks present, split into 1+ regions
        temp = [rounding*np.floor((peak_list_trim[0] - window)/rounding), rounding*np.floor((peak_list_trim[0] + window)/rounding)]
        for peak in peak_list_trim:
            # check each peak to see if it falls within window of current region
            if rounding*np.floor((peak)/rounding) > temp[1]:
                # peak falls outside current region, save current region to regions list
                regions.append(temp)
                # create new region centered on peak
                temp = [rounding*np.floor((peak - window)/rounding), rounding*np.ceil((peak + window)/rounding)]
            else:
                # peak falls inside current region, extend region to include it
                temp[-1] = rounding*np.ceil((peak + window)/rounding)
        regions.append(temp)
    if debug == True:
        print("    "*(inset+1), regions)
    return np.asarray(regions)

def fit_peaks(measurement, x_key='raman_shift', y_key='y_av_sub', output_key='fitted_peaks', function='pv', peak_positions=[], max_shift=20, max_fwhm=150, min_fwhm=5, vary_baseline=True, region_window=150., x_start=None, x_end=None, SNR_threshold=10, norm_threshold=0.05, min_sep=25, noise=None, noise_region=(2400,2500), plot=False, show_plot=True, plot_name=None, save_to_file=True, log=None, inset=0, debug=False):
    # script for finding peaks in a spectrum
    # argument summary:
    #   - measurement:      Measurement object containing all data
    #   - function:         function to use for fitting peaks ('g', 'l', 'pv', or 'fd')
    #   - peak_positions:   list of peak positions to use, leave empty for automatic (list of ints or floats)
    #   - max_shift:        maximum change in position allowed during fitting (int or float)
    #   - max_fwhm:         maximum full-width-half-maximum (peak width) allowed during fitting (int or float)
    #   - vary_baseline:    allow fit to change baseline parameters as part of fit? (boolean)
    #   - region_window:    max separation between peaks for including in same fit (int or float)
    #   - x_start:          start of x axis range to be processed (int or float)
    #   - x_end:            end of x axis range to be processed (int or float)
    #   - SNR_threshold:    minimum signal:noise ratio for automatic peak detection (int or float)
    #   - norm_threshold:   minimum intensity for automatic peak detection (float, 0-1)
    #   - noise:            user-defined noise level (int or float), or None for automatic noise estimation
    #   - noise_region:     user-defined region to calculate noise in (tuple of 2 ints or 2 floats)
    #   - min_sep:          minimum separation between valid peaks, in x axis units (int or float)
    #   - x_key:            key for x axis values in spec dict (string)
    #   - y_key:            key for y axis values in spec dict (string)
    #   - output_key:       key for outputting results into spec dict (string)
    #   - debug:            print debug messages? (boolean)
    #   - plot:             generate summary figure for process? (boolean) - only applies to single spectra!
    #   - show_plot:        show plot in console window? (boolean)
    #   - plot_name:        suffix to use when naming figure file (string)
    
    # ==================================================
    # do input checks, raise Exceptions if needed
    print()
    print("    "*(inset) + "running peak fitting on measurement %s" % measurement.title)
    if isinstance(measurement, Measurement) == False:
        raise Exception("    "*(inset+1) + "first argument passed to subtract_baseline function must be a Measurement instance!")
    elif hasattr(measurement, x_key) == False:
        raise Exception("    "*(inset+1) + "%s not in measurement data!" % x_key)
    elif hasattr(measurement, y_key) == False:
        raise Exception("    "*(inset+1) + "%s not in measurement data!" % y_key)
    if noise != None and type(noise) not in [int, float, np.int16, np.int32, np.float32, np.float64]:
        raise Exception("    "*(inset+1) + "noise argument passed to find_peaks must be integer or float!")
    elif type(noise) in [int, float, np.int16, np.int32, np.float32, np.float64]:
        if noise <= 0:
            raise Exception("    "*(inset+1) + "noise argument passed to find_peaks must be greater than 0!")
    if plot == True and plot_name == None:
        plot_name = "peak-fitting"
    if type(log) != str:
        log = 'automatic peak fitting'
    
    # ==================================================
    # get data from measurement object
    x, y = measurement(x_key, y_key)
    old_log = measurement[y_key].log
    indices = measurement[y_key].indices
    if debug == True:
        print("    "*(inset+1) + "input arrays:", x_key, np.shape(x), y_key, np.shape(y))
        print("    "*(inset+1) + "previous y processing:", "\n".join(old_log))
    # smooth data for fitting
    if y.ndim == 1:
        # convert single-spec y to 2D array for handling
        single_spec = True
        y_s = y[:,np.newaxis]
    elif np.shape(y)[1] == 1:
        # y array is 2D but contains single spectrum
        single_spec = True
        y_s = y
    else:
        # y array is 2D contains multiple spectra to process individually
        single_spec = False
        y_s = y
        if plot == True:
            print("    "*(inset+1) + "cannot plot results of peak fitting for multi-spec measurement!")
            plot = False
    if x_start == None:
        x_start = np.amin(x)
    if x_end == None:
        x_end = np.amax(x)
        
    # ==================================================
    # determine which function to use
    if function.lower() in ['pv', 'pseudo-voigt', 'pseudo voigt']:
        function = 'pv'
        function_title = 'Pseudo-Voigt'
    elif function.lower() in ['fd', 'fermi-dirac', 'fermi dirac']:
        function = 'fd'
        function_title = 'Fermi-Dirac'
    elif function.lower() in ['l', 'lorentz', 'lorentzian']:
        function = 'l'
        function_title = 'Lorentzian'
    else:
        function = 'g'
        function_title = 'Gaussian'
    if debug == True:
        print("    "*(inset+1) + "fitting function:", function_title)
    
    # ==================================================
    # create arrays to store results
    input_peaks = [[] for i in range(np.shape(y)[1])]
    peak_data = {'spec_index': [], 'function': [], 'centers': [], 'centers_err': [], 'amplitudes': [], 'amplitudes_err': [], 'fwhm': [], 'fwhm_err': [], 'heights': [],}
    # add more arrays depending on chosen function
    if function == 'pv':
        peak_data['sigmas'] = []
        peak_data['sigmas_err'] = []
        peak_data['etas'] = []
        peak_data['etas_err'] = []
    elif function == 'fd':
        peak_data['rounds'] = []
        peak_data['rounds_err'] = []
        peak_data['widths'] = []
        peak_data['widths_err'] = []
    elif function == 'l':
        peak_data['gammas'] = []
        peak_data['gammas_err'] = []
    else:
        peak_data['sigmas'] = []
        peak_data['sigmas_err'] = []
        
    # ==================================================
    # determine which input peak positions to use for fitting
    if len(peak_positions) > 0:
        # use manually specified peaks
        if debug == True:
            print("    "*(inset+1) + "using manually specified peak positions:")
            print("    "*(inset+2) + "all spectra:", ", ".join(["%0.f" % peak for peak in peak_positions]))
        # pass list of peaks that are inside range x_start - x_end
        temp = np.asarray([peak for peak in peak_positions if x_start <= peak and peak <= x_end])
        for i in range(np.shape(y)[1]):
            # ensures each input spectrum gets its own copy of the list
            input_peaks[i] = temp
    else:
        # use automatic peak detection to get peaks
        if hasattr(measurement, 'detected_peaks'):
            # use previously detected peaks
            if debug == True:
                print("    "*(inset+1) + "no peaks specified, using previously detected peak positions:")
            for i in range(np.shape(y)[1]):
                # ensures each input spectrum gets its own list from detected_peaks
                sort = measurement['detected_peaks']['spec_index'].values == indices[i]
                temp = np.asarray(measurement['detected_peaks'].loc[sort,'centers'])
                print(temp)
                print(np.shape(temp))
                if debug == True:
                    print("    "*(inset+2) + "spec %s:" % i, ", ".join(["%0.f" % peak for peak in temp]))
                input_peaks[i] = temp
        else:
            # no peak detection recorded, do it now
            if debug == True:
                print("    "*(inset+1) + "no peaks specified, doing automatic peak detection")
            # run detect_peaks() using input variables
            detect_peaks(measurement, x_key, y_key, 'detected_peaks', SNR_threshold=SNR_threshold, norm_threshold=norm_threshold, min_sep=min_sep, noise=noise, noise_region=noise_region, x_start=x_start, x_end=x_end, plot=plot, inset=inset+1, debug=debug)
            for i in range(np.shape(y)[1]):
                # ensures each input spectrum gets its own list from detected_peaks
                sort = measurement['detected_peaks']['spec_index'].values == indices[i]
                temp = np.asarray(measurement['detected_peaks'].loc[sort,'centers'])
                if debug == True:
                    print("    "*(inset+2) + "spec %s:" % i, ", ".join(["%0.f" % peak for peak in temp]))
                input_peaks[i] = temp
    
    # ==================================================
    # estimate noise (for calculating SNR)
    noise = get_noise(x, y, start=noise_region[0], end=noise_region[1], inset=inset+1, debug=debug)
    
    # ==================================================
    # proceed with fitting
    for i in range(np.shape(y)[1]):
        # for each spectrum
        peaks = input_peaks[i]
        if debug == True:
            print()
            print("    "*(inset+1) + "fitting spectrum %s..." % i)
        if len(peaks) == 0:
            print("    "*(inset+2) + "no input peaks!")
        else:
            # at least one peak found, proceed
            
            # calculate noise for this spectrum
            noise = get_noise(x, y_s[:,i], start=noise_region[0], end=noise_region[1], inset=inset+2, debug=debug)
            if debug == True:
                print("    "*(inset+2) + "noise level: %0.1f" % noise)

            # split into regions
            regions = fitting_regions(input_peaks[i], start=x_start, end=x_end, window=region_window, inset=inset+2)
            if debug == True:
                print("    "*(inset+2) + "spec %s divided into %s regions:" % (i, len(regions)), ", ".join(["(%0.1f-%0.1f)" % (reg[0], reg[1]) for reg in regions]))
                
            for region in regions:
                # ==================================================
                # for each region in spectrum
                start, end = region
                if debug == True:
                    print()
                    print("    "*(inset+2) + "fitting region %0.f - %0.f cm-1" % (start, end))

                # pick out only those peaks in this region
                reg_peaks = peaks[np.where((start < peaks) & (peaks < end))]
                if debug == True:
                    print("    "*(inset+3) + "%s peaks in region:" % len(reg_peaks), reg_peaks)

                # slice spectrum to region
                x_slice, y_slice = slice_spectrum(x, y_s[:,i], start=start, end=end, inset=inset+3)

                # proceed with peak fit
                fit_output, fit_curve = peak_fit_script(x_slice, y_slice, reg_peaks, function=function, max_shift=max_shift, min_fwhm=min_fwhm, max_fwhm=max_fwhm)
                
                if debug == True and fit_output.params['center_0'].stderr == None:
                    print("    "*(inset+3) + "could not estimate standard errors")

                # ==================================================
                # convert results to numpy arrays
                for i2, peak in enumerate(reg_peaks):
                    # add parameters to storage array
                    peak_data['spec_index'].append(indices[i])
                    peak_data['function'].append(function)
                    peak_data['heights'].append(fit_output.params['amplitude_%s' % i2].value + fit_output.params['gradient'].value * fit_output.params['center_%s' % i2].value + fit_output.params['intercept'].value)
                    for prop in ['center', 'amplitude', 'sigma', 'gamma', 'eta', 'width', 'round']:
                        key = prop+"_%s" % i2
                        if key in fit_output.params.keys():
                            peak_data["%ss" % prop].append(fit_output.params[key].value)
                            if fit_output.params[key].stderr != None:
                                peak_data[prop+"s_err"].append(fit_output.params[key].stderr)
                            else:
                                peak_data[prop+"s_err"].append(0.)
                    # get FWHM and error
                    peak_data['fwhm'].append(fwhm(fit_output.params, i2, function))
                    peak_data['fwhm_err'].append(fwhm_err(fit_output.params, i2, function))

                # ==================================================
                # create region fit figures if required
                if plot == True and i == 0:
                    # for each region in spectrum 0
                    # prepare figure
                    plt.figure(figsize=(8,6))
                    # ax1: results of fit
                    ax1 = plt.subplot2grid((4,5), (0,0), colspan=4, rowspan=3)
                    ax1.set_title("%s\n%0.f-%0.f cm$^{-1}$ %s Peak Fitting" % (measurement.title, start, end, function_title))
                    ax1.set_ylabel("Average Intensity")
                    # ax2: residuals
                    ax2 = plt.subplot2grid((4,5), (3,0), colspan=4, sharex=ax1)
                    ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
                    ax2.set_ylabel("Residual")
                    # histogram of residuals
                    ax3 = plt.subplot2grid((4,5), (3,4))
                    ax3.set_yticks([])
                    # determine y limits for residual, hist plots
                    y_min = np.amin(y_slice-fit_curve)
                    y_max = np.amax(y_slice-fit_curve)
                    res_min = y_min - 0.1*(y_max-y_min)
                    res_max = y_max + 0.1*(y_max-y_min)
                    ax2.set_ylim(res_min, res_max)
                    # plot input data and residuals
                    ax1.plot(x_slice, y_slice, 'k')
                    ax2.plot(x_slice, y_slice-fit_curve, 'k')
                    ax3.hist(y_slice-fit_curve, range=(res_min, res_max), bins=20, orientation='horizontal', color='k')
                    x_temp = np.linspace(start, end, 10*len(x_slice))
                    # plot total peak fit
                    if function == 'pv':
                        total_curve = multiPV_curve(x_temp, fit_output.params, reg_peaks)
                    elif function == 'fd':
                        total_curve = multiFD_curve(x_temp, fit_output.params, reg_peaks)
                    elif function == 'l':
                        total_curve = multiL_curve(x_temp, fit_output.params, reg_peaks)
                    else:
                        total_curve = multiG_curve(x_temp, fit_output.params, reg_peaks)
                    ax1.plot(x_temp, total_curve, 'b--')
                    # determine y axis limits
                    y_max = np.amax(y_slice)
                    y_min = np.amin([-0.2*y_max, np.amin(y_slice), np.amin(total_curve)])
                    ax1.set_xlim(start, end)
                    ax1.set_ylim(y_min, 1.2*y_max)
                    # plot individual peak fits
                    for i2, peak in enumerate(reg_peaks):
                        # plot and report peak positions
                        ax1.text(fit_output.params["center_%s" % i2]+0.01*(end-start), y_min+0.95*(1.2*y_max-y_min), i2+1)
                        plt.figtext(0.82, 0.93-0.08*i2, "Center %s: %.1f" % (i2+1, fit_output.params["center_%s" % i2]))
                        plt.figtext(0.82, 0.9-0.08*i2, "  FWHM %s: %.1f" % (i2+1, fwhm(fit_output.params, i2, function)))
                        ax1.axvline(fit_output.params["center_%s" % i2], color='k', linestyle=':')
                        # create function curve for plotting this specific peak
                        params = lmfit.Parameters()
                        params.add('gradient', value=fit_output.params["gradient"])
                        params.add('intercept', value=fit_output.params["intercept"])
                        params.add('amplitude_0', value=fit_output.params["amplitude_%s" % i2])
                        params.add('center_0', value=fit_output.params["center_%s" % i2])
                        if function == 'pv':
                            params.add('sigma_0', value=fit_output.params["sigma_%s" % i2])
                            params.add('eta_0', value=fit_output.params["eta_%s" % i2])
                            peak_curve = multiPV_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                        elif function == 'fd':
                            params.add('width_0', value=fit_output.params["width_%s" % i2])
                            params.add('round_0', value=fit_output.params["round_%s" % i2])
                            peak_curve = multiFD_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                        elif function == 'l':
                            params.add('gamma_0', value=fit_output.params["gamma_%s" % i2])
                            peak_curve = multiL_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                        else:
                            params.add('sigma_0', value=fit_output.params["sigma_%s" % i2])
                            peak_curve = multiG_curve(x_temp, params, [fit_output.params["center_%s" % i2]])
                        ax1.plot(x_temp, peak_curve, 'b:')
                    # finish figure
                    ax1.minorticks_on()
                    ax2.minorticks_on()
                    ax3.minorticks_on()
                    plt.tight_layout()
                    # save to file
                    figdir = "%s%s_%0.f-%0.fcm_%s-fit.png" % (measurement.fig_dir, measurement.title, start, end, function.upper())
                    if debug == True:
                        print("    "*(inset+3) + "saving figure to %s" % figdir)
                    plt.savefig(figdir, dpi=300)
                    if show_plot == True:
                        plt.show()
                    else:
                        plt.close()
                
    # ==================================================
    # convert spectrum results to pandas dataframe
    peak_data = pd.DataFrame(peak_data)
    
    print("    "*(inset) + "total %s peaks fitted across %s spectra" % (len(peak_data['spec_index']), len(np.unique(peak_data['spec_index']))))
    
    # ==================================================
    # add results to measurement object
    setattr(measurement, output_key, peak_data)
    if debug == True:
        print("    "*(inset) + "output %s array:" % output_key)
        print(peak_data.info())
        print("    "*(inset+1) + "output log:", old_log + [log])
        
    # save fitted peaks to file
    if len(peak_data['centers']) > 0 and save_to_file == True:
        print("    "*(inset) + "saving peak fit data to file")
        # save to file
        outdir = "%s%s_fitted-peaks.csv" % (measurement.out_dir, measurement.title)
        if debug == True:
            print("    "*(inset+1) + "saving to %s" % outdir)
        # save data to output folder
        peak_data.to_csv(outdir)
    
    # ==================================================
    # create summary figure if required
    if plot == True:
        # prepare figure
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)
        ax1.set_title("%s\nAutomatic Peak Fitting" % (measurement.title))
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax1.set_ylabel("Intensity (counts)")
        # plot spectrum
        x, y = measurement(x_key, y_key)
        y = measurement[y_key].mean()
        y_max = np.amax(y)
        y_min = np.amin(y)
        plt.plot(x, y, 'k', label=y_key)
        # plot fitted peak positions and intensities
        x = peak_data['centers']
        y = peak_data['heights']
        ax1.plot(x, y, 'bo', label='fitted peaks')
        if debug == True:
            print("    "*(inset+1) + "fitted peaks:")
        for i2 in range(len(x)):
            # report detected positions
            ax1.text(x[i2], y[i2]+0.05*(y_max-y_min), "%0.f" % x[i2], rotation=90, va='bottom', ha='left')
            if debug == True:
                print("    "*(inset+2) + "%0.f: %0.2f" % (x[i2], y[i2]/y_max))
        # get axis limits
        ax1.set_xlim(x_start, x_end)
        ax1.set_ylim(-0.2*y_max, 1.2*y_max)
        if noise > 0:
            # create second y axis for SNR values
            ax2 = ax1.twinx()
            ax2.set_ylim(intensity2snr(ax1.get_ylim(), noise))
            ax2.set_ylabel("SNR")
        # finish figure
        ax1.legend()
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig("%s%s_%s.png" % (measurement.fig_dir, measurement.title, plot_name), dpi=300)
        if show_plot == True:
            plt.show()
        else:
            plt.close()
    
    # ==================================================
    # end of function
    if debug == True:
        print("    "*(inset) + "end of fit_peaks() function")
        
"""
# ==================================================
# functions for creating generic plots
# ==================================================
"""

    
"""
# ==================================================
# functions for saving measurements to file
# ==================================================
"""

def save_measurement(measurement, keys=[], headers=[], start=None, end=None, do_not_include=[], save_name=None, inset=0, debug=False):
    # script for saving a given set of spectra from a measurement
    # argument summary:
    #   - measurement:      Measurement instance containing data
    #   - *args:            series of spectrum keys to add to save file (strings)
    #   - save_name:        suffix to use when naming figure file (string)
    global Spectrum
    # ==================================================
    # do input checks, raise Exceptions if needed
    if debug == True:
        print()
        print("    "*(inset) + "saving measurement %s to file" % measurement.title)
        print("    "*(inset+1) + "spectra to include:", ", ".join(keys))
    check = [hasattr(measurement, key) for key in keys]
    if np.any(check) == False:
        raise Exception("    "*(inset+1) + "the following keys are missing from measurement:", np.asarray(keys)[check])
    if len(keys) != len(headers):
        raise Exception("    "*(inset+1) + "header list passed to save_measurement() does not match key list length!")
    if save_name == None:
        save_name = "av-spectrum"
    if debug == True:
        print("    "*(inset+1) + "file suffix: %s.csv" % save_name)
    
    # ==================================================
    # gather spectral data
    header_info = []
    save_data = []
    for key, header in zip(keys, headers):
        # for each key in list, get array from matching property
        if key in ['x', 'raman_shift', 'wavelength', 'frequency']:
            # ==================================================
            # desired array is a direct property of Measurement object
            header_info.append(header)
            save_data.append(measurement[key][:,np.newaxis])
            if debug == True:
                print("    "*(inset+2), header, np.shape(save_data[-1]))
        elif key[-5:] == '_norm' and hasattr(measurement[key[:-5]], 'y') == True:
            # ==================================================
            # desired array is a normalised Spectrum object
            x,y = measurement('x', key)
            indices = measurement[key[:-5]].indices
            if np.shape(y)[1] > 1:
                # more than 1 spectrum, each needs its own header
                for i in range(np.shape(y)[1]):
                    header_info.append("%s spec %s" % (header, indices[i]))
            else:
                # single spectrum, single header
                header_info.append(header)
            save_data.append(y)
            if debug == True:
                print("    "*(inset+2), header, np.shape(save_data[-1]))
        elif hasattr(measurement[key], 'y') == True:
            # ==================================================
            # desired array is inside a Spectrum object within Measurement
            x,y = measurement('x', key)
            indices = measurement[key].indices
            if np.shape(y)[1] > 1:
                # more than 1 spectrum, each needs its own header
                for i in range(np.shape(y)[1]):
                    header_info.append("%s spec %s" % (header, indices[i]))
            else:
                # single spectrum, single header
                header_info.append(header)
            save_data.append(y)
            if debug == True:
                print("    "*(inset+2), header, np.shape(save_data[-1]))
        else:
            # ==================================================
            # key is not a valid data type, skip
            if debug == True:
                print("    "*(inset+2) + "%s is not a valid key for measurement %s" % (key, measurement.ID))
            pass
        
    # ==================================================
    # convert save_data to pandas DataFrame and save to CSV
    save_data = pd.DataFrame(np.hstack(save_data), columns=header_info)
    if debug == True:
        print("    "*(inset) + "resulting dataframe:")
        print(save_data.info())
    save_data.to_csv("%s%s_%s.csv" % (measurement.out_dir, measurement.title, save_name), index=False)
    
    # ==================================================
    # generate metadata frame and save to CSV
    if debug == True:
        print()
        print("    "*(inset) + "generating metadata file for %s" % measurement.title)
    headers = []
    values = []
    # look up most important info
    for header in ['ID', 'title', 'sample', 'subsample', 'notes', 'measurement_date', 'filename', 'technique', 'laser_wavelength', 'laser_power', 'accumulations', 'exposure_time']:
        if hasattr(measurement, header):
            headers.append(header)
            values.append(measurement[header])
            if debug == True:
                print("    "*(inset+1), header, "=", measurement[header])
    for header in measurement.__dict__.keys():
        # for all other attributes of measurement
        if header in ['x', 'y', 'raman_shift', 'frequency', 'wavelength', 'x_coords', 'y_coords'] or hasattr(measurement[header], 'y'):
            # attribute is a spectrum array, ignore
            if debug == True:
                print("    "*(inset+1), header, "= spectral data, skipping")
        elif header not in headers and header not in do_not_include:
            # add to array
            headers.append(header)
            values.append(measurement[header])
            if debug == True:
                print("    "*(inset+1), header, "=", measurement[header])
    metadata = pd.DataFrame(np.asarray([headers, values]).transpose(), columns=['Parameter', 'Value'])
    if debug == True:
        print("    "*(inset) + "resulting dataframe:")
        print(metadata.info())
    metadata.to_csv("%s%s_metadata.csv" % (measurement.out_dir, measurement.title), index=False)
    
    # ==================================================
    # end of function
    if debug == True:
        print("    end of save_measurement()")
