import os
import pickle
import numpy as np
import pynapple as nap
import nwbmatic as ntm
import matplotlib.pyplot as plt

from pathlib import Path

if __name__ == "__main__":


    """
        This script calculates the Rayleigh r-value for each neuron. R-vector is a way to quantify the 
        directionality or preferred direction of firing rates in neural spike data.

        Set up path for loading files and saving metadata.
    """    
    
    location = input("Enter file path: ")
    
    directory = os.path.dirname(location)

    path_string = Path(location)
    path_parts = path_string.parts
    recording_basename = path_parts[4]

    """
        Load peri-event time histograms, HD tuning parameters, and baseline firing rates
    """

    perievent_filename = directory + '\\' + path_parts[4] + '\\' + path_parts[4] + '-Stimuli_Peri_Events' + '.pkl'
    perievent_file = open(perievent_filename, 'rb')
    diffeo_perievent, texture_perievent, regular_perievent, normalised_diffeo_perievents, \
            normalised_texture_perievents, normalised_regular_perievents, variance_diffeo_perievents, \
            variance_texture_perievents, variance_regular_perievents = pickle.load(perievent_file)

    HD_tuning_properties_filename = directory + '\\' + path_parts[4] + '\\' + path_parts[4] + '-HDTuning_Properties' + '.pkl'
    HD_tuning_properties_file = open(HD_tuning_properties_filename, 'rb')
    mean_vector, mean_vector_length, R_value, preferred_direction, spatial_information_as_ndarray = pickle.load(HD_tuning_properties_file)

    firing_rates_filename = directory + '\\' + path_parts[4] + '\\' + path_parts[4] + '-Baseline_Firing_Rates' + '.pkl'
    firing_rates_file = open(firing_rates_filename, 'rb')
    baseline_firing_rates = pickle.load(firing_rates_file)


    """
        Calculate differences in peak firing rate between regular images and diffeo images, and regular images and textures
    """

    reg_diffeo_peak_variance_difference = []
    reg_texture_peak_variance_difference = []

    for cells in np.arange(len(variance_regular_perievents)):
        peak_diffeo_variance = variance_diffeo_perievents[cells].max()
        peak_regular_variance = variance_regular_perievents[cells].max()
        peak_texture_variance = variance_texture_perievents[cells].max()

        diffeo_difference_in_variance = peak_regular_variance - peak_diffeo_variance
        texture_difference_in_variance = peak_regular_variance - peak_texture_variance

        reg_diffeo_peak_variance_difference.append(diffeo_difference_in_variance)
        reg_texture_peak_variance_difference.append(texture_difference_in_variance)

    """
        Plot difference in firing rates against rayleighs R value. This will tell us if the change in activity is mainly
        among HD cells or non-HD cells
    """

    b, a = np.polyfit(spatial_information_as_ndarray, reg_diffeo_peak_variance_difference, deg=1)
    xseq = np.linspace(0, 1.5, num=100)

    plt.scatter(spatial_information_as_ndarray, reg_diffeo_peak_variance_difference)
    plt.plot(xseq, a + b * xseq, color="k", lw=1.5);
    plt.title("Difference in Firing Rate Modulation (Regular - Diffeo) against Head Direction Tuning Strength")
    plt.ylabel("Difference in Firing Rate Modulation (Regular Images - Diffeo Images; AU)")
    plt.xlabel("Spatial Information (bits/spike)")
    plt.show()

    """
        Plot difference in firing rates against head direction. This will tell us if the difference in firing rates is 
        constrained to a specific head direction.

        First we must exclude any non-HD cells. We will do this by setting the preferred directions of cells with a 
        Rayleigh's value <0.15 as NaN. This will exlude them from plotting.
    """    

    R_value_threshold = 0.015
    spatial_information_threshold = 0.1

    for cell in np.arange(len(spatial_information_as_ndarray)):
        if spatial_information_as_ndarray[cell] < spatial_information_threshold:
            preferred_direction[cell] = np.nan

    plt.scatter(preferred_direction, reg_diffeo_peak_variance_difference)
    plt.title("Difference in Firing Rate Modulation (Regular - Diffeo) against Head Direction Tuning Strength")
    plt.ylabel("Difference in Firing Rate Modulation (Regular Images - Diffeo Images; AU)")
    plt.xlabel("Preferred Direction (Radiens)")
    plt.show()

    """
        Plot difference in firing rates in response to stims against baseline firing rates. This will tell us if the 
        difference in firing rates is primarily a property of interneurons or pyramidal neurons. PoS fast-spiking interneurons
        have a firing rate >15Hz
    """    
    plt.scatter(baseline_firing_rates, reg_diffeo_peak_variance_difference)
    plt.title("Difference in Firing Rate Modulation (Regular - Diffeo) against Firing Rate")
    plt.ylabel("Difference in Firing Rate Modulation (Regular Images - Diffeo Images; AU)")
    plt.xlabel("Baseline Firing Rate (Hz)")
    plt.show()


