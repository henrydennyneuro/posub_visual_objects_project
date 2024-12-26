import os
import pickle
import numpy as np
import pandas as pd
import pynapple as nap
import nwbmatic as ntm
import pynacollada as pyna
import matplotlib.pyplot as plt

from pathlib import Path

if __name__ == '__main__':

    """
        This script is designed to link image stims to their corresponding TTL. 

        Set up path for loading metadata files.
    """    
    
    folder = input("Enter file path: ")
    
    directory = os.path.dirname(folder)

    path_string = Path(folder)
    path_parts = path_string.parts
    recording_basename = path_parts[4]

    """
        Load metadata: TTL peak intervals, stimuli epochs, and stimuli order.
    """    

    TTL_intervals_filename = f'{directory}\\{path_parts[4]}\\{path_parts[4]}-TTL_IntervalSet.npz'
    TTL_interval_file = np.load(TTL_intervals_filename)
    TTL_interval_set = nap.IntervalSet(TTL_interval_file['start'], TTL_interval_file['end'])

    stimuli_epochs_filename = f'{directory}\\{path_parts[4]}\\{path_parts[4]}-Stimuli_IntervalSet.npz'
    stimuli_epochs_file = np.load(stimuli_epochs_filename)
    stimuli_epochs = nap.IntervalSet(stimuli_epochs_file['start'], stimuli_epochs_file['end'])

    vis1_filename = f'{directory}\\{path_parts[4]}\\{path_parts[3]}_v1_stimuli_list.csv'
    PoS_filename = f'{directory}\\{path_parts[4]}\\{path_parts[3]}_PoS_stimuli_list.csv'

    is_vis1_file = os.path.isfile(vis1_filename)
    is_PoS_file = os.path.isfile(PoS_filename)

    if is_PoS_file == True:
        stimuli_order = pd.read_csv(PoS_filename, header = None)
        print('Found csv file for Post-Subiculum recording')
    elif is_vis1_file == True:
        stimuli_order = pd.read_csv(vis1_filename, header = None)
        print('Found csv file for Visual Cortex 1 recording')
    else:
        raise Exception("Stimuli order file not found.")

    """
        Now we can create peri-event histograms for each stim type. Start by loading in the neural data.
    """    

    data = ntm.load_session(folder, "neurosuite")

    frequency = 1250
    headfix_epoch = data.epochs['Headfix']
    spikes = data.spikes.restrict(headfix_epoch)

    # analogin_start_time = stimuli_epoch.starts.times()

    """
        Extract the TTL epochs in the Image Stimuli epoch of the recording. Get baseline firing rate epochs (periods outside
        stims). 

        I wrote some code here to reset the TTL times to start from the end of the exploration epoch, rather than 0. This has
        since been moved to SmoothTTLs.py in the PreprocessingInPynapple repository, but i've kept it here in case that change
        proves buggy. 
    """    
    
    image_TTLs = TTL_interval_set.intersect(stimuli_epochs[1]) #nap.IntervalSet(stimuli_epochs['start'][1], stimuli_epochs['end'][1]))
    
    # TTLs_start_times = image_TTLs.starts.times(units = 's')
    # TTLs_end_times = image_TTLs.ends.times(units = 's')

    # corrected_TTLs_start_times = TTLs_start_times + analogin_start_time
    # corrected_TTLs_end_times = TTLs_end_times + analogin_start_time
    
    # corrected_TTLs_dataframe = pd.DataFrame({'start' : np.transpose(corrected_TTLs_start_times), 'end' : np.transpose(corrected_TTLs_end_times)})
    # corrected_TTLs_interval_set = nap.IntervalSet(start = corrected_TTLs_dataframe['start'], end = corrected_TTLs_dataframe['end'], time_units = 's')

    """
        Get the firing rates of your neurons during baseline (inter-TTL epochs).
    """    

    baseline = stimuli_epochs[1].set_diff(image_TTLs)
    baseline_firing_rate = spikes.restrict(baseline)
    baseline_firing_rate_nd_array = baseline_firing_rate.rates.to_numpy()

    """
        Check that the number of stims found in the csv is equivalent to the number of TTLs found in the image epoch.
    """    

    if len(image_TTLs) == len(stimuli_order):
        print("Number of images in stimuli_order file match number of TTL's found in the image epoch")
    else:
        raise Exception("Number of images found in csv does not match the number of TTL's found in the image epoch")

    """
        The plot below overlays the image epoch ttl times over all ttl times. This allows you to double check ttls are for
        the correct times.
    """    

#     plt.figure()
#     plt.scatter(TTL_interval_set.starts.times(), (TTL_interval_set.ends.times() - TTL_interval_set.starts.times()))
#     plt.scatter(image_TTLs.starts.times(), (image_TTLs.ends.times() - image_TTLs.starts.times()))
#     plt.axvline(stimuli_epochs.starts.times()[1])
#     plt.title.set_text("Distribution of TTL Peak Durations")
#     plt.show()


    """
        Pool stims into "diffeo", "texture", and "regular" groups by appending label to each stim.
    """    

    stimuli_category = []

    for index, row in stimuli_order.iterrows():
        #print(index)
        if "diffeo" in stimuli_order[0].loc[index]:
            stimuli_category.append('diffeo')
        elif "texture" in stimuli_order[0].loc[index]:
            stimuli_category.append('texture')
        else:
            stimuli_category.append('regular')


    stimuli_category_as_series = pd.Series(stimuli_category)
    stimuli_order['type'] = stimuli_category_as_series.values

    """
        Create IntervalSets for each category.
    """    

    diffeo_stims_indexes = stimuli_order.index[stimuli_order['type'] == 'diffeo'].tolist()
    texture_stims_indexes = stimuli_order.index[stimuli_order['type'] == 'texture'].tolist()
    regular_stims_indexes = stimuli_order.index[stimuli_order['type'] == 'regular'].tolist()
    
    diffeo_intervals = image_TTLs[diffeo_stims_indexes]
    texture_intervals = image_TTLs[texture_stims_indexes]
    regular_intervals = image_TTLs[regular_stims_indexes]

    diffeo_centers = diffeo_intervals.get_intervals_center()
    texture_centers = texture_intervals.get_intervals_center()
    regular_centers = regular_intervals.get_intervals_center()

    """
        Compute peri-events for each cell across each stim type.
    """    

    # mean_baseline_firing_rate = np.mean(baseline_firing_rate.get_info('rate'))

    diffeo_perievent = nap.compute_perievent(spikes, diffeo_centers, (-1000, 1000), 'ms')
    texture_perievent = nap.compute_perievent(spikes, texture_centers, (-1000, 1000), 'ms')
    regular_perievent = nap.compute_perievent(spikes, regular_centers, (-1000, 1000), 'ms')

    fig, axs = plt.subplots(10, 8)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(np.mean(diffeo_perievent[i].count(0.05, time_units = 's')/0.05, 1), color = 'green') #/baseline_firing_rate.get_info('rate')[i]
        ax.plot(np.mean(texture_perievent[i].count(0.05, time_units = 's')/0.05, 1), color = 'blue')
        ax.plot(np.mean(regular_perievent[i].count(0.05, time_units = 's')/0.05, 1), color = 'red')
        ax.axvline(-0.25, color = 'black')
        ax.axvline(0.25, color = 'black')
    fig.legend(["Diffeo", "Textures", "Regular"], loc = "center right")
    plt.show()


    """
        Normalise firing rates in the peri-event histograms.
    """    

    normalised_diffeo_perievents = []
    normalised_texture_perievents = []    
    normalised_regular_perievents = []

    for count, value in enumerate(spikes):
        normalised_diffeo_perievents.append(np.mean(diffeo_perievent[count].count(0.05, time_units = 's')/0.05, 1)/baseline_firing_rate.get_info('rate')[count])
        normalised_texture_perievents.append(np.mean(texture_perievent[count].count(0.05, time_units = 's')/0.05, 1)/baseline_firing_rate.get_info('rate')[count])
        normalised_regular_perievents.append(np.mean(regular_perievent[count].count(0.05, time_units = 's')/0.05, 1)/baseline_firing_rate.get_info('rate')[count])
       
    fig, axs = plt.subplots(10, 4)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(normalised_diffeo_perievents[i], color = 'green') 
        ax.plot(normalised_texture_perievents[i], color = 'blue')
        ax.plot(normalised_regular_perievents[i], color = 'red')
        ax.axvline(-0.25, color = 'black')
        ax.axvline(0.25, color = 'black')
    fig.legend(["Diffeo", "Textures", "Regular"], loc = "center right")
    plt.show()

    normalised_diffeo_MUA_sum = np.zeros(np.shape(normalised_diffeo_perievents[0]))
    normalised_texture_MUA_sum = np.zeros(np.shape(normalised_texture_perievents[0]))
    normalised_regular_MUA_sum = np.zeros(np.shape(normalised_regular_perievents[0]))

    for count, value in enumerate(spikes):
        for iterator in np.arange(len(normalised_diffeo_perievents[count])):
            normalised_regular_MUA_sum[iterator] = np.add(normalised_regular_MUA_sum[iterator], normalised_regular_perievents[count][iterator])
            normalised_texture_MUA_sum[iterator] = np.add(normalised_texture_MUA_sum[iterator], normalised_texture_perievents[count][iterator])
            normalised_diffeo_MUA_sum[iterator] = np.add(normalised_diffeo_MUA_sum[iterator], normalised_diffeo_perievents[count][iterator])

    normalised_diffeo_MUA = normalised_diffeo_MUA_sum/len(spikes)
    normalised_texture_MUA = normalised_texture_MUA_sum/len(spikes)
    normalised_regular_MUA = normalised_regular_MUA_sum/len(spikes)

    """
        I wrote some code for smoothing to see if it made the data easier to interpret by eye, but it didn't really.
    """    

    # smoothing_kernel = 3

    # smoothed_normalised_diffeo_MUA = pd.DataFrame(normalised_diffeo_MUA_sum).rolling(window = smoothing_kernel).mean()
    # smoothed_normalised_texture_MUA = pd.DataFrame(normalised_texture_MUA_sum).rolling(window = smoothing_kernel).mean()
    # smoothed_normalised_regular_MUA = pd.DataFrame(normalised_regular_MUA_sum).rolling(window = smoothing_kernel).mean()

    plt.figure()
    plt.plot(normalised_diffeo_MUA, color = 'green')
    plt.plot(normalised_texture_MUA, color = 'blue')
    plt.plot(normalised_regular_MUA, color = 'red')
    plt.axvline(25, color = 'black')
    plt.axvline(15, color = 'black')
    plt.legend(["Diffeo", "Textures", "Regular"], loc = "center right")
    plt.show()


    """
        Calculate the variance from baseline for each cell. This will allow us to quantify the change across the whole
        population
    """  

    variance_diffeo_perievents = np.empty(np.shape(normalised_diffeo_perievents))
    variance_texture_perievents = np.empty(np.shape(normalised_texture_perievents))
    variance_regular_perievents = np.empty(np.shape(normalised_regular_perievents))

    for count, value in enumerate(spikes):
        for iterator in np.arange(len(normalised_diffeo_perievents[count])):
            variance_diffeo_perievents[count][iterator] = np.sqrt((normalised_diffeo_perievents[count][iterator]-1)*(normalised_diffeo_perievents[count][iterator]-1))
            variance_texture_perievents[count][iterator] = np.sqrt((normalised_texture_perievents[count][iterator]-1)*(normalised_texture_perievents[count][iterator]-1))
            variance_regular_perievents[count][iterator] = np.sqrt((normalised_regular_perievents[count][iterator]-1)*(normalised_regular_perievents[count][iterator]-1))

    fig, axs = plt.subplots(10, 4)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(variance_diffeo_perievents[i], color = 'green') 
        ax.plot(variance_texture_perievents[i], color = 'blue')
        ax.plot(variance_regular_perievents[i], color = 'red')
        ax.axvline(15, color = 'black')
        ax.axvline(25, color = 'black')
    fig.legend(["Diffeo", "Textures", "Regular"], loc = "center right")
    plt.show()

    variance_diffeo_sum = np.zeros(np.shape(normalised_diffeo_perievents[0]))
    variance_texture_sum = np.zeros(np.shape(normalised_texture_perievents[0]))
    variance_regular_sum = np.zeros(np.shape(normalised_regular_perievents[0]))

    for count, value in enumerate(spikes):
        for iterator in np.arange(len(variance_diffeo_perievents[count])):
            variance_regular_sum[iterator] = np.add(variance_regular_sum[iterator], variance_regular_perievents[count][iterator])
            variance_texture_sum[iterator] = np.add(variance_texture_sum[iterator], variance_texture_perievents[count][iterator])
            variance_diffeo_sum[iterator] = np.add(variance_diffeo_sum[iterator], variance_diffeo_perievents[count][iterator])

    variance_diffeo_MUA = variance_diffeo_sum/len(spikes)
    variance_texture_MUA = variance_texture_sum/len(spikes)
    variance_regular_MUA = variance_regular_sum/len(spikes)

    plt.figure()
    plt.plot(variance_diffeo_MUA, color = 'green')
    plt.plot(variance_texture_MUA, color = 'blue')
    plt.plot(variance_regular_MUA, color = 'red')
    plt.axvline(25, color = 'black')
    plt.axvline(15, color = 'black')
    plt.legend(["Diffeo", "Textures", "Regular"], loc = "center right")
    plt.show()

    """
    Save the peri-event time histograms and the baseline firing rates as pickle files for further analysis.

    I also save the Image TTL's as a CSV for Dom. The Image TTL files must be a pandas dataframe to save as a CSV.
    """  

    with open(directory + '\\' + recording_basename + '\\' + recording_basename + '-Stimuli_Peri_Events' + '.pkl', 'wb') as file: 
        pickle.dump([diffeo_perievent, texture_perievent, regular_perievent, normalised_diffeo_perievents, \
            normalised_texture_perievents, normalised_regular_perievents, variance_diffeo_perievents, \
            variance_texture_perievents, variance_regular_perievents], file) 

    with open(directory + '\\' + recording_basename + '\\' + recording_basename + '-Baseline_Firing_Rates' + '.pkl', 'wb') as file: 
        pickle.dump(baseline_firing_rate_nd_array, file) 

    image_TTLs_as_dataframe = image_TTLs.as_dataframe()
    image_TTLs_as_dataframe.to_csv(f'{directory}\\{recording_basename}\\{recording_basename}-Image_TTL_IntervalSet.csv')
