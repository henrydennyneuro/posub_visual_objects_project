import numpy as np
import pandas as pd
import nwbmatic as ntm
import pynapple as nap
from matplotlib.pyplot import *

#def FindOptimalPlotDimensions(spikes)

    

def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
    
    new_tuning_curves = {}  
    for i in tuning_curves.columns:
        tcurves = tuning_curves[i]
        offset = np.mean(np.diff(tcurves.index.values))
        padded  = pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi)-offset,
                                                tcurves.index.values,
                                                tcurves.index.values+(2*np.pi)+offset)),
                            data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)      
        new_tuning_curves[i] = smoothed.loc[tcurves.index]

    new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

    return new_tuning_curves

location = input("Enter file path: ")

data = ntm.load_session(location, "neurosuite")

spikes = data.spikes
position = data.position

epoch_dictionary = {}
tuning_curves = {}
smooth_tuning_curves = {}

for epochs in np.arange(len(data.epochs)):
    epoch_dictionary['Wake'+str(epochs+1)] = data.epochs['Wake'+str(epochs+1)].intersect(position.time_support)
    tuning_curves['Wake'+str(epochs+1)] = nap.compute_1d_tuning_curves(group = spikes,
                                                                                    feature = position['ry'], 
                                                                                    ep = epoch_dictionary['Wake'+str(epochs+1)], 
                                                                                    nb_bins = 120,  
                                                                                    minmax=(0, 2*np.pi) )
    smooth_tuning_curves['Wake'+str(epochs+1)] = smoothAngularTuningCurves(tuning_curves['Wake'+str(epochs+1)])

plot_colors = {0 : 'red', 1 : 'green', 2 : 'blue'}

figure()
for i in spikes:
    subplot(8,10,i+1, projection = 'polar')
    for epochs in np.arange(len(data.epochs)):
        plot(smooth_tuning_curves['Wake'+str(epochs+1)][i], color = plot_colors[epochs])
        xticks([])
        ylabel('')
legend(['Initial', 'Cue Rotation', 'Scrambled Cue Rotation'], loc = "center right", bbox_to_anchor=(1,-1))
show()