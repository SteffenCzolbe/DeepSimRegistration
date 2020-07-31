import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing import event_accumulator
from .config import *


# read logs
def read_tb_scalar_log(dir, scalar):
    """
    searches for the tensorboard log in directory dir, and reads sclar log called scalar.
    """
    files = os.listdir(dir)
    file = list(filter(lambda s: 'events.out' in s, files))[0]

    ea = event_accumulator.EventAccumulator(os.path.join(dir, file),
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        })
    ea.Reload() # loads events from file
    events = ea.Scalars(scalar)
    x = []
    y = []
    for event in events:
        x.append(event.step)
        y.append(event.value)

    return x, y

def smooth(ys, smoothing_factor=0.6):
    new_y = [ys[0]]

    for y in ys[1:]:
        new_y.append(new_y[-1] * smoothing_factor + (1 - smoothing_factor) * y)
    return new_y


# set up sup-plots
fig = plt.figure(figsize=(10,3))
axs = fig.subplots(1, len(DATASET_ORDER)) 
plt.subplots_adjust(bottom=0.15)

for i, dataset in enumerate(DATASET_ORDER):
    axs[i].set_title(PLOT_CONFIG[dataset]['display_name'])
    for loss_function in LOSS_FUNTION_ORDER:
        path = os.path.join('./weights/', dataset, 'registration', loss_function)
        if not os.path.isdir(path):
            continue
        x, y = read_tb_scalar_log(path, 'train/dice_overlap')
        y = smooth(y, PLOT_CONFIG[dataset]['smoothing_factor'])
        c = LOSS_FUNTION_CONFIG[loss_function]['primary_color']
        line = axs[i].plot(x, y, color=c, linewidth=2) # some trickery required to show lines on the legend of subplots not containing them
        LOSS_FUNTION_CONFIG[loss_function]['handle'] = line[0]

# add labels
fig.text(0.5, 0.02, 'Gradient Updates', ha='center', va='center')
fig.text(0.06, 0.5, 'Mean Dice Overlap', ha='center', va='center', rotation='vertical')
handles = [LOSS_FUNTION_CONFIG[loss_function]['handle'] for loss_function in LOSS_FUNTION_ORDER]
labels = [LOSS_FUNTION_CONFIG[loss_function]['display_name'] for loss_function in LOSS_FUNTION_ORDER]
axs[-1].legend(handles, labels, loc='lower right')

plt.savefig('./src/plots/convergence.pdf')
plt.savefig('./src/plots/convergence.png')