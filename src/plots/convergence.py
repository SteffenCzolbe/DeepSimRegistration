import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing import event_accumulator

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

# read datasets
datasets = os.listdir('./weights/')

# set up sup-plots
fig = plt.figure()
axs = fig.subplots(1, len(datasets)) 

for i, dataset in enumerate(datasets):
    loss_functions = sorted(os.listdir(os.path.join('./weights/', dataset, 'registration')))
    for loss_function in loss_functions:
        x, y = read_tb_scalar_log(os.path.join('./weights/', dataset, 'registration', loss_function), 'train/dice_overlap')
        y = smooth(y, 0.95)
        axs[i].plot(x, y, label=loss_function)
        axs[i].set_title(dataset)

# add labels
fig.text(0.5, 0.04, 'Gradient Updates', ha='center', va='center')
fig.text(0.06, 0.5, 'Mean Dice Overlap', ha='center', va='center', rotation='vertical')
axs[-1].legend(loc='lower right')

plt.savefig('./src/plots/convergence.pdf')
plt.savefig('./src/plots/convergence.png')