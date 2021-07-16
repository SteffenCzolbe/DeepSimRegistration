import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
from .config import *
from matplotlib.ticker import FormatStrFormatter


# read logs
def read_tb_scalar_logs(dir, scalar):
    """
    searches for the tensorboard log in directory dir, and reads sclar log called scalar.
    """
    files = os.listdir(dir)
    log_files = list(
        map(lambda f: os.path.join(dir, f), filter(
            lambda s: "events.out" in s, files))
    )

    # sort by timestamp, newer first (newer logs overwrite older ones in case of overlaps)
    log_files = list(reversed(sorted(log_files)))
    records = []
    max_step = 39000
    for log_file in log_files:
        records += list(
            reversed(read_tb_scalar_log(log_file, scalar, max_step=max_step))
        )
        max_step = records[-1][0] - 1

    # back to ascending order
    records = list(reversed(records))

    # extract x, y for plotting
    x = list(map(lambda t: t[0], records))
    y = list(map(lambda t: t[1], records))
    return x, y


# read logs
def read_tb_scalar_log(file, scalar, max_step=None):
    """
    reads the tensorboard log.
    """
    ea = event_accumulator.EventAccumulator(
        file,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1,
        },
    )
    ea.Reload()  # loads events from file
    #step_to_epoch = ea.Scalars("epoch")
    #step_to_epoch = dict([(event.step, event.value) for event in step_to_epoch])

    events = ea.Scalars(scalar)

    records = []
    for event in events:
        if max_step is None or event.step < max_step:
            records.append((event.step, event.value))
        else:
            break

    return records


def smooth(ys, smoothing_factor=0.6):
    new_y = [ys[0]]

    for y in ys[1:]:
        new_y.append(new_y[-1] * smoothing_factor + (1 - smoothing_factor) * y)
    return new_y


# set up sup-plots
fig = plt.figure(figsize=(8.5, 2.5))
axs = fig.subplots(1, len(DATASET_ORDER))
plt.subplots_adjust(bottom=0.18)

for i, dataset in enumerate(DATASET_ORDER):
    axs[i].set_title(PLOT_CONFIG[dataset]["display_name"])
    for loss_function in tqdm(
        LOSS_FUNTION_ORDER, desc=f"plotting convergence of loss-finctions on {dataset}"
    ):
        path = os.path.join("./weights/", dataset,
                            "registration", loss_function)
        if not os.path.isdir(path):
            continue
        x, y = read_tb_scalar_logs(path, "train/dice_overlap")
        y = smooth(y, PLOT_CONFIG[dataset]["smoothing_factor"])
        c = LOSS_FUNTION_CONFIG[loss_function]["primary_color"]
        line = axs[i].plot(
            x, y, color=c, linewidth=2
        )  # some trickery required to show lines on the legend of subplots not containing them
        LOSS_FUNTION_CONFIG[loss_function]["handle"] = line[0]

# add labels
fig.text(0.5, 0.03, "Gradient Update Steps",
         ha="center", va="center", fontsize=16)
fig.text(0.07, 0.5, "Train Mean Dice Overlap", ha="center",
         va="center", rotation="vertical", fontsize=16)
handles = [
    LOSS_FUNTION_CONFIG[loss_function]["handle"] for loss_function in LOSS_FUNTION_ORDER
]
labels = [
    LOSS_FUNTION_CONFIG[loss_function]["display_name"]
    for loss_function in LOSS_FUNTION_ORDER
]
axs[-1].legend(handles, labels, loc="lower right", fontsize="small")

# configure precision
for ax in axs:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

os.makedirs("./out/plots/", exist_ok=True)
plt.savefig("./out/plots/convergence.pdf")
plt.savefig("./out/plots/convergence.png")
