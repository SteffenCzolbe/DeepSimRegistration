from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

from .config import *


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

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--mode', type=str, default='val', help='val or train')
    args = parser.parse_args()

    # set up sup-plots
    fig = plt.figure(figsize=(8, 3))
    axs = fig.subplots(1, len(DATASET_ORDER))
    plt.subplots_adjust(bottom=0.33, wspace=0.275)

    if args.mode == 'val':
        mode = 'val'
        title = 'Val.'
    elif args.mode == 'train':
        mode = 'train'
        title = 'Train'
    else:
        raise ValueError(f'wrong mode "{args.mode}". Use train or val')

    for i, dataset in enumerate(DATASET_ORDER):
        axs[i].set_title(PLOT_CONFIG[dataset]["display_name"])#, fontsize = 20)
        for loss_function in tqdm(
            LOSS_FUNTION_ORDER, desc=f"plotting convergence of loss-functions on {dataset}"
        ):
            path = os.path.join("./weights/", dataset,
                                "registration", loss_function)
            if not os.path.isdir(path):
                continue
            x, y = read_tb_scalar_logs(path, f"{mode}/dice_overlap")

            if dataset == 'platelet-em' and loss_function == 'mind':
                y = smooth(y, 0.999)
            else:
                y = smooth(y, PLOT_CONFIG[dataset]["smoothing_factor"])
            c = LOSS_FUNTION_CONFIG[loss_function]["primary_color"]
            line = axs[i].plot(
                x, y, color=c, linewidth=2
            )  # some trickery required to show lines on the legend of subplots not containing them
            LOSS_FUNTION_CONFIG[loss_function]["handle"] = line[0]

    # add labels
    fig.text(0.5, 0.2, "Gradient Update Steps",
            ha="center", va="center", fontsize=16)
    fig.text(0.06, 0.58, f"{title} Mean Dice Overlap", ha="center",
            va="center", rotation="vertical", fontsize=16)
    handles = [
        LOSS_FUNTION_CONFIG[loss_function]["handle"] for loss_function in LOSS_FUNTION_ORDER
    ]
    labels = [
        LOSS_FUNTION_CONFIG[loss_function]["display_name"]
        for loss_function in LOSS_FUNTION_ORDER
    ]

    if mode == 'val':
        fig.legend(handles, labels, loc="lower center",
                ncol=len(handles), handlelength=1, columnspacing=1, fontsize=10.5)
    else:
        legend = fig.legend([], [], loc="lower center",
                ncol=len(handles), handlelength=1, columnspacing=1)
        legend.get_frame().set_edgecolor("white")

    # configure precision
    for ax in axs:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #ax.set_xticks([0, 10000, 20000, 30000])

    os.makedirs("./out/plots/pdf/", exist_ok=True)
    os.makedirs("./out/plots/png/", exist_ok=True)

    plt.savefig(f"./out/plots/pdf/convergence_{mode}.pdf", bbox_inches='tight')
    plt.savefig(f"./out/plots/png/convergence_{mode}.png", bbox_inches='tight')
