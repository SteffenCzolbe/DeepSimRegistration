import matplotlib.pyplot as plt

# enable math typesetting in matplotlib
import matplotlib

matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

DATASET_ORDER = ["brain-mri", "platelet-em", "phc-u373"]
PLOT_CONFIG = {
    "phc-u373": {"display_name": "PhC-U373", "smoothing_factor": 0.98},
    "platelet-em": {"display_name": "Platelet-EM", "smoothing_factor": 0.997},
    "brain-mri": {"display_name": "Brain-MRI", "smoothing_factor": 0.8},
}

LOSS_FUNTION_ORDER = [
    "l2",
    "ncc2",
    "ncc2+supervised",
    "nmi",
    "vgg",
    "deepsim-ae",
    "deepsim",
]

EXTRACT_BEFORE_WARP_LOSS_FUNTIONS = [
    "deepsim-ae",
    "deepsim-ae-ebw",
    "deepsim",
    "deepsim-ebw",
]

LOSS_FUNTION_CONFIG = {
    "l2": {
        "display_name": "MSE",
        "primary_color": plt.get_cmap("tab20c").colors[0],
        "marker": "s"},
    "ncc": {
        "display_name": "NCC",
        "primary_color": plt.get_cmap("tab20c").colors[1],
        "marker": "^"},
    "ncc2": {
        "display_name": r"$\text{NCC}$",
        "display_name_bold": r"$\textbf{NCC}$",
        "primary_color": plt.get_cmap("tab20c").colors[1],
        "marker": "^"
    },
    "ncc+supervised": {
        "display_name": r"$\text{NCC}_{\text{sup}}$",
        "display_name_bold": r"$\textbf{NCC}_{\textbf{sup}}$",
        "primary_color": plt.get_cmap("tab20c").colors[2],
        "marker": ">"
    },
    "ncc2+supervised": {
        "display_name": r"$\text{NCC}_{\text{sup}}$",
        "display_name_bold": r"$\textbf{NCC}_{\textbf{sup}}$",
        "primary_color": plt.get_cmap("tab20c").colors[2],
        "marker": ">"
    },
    "nmi": {
        "display_name": r"$\text{NMI}$",
        "display_name_bold": r"$\textbf{NMI}$",
        "primary_color": plt.get_cmap("tab20b").colors[2],
        "marker": "v"},
    "vgg": {
        "display_name": r"$\text{VGG}$",
        "display_name_bold": r"$\textbf{VGG}$",
        "primary_color": plt.get_cmap("tab20c").colors[3],
        "marker": "<"
    },
    "deepsim": {
        "display_name": r"$\text{DeepSim}_{\text{seg}}$",
        "display_name_bold": r"$\textbf{DeepSim}_{\textbf{seg}}$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x"
    },
    "deepsim-ae": {
        "display_name": r"$\text{DeepSim}_{\text{ae}}$",
        "display_name_bold": r"$\textbf{DeepSim}_{\textbf{ae}}$",
        "primary_color": plt.get_cmap("tab20c").colors[5],
        "marker": "o"
    },
    "transfer": {
        "display_name": "Transfer",
        "primary_color": plt.get_cmap("tab20c").colors[16],
        "marker": "x"
    },
    "deepsim-ebw": {
        "display_name": r"$\text{DeepSim}_{\text{seg}}\text{(EbT)}$",
        "display_name_bold": r"$\textbf{DeepSim}_{\textbf{seg}}\text{(EbT)}$",
        "primary_color": plt.get_cmap("tab20c").colors[6],
        "marker": "P"
    },
    "deepsim-ae-ebw": {
        "display_name": r"$\text{DeepSim}_{\text{ae}}\text{(EbT)}$",
        "display_name_bold": r"$\textbf{DeepSim}_{\textbf{ae}}\text{(EbT)}$",
        "primary_color": plt.get_cmap("tab20c").colors[7],
        "marker": "D"
    },
}
