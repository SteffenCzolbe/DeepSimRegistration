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

SYN_LOSS_FUNTIONS = [
    "syn",
    "syn_ae",
    "syn_seg",
]

ALL_METHODS = LOSS_FUNTION_ORDER + [None] + SYN_LOSS_FUNTIONS

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
        "marker": "s",
        "our_method": False},
    "ncc": {
        "display_name": "NCC",
        "primary_color": plt.get_cmap("tab20c").colors[1],
        "marker": "^",
        "our_method": False},
    "ncc2": {
        "display_name": r"$\text{NCC}$",
        "display_name_bold": r"$\textbf{NCC}$",
        "primary_color": plt.get_cmap("tab20c").colors[1],
        "marker": "^",
        "our_method": False
    },
    "ncc+supervised": {
        "display_name": r"$\text{NCC}_{\text{sup}}$",
        "display_name_bold": r"$\textbf{NCC}_{\textbf{sup}}$",
        "primary_color": plt.get_cmap("tab20c").colors[2],
        "marker": ">",
        "our_method": False
    },
    "ncc2+supervised": {
        "display_name": r"$\text{NCC}_{\text{sup}}$",
        "display_name_bold": r"$\textbf{NCC}_{\textbf{sup}}$",
        "primary_color": plt.get_cmap("tab20c").colors[2],
        "marker": ">",
        "our_method": False
    },
    "nmi": {
        "display_name": r"$\text{NMI}$",
        "display_name_bold": r"$\textbf{NMI}$",
        "primary_color": plt.get_cmap("tab20b").colors[2],
        "marker": "v",
        "our_method": False},
    "vgg": {
        "display_name": r"$\text{VGG}$",
        "display_name_bold": r"$\textbf{VGG}$",
        "primary_color": plt.get_cmap("tab20c").colors[3],
        "marker": "<",
        "our_method": False
    },
    "deepsim": {
        "display_name": r"$\text{DeepSim}_{\text{seg}}$",
        "display_name_bold": r"$\textbf{DeepSim}_{\textbf{seg}}$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae": {
        "display_name": r"$\text{DeepSim}_{\text{ae}}$",
        "display_name_bold": r"$\textbf{DeepSim}_{\textbf{ae}}$",
        "primary_color": plt.get_cmap("tab20c").colors[5],
        "marker": "o",
        "our_method": True
    },
    "transfer": {
        "display_name": "Transfer",
        "primary_color": plt.get_cmap("tab20c").colors[16],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ebw": {
        "display_name": r"$\text{DeepSim}_{\text{seg}}\text{(EbT)}$",
        "display_name_bold": r"$\textbf{DeepSim}_{\textbf{seg}}\text{(EbT)}$",
        "primary_color": plt.get_cmap("tab20c").colors[6],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae-ebw": {
        "display_name": r"$\text{DeepSim}_{\text{ae}}\text{(EbT)}$",
        "display_name_bold": r"$\textbf{DeepSim}_{\textbf{ae}}\text{(EbT)}$",
        "primary_color": plt.get_cmap("tab20c").colors[7],
        "marker": "o",
        "our_method": True
    },
    "syn": {
        "display_name": r"$\text{SyN}$",
        "feature_extractor": "none",
        "primary_color": plt.get_cmap("tab20c").colors[18],
        "our_method": False,
    },
    "syn_ae": {
        "display_name": r"$\text{SyN + }\text{DeepSim}_{\text{ae}}$",
        "display_name_bold": r"$\textbf{SyN + }\textbf{DeepSim}_{\text{ae}}$",
        "feature_extractor": "ae",
        "primary_color": plt.get_cmap("tab20c").colors[17],
        "our_method": True,
    },
    "syn_seg": {
        "display_name": r"$\text{SyN + }\text{DeepSim}_{\text{seg}}$",
        "display_name_bold": r"$\textbf{SyN + }\textbf{DeepSim}_{\text{seg}}$",
        "feature_extractor": "seg",
        "primary_color": plt.get_cmap("tab20c").colors[16],
        "our_method": True,
    },
}
