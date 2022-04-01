import matplotlib.pyplot as plt

# enable math typesetting in matplotlib
import matplotlib

#matplotlib.rc("text", usetex=True)
# matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

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
        "primary_color": "#3282bd",
        "marker": "s",
        "our_method": False},
    "ncc": {
        "display_name": "NCC",
        "primary_color": "#5598ca",
        "marker": "^",
        "our_method": False},
    "ncc2": {
        "display_name": "$NCC$",
        "display_name_bold": "$NCC$",
        "primary_color": "#5598ca",
        "marker": "^",
        "our_method": False
    },
    "ncc+supervised": {
        "display_name": "$NCC_{sup}$",
        "display_name_bold": "$NCC_{sup}$",
        "primary_color": "#78afd6",
        "marker": ">",
        "our_method": False
    },
    "ncc2+supervised": {
        "display_name": "$NCC_{sup}$",
        "display_name_bold": "$NCC_{sup}$",
        "primary_color": "#78afd6",
        "marker": ">",
        "our_method": False
    },
    "nmi": {
        "display_name": "$NMI$",
        "display_name_bold": "$NMI$",
        "primary_color": "#9ac5e3",
        "marker": "v",
        "our_method": False},
    "vgg": {
        "display_name": "$VGG$",
        "display_name_bold": "$VGG$",
        "primary_color": "#bddbef",
        "marker": "<",
        "our_method": False
    },
    "deepsim": {
        "display_name": "$DeepSim_{seg}$",
        "display_name_bold": "$\\bf{DeepSim}_{seg}$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae": {
        "display_name": "$DeepSim_{ae}$",
        "display_name_bold": "$\\bf{DeepSim}_{ae}$",
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
        "display_name": "$DeepSim_{seg}(EbT)$",
        "display_name_bold": "$DeeSim_{seg}(EbT)$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae-ebw": {
        "display_name": "$DeepSim_{ae}(EbT)$",
        "display_name_bold": "$DeeSim_{ae}EbT)$",
        "primary_color": plt.get_cmap("tab20c").colors[5],
        "marker": "o",
        "our_method": True
    },
    "syn": {
        "display_name": "$SyN$",
        "feature_extractor": "none",
        "primary_color": plt.get_cmap("tab20c").colors[18],
        "our_method": False,
    },
    "syn_ae": {
        "display_name": "$SyN + DeepSim_{ae}$",
        "display_name_bold": "$\\bf{SyN + Deepim}_{ae}$",
        "feature_extractor": "ae",
        "primary_color": plt.get_cmap("tab20c").colors[17],
        "our_method": True,
    },
    "syn_seg": {
        "display_name": "$SyN + DeepSim_{seg}$",
        "display_name_bold": "$\\bf{SyN + Deepim}_{seg}$",
        "feature_extractor": "seg",
        "primary_color": plt.get_cmap("tab20c").colors[16],
        "our_method": True,
    },
}
