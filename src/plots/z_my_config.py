import matplotlib.pyplot as plt
import numpy as np
# enable math typesetting in matplotlib
import matplotlib

matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

orange_cmap = plt.get_cmap("Oranges")
orange_colors = orange_cmap(np.arange(0, orange_cmap.N))
red_cmap = plt.get_cmap("Reds")
red_colors = red_cmap(np.arange(0, red_cmap.N))

#DATASET_ORDER = ["brain-mri", "platelet-em", "phc-u373"]
DATASET_ORDER = ["platelet-em", "phc-u373"]

PLOT_CONFIG = {
    "phc-u373": {"display_name": "PhC-U373", "smoothing_factor": 0.98},
    "platelet-em": {"display_name": "Platelet-EM", "smoothing_factor": 0.997},
    #"brain-mri": {"display_name": "Brain-MRI", "smoothing_factor": 0.8},
}

LOSS_FUNTION_ORDER = [
    "mind",
    #"l2",
    #"ncc2",
    #"ncc2+supervised",
    #"nmi",
    #"vgg",
    #"deepsim-ae",
    #"deepsim",
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

EXTRACT_TRANSFER_LOSS_FUNTIONS = [
    "deepsim-ae",
    "deepsim-transfer-ae",
    "deepsim",
    "deepsim-transfer",
    "vgg",
]

EXTRACT_LEVEL_AE_LOSS_FUNTIONS = [
    "deepsim-ae_0",
    "deepsim-ae_1",
    "deepsim-ae_2",
    "deepsim-ae_01",
    "deepsim-ae_02",
    "deepsim-ae_12",
    "deepsim-ae",
]

EXTRACT_LEVEL_SEG_LOSS_FUNTIONS = [
    "deepsim_0",
    "deepsim_1",
    "deepsim_2",
    "deepsim_01",
    "deepsim_02",
    "deepsim_12",
    "deepsim",
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
        "display_name": r"$\mathrm{NCC}$",
        "display_name_bold": r"$\mathrm{NCC}$",
        "primary_color": "#5598ca",
        "marker": "^",
        "our_method": False
    },
    "ncc+supervised": {
        "display_name": r"$\mathrm{NCC}_{\mathrm{sup}}$",
        "display_name_bold": r"$\mathrm{NCC}_{\mathrm{sup}}$",
        "primary_color": "#78afd6",
        "marker": ">",
        "our_method": False
    },
    "ncc2+supervised": {
        "display_name": r"$\mathrm{NCC}_{\mathrm{sup}}$",
        "display_name_bold": r"$\mathrm{NCC}_{\mathrm{sup}}$",
        "primary_color": "#78afd6",
        "marker": ">",
        "our_method": False
    },
    "nmi": {
        "display_name": r"$\mathrm{NMI}$",
        "display_name_bold": r"$\mathrm{NMI}$",
        "primary_color": "#9ac5e3",
        "marker": "v",
        "our_method": False},
    "vgg": {
        "display_name": r"$\mathrm{VGG}$",
        "display_name_bold": r"$\mathrm{VGG}$",
        "primary_color": "#bddbef",
        "marker": "<",
        "our_method": False
    },
    "mind": {
        "display_name": r"$\mathrm{MIND}$",
        "display_name_bold": r"$\mathrm{MIND}$",
        "primary_color": "blue",
        "marker": "d",
        "our_method": False
    },
    "deepsim": {
        "display_name": r"$\mathrm{DeepSim}_{\mathrm{seg}}$",
        "display_name_bold": r"$\mathrm{DeepSim}_{\mathrm{seg}}$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae": {
        "display_name": r"$\mathrm{DeepSim}_{\mathrm{ae}}$",
        "display_name_bold": r"$\mathrm{DeepSim}_{\mathrm{ae}}$",
        "primary_color": plt.get_cmap("tab20c").colors[5],
        "marker": "o",
        "our_method": True
    },
    "deepsim-transfer": {
        "display_name": r"$\mathrm{DeepSim}_{\mathrm{seg}}\mathrm{-Transfer}$",
        "display_name_bold": r"$\mathrm{DeepSim}_{\mathrm{seg}}\mathrm{-Transfer}$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x",
        "our_method": True
    },
    "deepsim-transfer-ae": {
        "display_name": r"$\mathrm{DeepSim}_{\mathrm{ae}}\mathrm{-Transfer}$",
        "display_name_bold": r"$\mathrm{DeepSim}_{\mathrm{ae}}\mathrm{-Transfer}$",
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
        "display_name": r"$\mathrm{DeepSim}_{\mathrm{seg}}\mathrm{(EbT)}$",
        "display_name_bold": r"$\mathrm{DeepSim}_{\mathrm{seg}}\mathrm{(EbT)}$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae-ebw": {
        "display_name": r"$\mathrm{DeepSim}_{\mathrm{ae}}\mathrm{(EbT)}$",
        "display_name_bold": r"$\mathrm{DeepSim}_{\mathrm{ae}}\mathrm{(EbT)}$",
        "primary_color": plt.get_cmap("tab20c").colors[5],
        "marker": "o",
        "our_method": True
    },
    "syn": {
        "display_name": r"$\mathrm{SyN}$",
        "feature_extractor": "none",
        "primary_color": plt.get_cmap("tab20c").colors[18],
        "our_method": False,
    },
    "syn_ae": {
        "display_name": r"$\mathrm{SyN + }\mathrm{DeepSim}_{\mathrm{ae}}$",
        "display_name_bold": r"$\mathrm{SyN + }\mathrm{DeepSim}_{\mathrm{ae}}$",
        "feature_extractor": "ae",
        "primary_color": plt.get_cmap("tab20c").colors[17],
        "our_method": True,
    },
    "syn_seg": {
        "display_name": r"$\mathrm{SyN + }\mathrm{DeepSim}_{\mathrm{seg}}$",
        "display_name_bold": r"$\mathrm{SyN + }\mathrm{DeepSim}_{\mathrm{seg}}$",
        "feature_extractor": "seg",
        "primary_color": plt.get_cmap("tab20c").colors[16],
        "our_method": True,
    },

    "deepsim_0": {
        "display_name": r"$\mathrm{DeepSim}^{1}_{\mathrm{seg}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{1}_{\mathrm{seg}}$",
        "primary_color": 'royalblue', #tuple(red_colors[64][:-1]),
        "marker": "*",
        "our_method": True
    },
    "deepsim_1": {
        "display_name": r"$\mathrm{DeepSim}^{2}_{\mathrm{seg}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{2}_{\mathrm{seg}}$",
        "primary_color": 'mediumturquoise', #tuple(red_colors[96][:-1]),
        "marker": "*",
        "our_method": True
    },
    "deepsim_2": {
        "display_name": r"$\mathrm{DeepSim}^{3}_{\mathrm{seg}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{3}_{\mathrm{seg}}$",
        "primary_color": 'limegreen', #tuple(red_colors[128][:-1]),
        "marker": "*",
        "our_method": True
    },
    "deepsim_01": {
        "display_name": r"$\mathrm{DeepSim}^{12}_{\mathrm{seg}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{12}_{\mathrm{seg}}$",
        "primary_color": 'greenyellow', #tuple(orange_colors[64][:-1]),
        "marker": "x",
        "our_method": True
    },
    "deepsim_02": {
        "display_name": r"$\mathrm{DeepSim}^{13}_{\mathrm{seg}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{13}_{\mathrm{seg}}$",
        "primary_color": 'yellowgreen', #tuple(orange_colors[96][:-1]),
        "marker": "x",
        "our_method": True
    },
    "deepsim_12": {
        "display_name": r"$\mathrm{DeepSim}^{23}_{\mathrm{seg}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{23}_{\mathrm{seg}}$",
        "primary_color": 'gold', #tuple(orange_colors[128][:-1]),
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae_0": {
        "display_name": r"$\mathrm{DeepSim}^{1}_{\mathrm{ae}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{1}_{\mathrm{ae}}$",
        #"primary_color": tuple(red_colors[64][:-1]),
        "primary_color": 'royalblue',
        "marker": "*",
        "our_method": True
    },
    "deepsim-ae_1": {
        "display_name": r"$\mathrm{DeepSim}^{2}_{\mathrm{ae}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{2}_{\mathrm{ae}}$",
        #"primary_color": tuple(red_colors[96][:-1]),
        "primary_color": 'mediumturquoise',
        "marker": "*",
        "our_method": True
    },
    "deepsim-ae_2": {
        "display_name": r"$\mathrm{DeepSim}^{3}_{\mathrm{ae}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{3}_{\mathrm{ae}}$",
        #"primary_color": tuple(red_colors[128][:-1]),
        "primary_color": 'limegreen',
        "marker": "*",
        "our_method": True
    },
    "deepsim-ae_01": {
        "display_name": r"$\mathrm{DeepSim}^{12}_{\mathrm{ae}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{12}_{\mathrm{ae}}$",
        #"primary_color": tuple(orange_colors[64][:-1]),
        "primary_color": 'greenyellow',
        "marker": "o",
        "our_method": True
    },
    "deepsim-ae_02": {
        "display_name": r"$\mathrm{DeepSim}^{13}_{\mathrm{ae}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{13}_{\mathrm{ae}}$",
        #"primary_color": tuple(orange_colors[96][:-1]),
        "primary_color":  'yellowgreen',
        "marker": "o",
        "our_method": True
    },
    "deepsim-ae_12": {
        "display_name": r"$\mathrm{DeepSim}^{23}_{\mathrm{ae}}$",
        "display_name_bold": r"$\mathrm{DeepSim}^{23}_{\mathrm{ae}}$",
        #"primary_color": tuple(orange_colors[128][:-1]),
        "primary_color": 'gold',
        "marker": "o",
        "our_method": True
    },
   
}