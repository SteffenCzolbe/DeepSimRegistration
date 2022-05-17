import matplotlib.pyplot as plt
import numpy as np
import matplotlib

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

orange_cmap = plt.get_cmap("Oranges")
orange_colors = orange_cmap(np.arange(0, orange_cmap.N))
red_cmap = plt.get_cmap("Reds")
red_colors = red_cmap(np.arange(0, red_cmap.N))

DATASET_ORDER = ["platelet-em", "phc-u373"]

PLOT_CONFIG = {
    "phc-u373": {"display_name": "PhC-U373", "smoothing_factor": 0.98},
    "platelet-em": {"display_name": "Platelet-EM", "smoothing_factor": 0.997},
}

LOSS_FUNTION_ORDER = [
    "l2",
    "ncc2",
    "ncc2+supervised",
    "nmi",
    "mind",
    "deepsim-ae",
    "deepsim",
]

MIND_AND_OTHER_LOSS_FUNTION = [
    "l2",
    "ncc2",
    "ncc2+supervised",
    "nmi",
    "mind",
    "deepsim-ae",
    "deepsim",
]

EXTRACT_ZERO_MEAN_LOSS_FUNCTIONS = [
    "deepsim-ae-zero",
    "deepsim-zero",
]


EXTRACT_TRANSMORPH_LOSS_FUNCTIONS = [
    "l2",
    "ncc2",
    "ncc2+supervised",
    "nmi",
    "mind",
    #"vgg",
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
        "display_name": "$MSE$",
        "primary_color": "#3282bd",
        "marker": "s",
        "our_method": False},
    "ncc": {
        "display_name": "$NCC$",
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
        "display_name": "$DeepSim_{VGG}$",
        "display_name_bold": "$DeepSim{VGG}$",
        "primary_color": plt.get_cmap("tab20b").colors[-3],
        "marker": "d",
        "our_method": False
    },
    "mind": {
        "display_name": "$MIND$",
        "display_name_bold": "$MIND$",
        "primary_color": "#bddbef",
        "marker": "<",
        "our_method": False
    },
    "deepsim": {
        "display_name": "$DeepSim_{seg}$",
        "display_name_bold": "$DeepSim_{seg}$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae": {
        "display_name": "$DeepSim_{ae}$",
        "display_name_bold": "$DeepSim_{ae}$",
        "primary_color": plt.get_cmap("tab20c").colors[5],
        "marker": "o",
        "our_method": True
    },
    "deepsim-transfer": {
        "display_name": "$DeepSim_{seg}-TL$",
        "display_name_bold": "$DeepSim_{seg}-TL$",
        "primary_color": plt.get_cmap("tab20c").colors[12],
        "marker": "*",
        "our_method": True
    },
    "deepsim-transfer-ae": {
        "display_name": "$DeepSim_{ae}-TL$",
        "display_name_bold": "$DeepSim_{ae}-TL$",
        "primary_color": plt.get_cmap("tab20c").colors[13],
        "marker": "p",
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
        "display_name_bold": "$DeepSim_{seg}(EbT)$",
        "primary_color": plt.get_cmap("tab20c").colors[4],
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae-ebw": {
        "display_name": "$DeepSim_{ae}(EbT)$",
        "display_name_bold": "$DeepSim_{ae}(EbT)$",
        "primary_color": plt.get_cmap("tab20c").colors[5],
        "marker": "o",
        "our_method": True
    },
    "syn": {
        "display_name": "$SyN$",
        "feature_extracto": "none",
        "primary_color": plt.get_cmap("tab20c").colors[18],
        "our_method": False,
    },
    "syn_ae": {
        "display_name": "$SyN + DeepSim_{ae}$",
        "display_name_bold": "$SyN + DeepSim_{ae}$",
        "feature_extracto": "ae",
        "primary_color": plt.get_cmap("tab20c").colors[17],
        "our_method": True,
    },
    "syn_seg": {
        "display_name": "$SyN + DeepSim_{seg}$",
        "display_name_bold": "$SyN + DeepSim_{seg}$",
        "feature_extracto": "seg",
        "primary_color": plt.get_cmap("tab20c").colors[16],
        "our_method": True,
    },

    "deepsim_0": {
        "display_name": "$DeepSim^{1}_{seg}$",
        "display_name_bold": "$DeepSim^{1}_{seg}$",
        "primary_color": plt.get_cmap("tab10").colors[-1],
        "marker": "|",
        "our_method": True
    },
    "deepsim_1": {
        "display_name": "$DeepSim^{2}_{seg}$",
        "display_name_bold": "$DeepSim^{2}_{seg}$",
        "primary_color": plt.get_cmap("tab10").colors[0],
        "marker": "|",
        "our_method": True
    },
    "deepsim_2": {
        "display_name": "$DeepSim^{3}_{seg}$",
        "display_name_bold": "$DeepSim^{3}_{seg}$",
        "primary_color": plt.get_cmap("tab10").colors[6],
        "marker": "|",
        "our_method": True
    },
    "deepsim_01": {
        "display_name": "$DeepSim^{12}_{seg}$",
        "display_name_bold": "$DeepSim^{12}_{seg}$",
        "primary_color": plt.get_cmap("tab10").colors[2],
        "marker": "x",
        "our_method": True
    },
    "deepsim_02": {
        "display_name": "$DeepSim^{13}_{seg}$",
        "display_name_bold": "$DeepSim^{13}_{seg}$",
        "primary_color": plt.get_cmap("tab10").colors[5],
        "marker": "x",
        "our_method": True
    },
    "deepsim_12": {
        "display_name": "$DeepSim^{23}_{seg}$",
        "display_name_bold": "$DeepSim^{23}_{seg}$",
        "primary_color": 'gold',
        "marker": "x",
        "our_method": True
    },
    "deepsim-ae_0": {
        "display_name": "$DeepSim^{1}_{ae}$",
        "display_name_bold": "$DeepSim^{1}_{ae}$",
        "primary_color": plt.get_cmap("tab10").colors[-1],
        "marker": ".",
        "our_method": True
    },
    "deepsim-ae_1": {
        "display_name": "$DeepSim^{2}_{ae}$",
        "display_name_bold": "$DeepSim^{2}_{ae}$",
        "primary_color": plt.get_cmap("tab10").colors[0],
        "marker": ".",
        "our_method": True
    },
    "deepsim-ae_2": {
        "display_name": "$DeepSim^{3}_{ae}$",
        "display_name_bold": "$DeepSim^{3}_{ae}$",
        "primary_color": plt.get_cmap("tab10").colors[6],
        "marker": ".",
        "our_method": True
    },
    "deepsim-ae_01": {
        "display_name": "$DeepSim^{12}_{ae}$",
        "display_name_bold": "$DeepSim^{12}_{ae}$",
        "primary_color": plt.get_cmap("tab10").colors[2],
        "marker": "o",
        "our_method": True
    },
    "deepsim-ae_02": {
        "display_name": "$DeepSim^{13}_{ae}$",
        "display_name_bold": "$DeepSim^{13}_{ae}$",
        "primary_color": plt.get_cmap("tab10").colors[5],
        "marker": "o",
        "our_method": True
    },
    "deepsim-ae_12": {
        "display_name": "$DeepSim^{23}_{ae}$",
        "display_name_bold": "$DeepSim^{23}_{ae}$",
        #"primary_color": tuple(orange_colors[128][:-1]),
        "primary_color": 'gold',
        "marker": "o",
        "our_method": True
    },
   
}