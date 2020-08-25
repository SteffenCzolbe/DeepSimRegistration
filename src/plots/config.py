import matplotlib.pyplot as plt

DATASET_ORDER = ['brain-mri', 'platelet-em', 'phc-u373']
PLOT_CONFIG = {'phc-u373':{'display_name':'PhC-U373',
                           'smoothing_factor': 0.97},
               'platelet-em':{'display_name':'Platelet-EM',
                           'smoothing_factor': 0.995},
               'brain-mri':{'display_name':'Brain-MRI',
                           'smoothing_factor': 0.0},
                           }
LOSS_FUNTION_ORDER = ['l2', 'ncc', 'ncc+supervised', 'deepsim', 'transfer', 'vgg']
LOSS_FUNTION_CONFIG = {'l2':{'display_name': 'MSE',
                             'primary_color': plt.get_cmap('tab20c').colors[0]},
                       'ncc':{'display_name': 'NCC',
                             'primary_color': plt.get_cmap('tab20c').colors[1]},
                       'ncc+supervised':{'display_name': '$\mathregular{NCC_{sup}}$',
                             'primary_color': plt.get_cmap('tab20c').colors[2]},
                       'deepsim':{'display_name': 'DeepSim',
                             'primary_color': plt.get_cmap('tab20c').colors[4]},
                       'vgg':{'display_name': 'VGG',
                             'primary_color': plt.get_cmap('tab20c').colors[16]},
                       'transfer':{'display_name': 'Transfer',
                             'primary_color': plt.get_cmap('tab20c').colors[5]},
}