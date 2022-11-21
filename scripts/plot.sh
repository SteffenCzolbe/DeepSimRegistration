# voxelmorph
python3 -m src.plots.tune_voxelmorph
python3 -m src.plots.run_models --net voxelmorph
python3 -m src.plots.smoothness_vs_dice_score --net voxelmorph
python3 -m src.plots.convergence --mode train
python3 -m src.plots.convergence --mode val
python3 -m src.plots.statistical_significance_test
python3 -m src.plots.metrics_to_csv
python3 -m src.plots.test_score
python3 -m src.plots.test_score_per_class
python3 -m src.plots.img_sample
python3 -m src.plots.loss_sample
python3 -m src.plots.feature_maps
python3 -m src.plots.run_models_with_added_noise
python3 -m src.plots.plot_models_with_added_noise

# transmorph
python3 -m src.plots.tune_transmorph
python3 -m src.plots.run_models --net transmorph
python3 -m src.plots.smoothness_vs_dice_score --net transmorph

# ablation studies
python3 -m src.plots.tune_levels --deepsim ae
python3 -m src.plots.tune_levels --deepsim seg
python3 -m src.plots.tune_transfer_learning
python3 -m src.plots.tune_extract_before_warp

# # crop pdfs for paper
# pdfcrop ./out/plots/pdf/convergence_train.pdf ./out/plots/pdf/convergence_train.pdf
# pdfcrop ./out/plots/pdf/convergence_val.pdf ./out/plots/pdf/convergence_val.pdf
# pdfcrop ./out/plots/pdf/extract_before_warp.pdf ./out/plots/pdf/extract_before_warp.pdf
# pdfcrop ./out/plots/pdf/img_sample_detail_all.pdf ./out/plots/pdf/img_sample_detail_all.pdf 
# pdfcrop ./out/plots/pdf/img_sample.pdf ./out/plots/pdf/img_sample.pdf 
# pdfcrop ./out/plots/pdf/levels_ae.pdf ./out/plots/pdf/levels_seg.pdf
# pdfcrop ./out/plots/pdf/loss_sampleA.pdf ./out/plots/pdf/loss_sampleA.pdf
# pdfcrop ./out/plots/pdf/loss_sampleB.pdf ./out/plots/pdf/loss_sampleB.pdf 
# pdfcrop ./out/plots/pdf/smoothness_vs_dice_overlap_transmorph.pdf ./out/plots/pdf/smoothness_vs_dice_overlap_transmorph.pdf 
# pdfcrop ./out/plots/pdf/smoothness_vs_dice_overlap_voxelmorph.pdf ./out/plots/pdf/smoothness_vs_dice_overlap_voxelmorph.pdf 
# pdfcrop ./out/plots/pdf/test_score_per_class_brain-mri.pdf ./out/plots/pdf/test_score_per_class_brain-mri.pdf 
# pdfcrop ./out/plots/pdf/test_score.pdf ./out/plots/pdf/test_score.pdf 
# pdfcrop ./out/plots/pdf/transfer_learing.pdf ./out/plots/pdf/transfer_learing.pdf 
# pdfcrop ./out/plots/pdf/transmorph.pdf ./out/plots/pdf/transmorph.pdf 
# pdfcrop ./out/plots/pdf/voxelmorph.pdf ./out/plots/pdf/voxelmorph.pdf 
