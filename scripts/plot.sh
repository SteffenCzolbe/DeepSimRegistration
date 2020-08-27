python3 -m src.plots.convergence
python3 -m src.plots.run_models
python3 -m src.plots.statistical_significance_test
python3 -m src.plots.test_score
python3 -m src.plots.test_score_bar
python3 -m src.plots.test_score_per_class
python3 -m src.plots.brain_sample
python3 -m src.plots.img_sample

# crop pdfs
pdfcrop ./out/plots/convergence.pdf ./out/plots/convergence.pdf 
pdfcrop ./out/plots/test_score.pdf ./out/plots/test_score.pdf 
pdfcrop ./out/plots/test_score_bar.pdf ./out/plots/test_score_bar.pdf 
pdfcrop ./out/plots/test_score_per_class_brain-mri.pdf ./out/plots/test_score_per_class_brain-mri.pdf 
pdfcrop ./out/plots/brain_sample.pdf ./out/plots/brain_sample.pdf 
pdfcrop ./out/plots/img_sample.pdf ./out/plots/img_sample.pdf 