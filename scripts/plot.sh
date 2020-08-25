python3 -m src.plots.convergence
python3 -m src.plots.run_models
python3 -m src.plots.statistical_significance_test
python3 -m src.plots.test_score
python3 -m src.plots.test_score_bar

# crop pdfs

pdfcrop ./out/plots/convergence.pdf ./out/plots/convergence.pdf 
pdfcrop ./out/plots/test_score.pdf ./out/plots/test_score.pdf 
pdfcrop ./out/plots/test_score_bar.pdf ./out/plots/test_score_bar.pdf 
pdfcrop ./out/plots/test_score_per_class_brain-mri.pdf ./out/plots/test_score_per_class_brain-mri.pdf 