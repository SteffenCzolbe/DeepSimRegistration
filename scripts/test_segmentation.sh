python3 -m src.test_2d_segmentation --weights ./weights/platelet-em/segmentation/weights.ckpt --out ./out/platelet-em/segmentation/segmentation.tif
python3 -m src.test_2d_segmentation --weights ./weights/phc-u373/segmentation/weights.ckpt --out ./out/phc-u373/segmentation/segmentation.tif
python3 -m src.test_3d_segmentation --weights ./weights/brain-mri/segmentation/weights.ckpt --out ./out/brain-mri/segmentation/
python3 -m src.test_3d_segmentation --weights ./weights/hippocampusmr/segmentation/weights.ckpt --out ./out/hippocampusmr/segmentation/