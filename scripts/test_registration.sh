python3 -m src.test_2d_registration --weights ./weights/phc-u373/registration/l2/weights.ckpt --out ./out/phc-u373/registration/l2.tif
python3 -m src.test_2d_registration --weights ./weights/phc-u373/registration/ncc/weights.ckpt --out ./out/phc-u373/registration/ncc.tif
python3 -m src.test_2d_registration --weights ./weights/phc-u373/registration/ncc+supervised/weights.ckpt --out ./out/phc-u373/registration/ncc+supervised.tif
python3 -m src.test_2d_registration --weights ./weights/phc-u373/registration/deepsim/weights.ckpt --out ./out/phc-u373/registration/deepsim.tif
python3 -m src.test_2d_registration --weights ./weights/phc-u373/registration/deepsim_transfer/weights.ckpt --out ./out/phc-u373/registration/deepsim_transfer.tif
python3 -m src.test_2d_registration --weights ./weights/phc-u373/registration/vgg/weights.ckpt --out ./out/phc-u373/registration/vgg.tif

python3 -m src.test_2d_registration --weights ./weights/platelet-em/registration/l2/weights.ckpt --out ./out/platelet-em/registration/l2.tif
python3 -m src.test_2d_registration --weights ./weights/platelet-em/registration/ncc/weights.ckpt --out ./out/platelet-em/registration/ncc.tif
python3 -m src.test_2d_registration --weights ./weights/platelet-em/registration/ncc+supervised/weights.ckpt --out ./out/platelet-em/registration/ncc+supervised.tif
python3 -m src.test_2d_registration --weights ./weights/platelet-em/registration/deepsim/weights.ckpt --out ./out/platelet-em/registration/deepsim.tif
python3 -m src.test_2d_registration --weights ./weights/platelet-em/registration/deepsim_transfer/weights.ckpt --out ./out/platelet-em/registration/deepsim_transfer.tif
python3 -m src.test_2d_registration --weights ./weights/platelet-em/registration/vgg/weights.ckpt --out ./out/platelet-em/registration/vgg.tif

python3 -m src.test_3d_registration --weights ./weights/brain-mri/registration/l2/weights.ckpt --out ./out/brain-mri/registration/l2/
python3 -m src.test_3d_registration --weights ./weights/brain-mri/registration/ncc/weights.ckpt --out ./out/brain-mri/registration/ncc/
python3 -m src.test_3d_registration --weights ./weights/brain-mri/registration/ncc+supervised/weights.ckpt --out ./out/brain-mri/registration/ncc+supervised/
python3 -m src.test_3d_registration --weights ./weights/brain-mri/registration/deepsim/weights.ckpt --out ./out/brain-mri/registration/deepsim/
