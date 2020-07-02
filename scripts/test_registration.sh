python3 -m src.test_registration --weights ./weights/phc-u373/registration/ncc/weights.ckpt --out ./out/phc-u373/registration/ncc.tif
python3 -m src.test_registration --weights ./weights/phc-u373/registration/ncc+supervised/weights.ckpt --out ./out/phc-u373/registration/ncc+supervised.tif
python3 -m src.test_registration --weights ./weights/phc-u373/registration/deepsim/weights.ckpt --out ./out/phc-u373/registration/deepsim.tif

python3 -m src.test_registration --weights ./weights/platelet-em/registration/ncc/weights.ckpt --out ./out/platelet-em/registration/ncc.tif
python3 -m src.test_registration --weights ./weights/platelet-em/registration/ncc+supervised/weights.ckpt --out ./out/platelet-em/registration/ncc+supervised.tif
python3 -m src.test_registration --weights ./weights/platelet-em/registration/deepsim/weights.ckpt --out ./out/platelet-em/registration/deepsim.tif