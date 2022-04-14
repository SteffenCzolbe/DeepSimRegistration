#!/bin/sh

echo "Downloading and preprocessing PhC-U373..."
PSHOME=$(pwd)
cd data/PhC-U373/
python3 PhC-U373_download_and_preprocess.py 
cd $PSHOME