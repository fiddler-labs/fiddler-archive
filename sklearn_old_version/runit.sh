#!/bin/bash


echo 'Running training code'

dir=$(cd $(dirname "$0"); pwd)
cd $dir

python train.py
