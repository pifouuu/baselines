#!/bin/sh
#PBS -N ddpg_mountain_car
#PBS -o ddpg_mountain_car.out
#PBS -b ddpg_mountain_car.err
#PBS -m abe
#PBS -M pierre.fournier@isir.upmc.fr
#PBS -l walltime=12:00:00
#PBS -l ncpus=8
/usr/bin/python3.4 /home/fournier/baselines/baselines/ddpg/main.py --log-dir /home/fournier/baselines/baselines/ddpg/log/ --nb-epochs 200