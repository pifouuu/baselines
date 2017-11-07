#!/bin/sh
#PBS -N test_job
#PBS -o test_job.out
#PBS -b test_job.err
#PBS -m abe
#PBS -M pierre.fournier@isir.upmc.fr
#PBS -l walltime=00:01:00
#PBS -l ncpus=1
/usr/bin/python3.5 /home/fournier/baselines/baselines/ddpg/main.py --nb-epsiodes 2 --env-wrapper-type standard