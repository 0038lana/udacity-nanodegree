# Image Classifier

Classifying types of flowers using Convolutional Neural Networks.

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, first code for an image classifier built with PyTorch was developed, then it was converted into a command line application.

CLI usage:
Train a new network on a data set with train.py:
 * Basic usage: `python train.py data_directory` where data_directory is the directory where data is stored, prints out training loss, validation loss, and validation accuracy as the network trains
 * Options:
  * Set directory to save checkpoints: `python train.py data_directory --save_dir save_directory`save_directory is the directory where trained model should be saved
  * Set hyperparameters: `python train.py data_directory --learning_rate 0.01 --hidden_units 512 --epochs 20`
  * Use GPU for training: `python train.py data_directory --gpu` otherwise it uses CPU
