# Neuronal-Network-TPMI-classifier-5G
This script allows to chose a TPMI index in order to minimize the time of computing in a PUSCH simulation.
The folder model contains a trained model and user has only to predict the value.
Neuronal Network has been trained with a 372000 samples of a CDL channel using one layer and two display port and 10 PRB .
The Neuronal Network is a custom keras model using a different layers. Also for trainings it has been used a custom keras data generator to allows to reduce the uses of memory.

To predict the TPMI:
input:
channel wich is a 120x8 matrix complex
noise estimation
pathloss

output:
TPMI.

