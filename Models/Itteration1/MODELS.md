### Model weights from the first itteration of the project

Models were trained on 10080 training audios. 2580 bonafide ones and 7500 spoofed ones.

`best_audio_spoof_model_weighted` used a class weight of 3 on the bonafide audios.
`best_audio_spoof_model_unweighted` did not used any class weight

The architecture of the 2 models:

ResNet18 convolutions -> Linear(512, 128) -> ReLu -> Linear(128, 1) with a dropout probability of 0.5
