# Prepare SceneFlow dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{mayer2016large,
  title={A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation},
  author={Mayer, Nikolaus and Ilg, Eddy and Hausser, Philip and Fischer, Philipp and Cremers, Daniel and Dosovitskiy, Alexey and Brox, Thomas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4040--4048},
  year={2016}
}
```

## Dataset structure

- please place the dataset as described in the `Datasets/Stereo/scene_flow_training_list.csv` and `Datasets/Stereo/scene_flow_testing_list.csv` 
- Optionally you can write your own csv file and place the dataset as you like. 

**Note** : the directory in the .csv file shoule be replaced.  

You can download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) from this website. Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.
