# Prepare KITTI 2012&2015 dataset

<!-- [DATASET] -->

```bibtex
@ARTICLE{Menze2018JPRS,
  author = {Moritz Menze and Christian Heipke and Andreas Geiger},
  title = {Object Scene Flow},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing (JPRS)},
  year = {2018}
}

@INPROCEEDINGS{Menze2015ISA,
  author = {Moritz Menze and Christian Heipke and Andreas Geiger},
  title = {Joint 3D Estimation of Vehicles and Scene Flow},
  booktitle = {ISPRS Workshop on Image Sequence Analysis (ISA)},
  year = {2015}
}
```

[comment]: <> (```text)

[comment]: <> ([comment]: <> &#40;kitti2015&#41;)

[comment]: <> ([comment]: <> &#40;|   |   ├── training&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   ├── disp_occ_0&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   |   ├── xxxxxx_xx.png&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   ├── image_3&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   |   ├── xxxxxx_xx.png&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   ├── image_2&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   |   ├── xxxxxx_xx.png&#41;)

[comment]: <> ([comment]: <> &#40;```&#41;)

[comment]: <> ([comment]: <> &#40;```text&#41;)

[comment]: <> ([comment]: <> &#40;kitti2012&#41;)

[comment]: <> ([comment]: <> &#40;|   |   ├── training&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   ├── colored_0&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   |   ├── xxxxxx_xx.png&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   ├── colored_1&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   |   ├── xxxxxx_xx.png&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   ├── disp_occ&#41;)

[comment]: <> ([comment]: <> &#40;|   |   |   |   ├── xxxxxx_xx.png&#41;)

[comment]: <> (```)

##Dataset structure

- please place the dataset as described in the `Datasets/Stereo/kitti2012_training_list.csv` and `Datasets/Stereo/kitti2015_training_list.csv` 
- Optionally you can write your own csv file and place the dataset as you like.

**Note** : the directory in the .csv file shoule be replaced.  


You can download datasets on this [webpage](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.