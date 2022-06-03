#  Code for "How Well Do Sparse ImageNet Models Transfer?"

This code allows replicating the transfer learning experiments, for both linear and full finetuning, of the CVPR 2022 paper **How Well Do Sparse ImageNet Models Transfer?**. The arXiv version of our paper can be found [here](https://arxiv.org/pdf/2111.13445.pdf "here").


Part of our implementation is based on the open-source code accompanying the NeurIPS 2021 [paper](https://openreview.net/forum?id=T3_AJr9-R5g "paper") "AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks", which can be found [here](http://github.com/IST-DASLab/ACDC "here"). The dataset loading functionalities were adapted from the code for the NeurIPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/file/24357dd085d2c4b1a88a7e0692e60294-Paper.pdf "paper") "Do Adversarially Robust ImageNet Models Transfer Better?" (Salman, Ilyas et al), available [here](https://github.com/microsoft/robust-models-transfer "here").

## Datasets
For our main transfer learning experiments, we use 12 benchmark datasets. Please see our paper for the complete list of these datasets, along with the corresponding citations. The instructions for downloading the relevant datasets can be found [here](https://github.com/microsoft/robust-models-transfer "here").


## ImageNet Checkpoints
Some of the sparse ImageNet checkpoints we have tested are publicly available ([AC/DC](https://github.com/IST-DASLab/ACDC "AC/DC"), [RigL](https://github.com/google-research/rigl "RigL"), [STR](https://github.com/RAIVNLab/STR "STR")). Others we have trained ourselves; for example GMP and STR without label smoothing (following the official implementation). Finally, the [WoodFisher](https://github.com/IST-DASLab/WoodFisher "WoodFisher") and [LTH-T](https://github.com/VITA-Group/CV_LTH_Pre-training "LTH-T") checkpoints were provided by the authors, upon request.

For MobileNetV1 models we additionally used M-FAC checkpoints, which are also publicly available [here](https://github.com/IST-DASLab/M-FAC "here"). 

All checkpoints, whether created by us or by others, except for LTH-T (which are not public), have been converted to a common format for importing convenience. The converted checkpoints can be found at the following links: 
* [ResNet50](https://seafile.ist.ac.at/d/91de9650cfb54b77b7d1/)
* [MobileNetV1](https://seafile.ist.ac.at/d/48652b95fdd04bcd8bb6/)

## Transfer Learning Results

### Raw results
The raw numerical results presented in the paper, for both linear and full finetuning, using ResNet50 and MobileNetV1 architectures, can be found in the `raw_results` folder.

### Downstream Checkpoints
All checkpoints can be found [here](https://seafile.ist.ac.at/d/08fcf3f2835d4d7787ae/).

We make available one checkpoint for each transfer method/pruning method/sparsity/dataset.

The **Resnet50 downstream checkpoints** are available at the following links:
* [full transfer](https://seafile.ist.ac.at/d/c1ca5c38daff43a996ec/)
* [linear transfer](https://seafile.ist.ac.at/d/750fc67c50384262ae01/)

The **MobileNetV1 downstream checkpoints** are available at the following links:
* [full transfer](https://seafile.ist.ac.at/d/dffcfffb43f740a281d2/)
* [linear transfer](https://seafile.ist.ac.at/d/e691c5b6ef6a405fb356/).

## To reproduce our results:
We recommend access to at least one GPU for each experiment (as the batch sizes are small, we were able to train even the larger networks on just one GPU). We recommend Pytorch 1.8, and, if possible, Weights & Biases (Wandb). If using Wandb  is not possible, code should be run with the `--use_wandb` flag **disabled**.
For convenience, we provide shell scripts to execute the most common experiments. Below are example commands to run resnet50 experiments. (the first argument is the GPU number(s), the second is the names of the datasets, and the third is the paths to these datasets on disk):

**Important** Please keep in mind that for _full finetuning_, our experiments on the Aircraft and Cars datasets use a higher learning rate than for all others.
For this reason, we provide full finetuning scripts, with two different learning rates. Likewise, there are some special settings for training RigL models.  First, the original RigL models were trained with images resized
using _bicubic_ interpolation (default in TensorFlow), while all other upstream checkpoints were trained using images resized with bilinear interpolation. In addition, the layer names are different between RigL and all other models, and so the configuration file that specifies which layers to load is also different. While these modifications  apply to both full and linear finetuning, they  must be specified only in the full finetuning runs; in the linear finetuning runs it is handled automatically for any upstream checkpoint with "rigl" in the name. Sample scripts that handle this correctly are provided for the full finetuning case.

    ./run_dataset_generalize_full_training_lr01.sh 1 cars /path/to/cars/data

    ./run_dataset_generalize_full_training_lr001.sh 1 cifar10 /path/to/cifar10/data

    ./run_dataset_generalize_full_training_rigl_lr001.sh 1 cifar10 /path/to/cifar10/data
    
    ./run_dataset_generalize_preextracted.sh 1 cars /path/to/cars/data

To  run linear finetuning experiments using the [DeepSparse](https://github.com/neuralmagic/deepsparse "DeepSparse") inference engine:

    ./run_dataset_generalize_linear_finetuning_deepsparse.sh DSET /PATH/TO/DSET
    
    ./run_dataset_generalize_linear_finetuning_deepsparse_rigl.sh DSET /PATH/TO/DSET


## Structure of the repository

* `main.py` is the entry point to run full finetuning. You will need to provide data and config paths and specify dataset and architecture names.
* `finetune_pre_extracted_features.py` is the entry point to run linear finetuning using SGD.
* `configs/` contains yaml config files we use for specifying transfer configs as well as training schedules and checkpoint configs. We would draw your attention especially to the checkpoint configs: as we preferred to use checkpoints shared in the original works, different checkpoints have slightly different flavours of the network architecture as well as the encoding of the checkpoint itself. While these variations do not impact the quality of the models, they must be specified correctly during checkpoint loading, and the configuration file enables this. To run the code, the configuration file must be updated with the correct checkpoint paths.
* `models/` directory contains currently available models. To add a new model, have it be loaded by `get_model` function in `models/__init__.py`.
* `policies/` contains `trainer`-type policies that train an entire model or submodules, and a `Manager` class which executes these policies as specified in a given config.
* `utils/` contains utilities for loading datasets and models, masking layers for pruning, and performing helper computations.

## References
* Alexandra Peste, Eugenia Iofinova, Adrian Vladu, and Dan Alistarh. *AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks*. Conference on Neural Information Processing Systems (NeurIPS), 2021 [link to paper](https://proceedings.neurips.cc/paper/2021/file/48000647b315f6f00f913caa757a70b3-Paper.pdf "link to paper")
* Elias Frantar, Eldar Kurtic, and Dan Alistarh. *M-FAC: Efficient matrix-free approximations of second-order information*. Conference on Neural Information Processing Systems (NeurIPS), 2021 [link to paper](https://proceedings.neurips.cc/paper/2021/file/7cfd5df443b4eb0d69886a583b33de4c-Paper.pdf "link to paper")
* Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Michael Carbin, and Zhangyang Wang. *The
lottery tickets hypothesis for supervised and self-supervised pre-training in computer vision models*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021 [link to paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_The_Lottery_Tickets_Hypothesis_for_Supervised_and_Self-Supervised_Pre-Training_in_CVPR_2021_paper.pdf "link to paper")
* Sidak Pal Singh and Dan Alistarh. *WoodFisher: Efficient second-order approximation for neural network compression*. Conference on Neural Information Processing Systems (NeurIPS), 2020 [link to paper](https://proceedings.neurips.cc/paper/2020/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf "link to paper")
* Hadi Salman, Andrew Ilyas, Logan Engstrom, Ashish Kapoor, and Aleksander Madry. *Do adversarially robust ImageNet models transfer better?* Conference on Neural Information Processing Systems (NeurIPS), 2020 [link to paper](https://proceedings.neurips.cc/paper/2020/file/24357dd085d2c4b1a88a7e0692e60294-Paper.pdf)
* Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, and Erich Elsen. *Rigging the lottery: Making all tickets winners*. In International Conference on Machine Learning (ICML), 2020 [link to paper](http://proceedings.mlr.press/v119/evci20a/evci20a.pdf "link to paper")
* Aditya Kusupati, Vivek Ramanujan, Raghav Somani, Mitchell Wortsman, Prateek Jain, Sham Kakade, and Ali Farhadi. *Soft threshold weight reparameterization for learnable sparsity*. In International Conference on Machine Learning (ICML), 2020 [link to paper](http://proceedings.mlr.press/v119/kusupati20a/kusupati20a.pdf "link to paper")
* Michael Zhu and Suyog Gupta. *To prune, or not to prune: exploring the efficacy of pruning for model compression*. arXiv
preprint arXiv:1710.01878, 2017. [link to paper](https://arxiv.org/pdf/1710.01878.pdf "link to paper")

## BibTeX

If you found this repository useful, please consider citing our work.
```
@article{iofinova2021well,
  title={How Well Do Sparse Imagenet Models Transfer?},
  author={Iofinova, Eugenia and Peste, Alexandra and Kurtz, Mark and Alistarh, Dan},
  journal={arXiv preprint arXiv:2111.13445},
  year={2021}
}
```
