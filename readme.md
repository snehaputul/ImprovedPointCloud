# Course Project: Comp6321
### Sneha Paul
#### Student ID: 40200126

## Project Structure
```text
experiments/  *** includes all modified experiments ***
criterions/   *** all the loss functions ***
datasets/     *** all the datasets ***
main.py
readme.md
```

## Datasets
Two datasets are used in this project: <br>

| Dataset Name | Download Link                                                                             |
|--------------|-------------------------------------------------------------------------------------------|
| ModelNet10   | <a href='http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'>Link</a> |
| ModelNet40   | <a href='http://modelnet.cs.princeton.edu/ModelNet40.zip'>Link</a>                        |

Please download the dataset and unzip it in the datasets folder with following structure:


```text 
datasets/
    ModelNet10/
        bathtub/
            train/
                <.off files>
            test/
                <.off files>
        bed/
    ModelNet40/
        ....
```

## How to run the code
To run the final best result of the mehtod, use the following script. This will give an accuracy of 91.52%.
 
```bash
python main.py
```



For different ablation, sensitivity analysis on hyper-parameters, and other experiments with model architecture, use the scripts in the <a href='experiments'>experiments</a> folder.

```bash
python experiments/lr_0.1.py
```

The experiment directory contains files for different experiments.
```text
batch_size_8.py
batch_size_16.py
batch_size_32.py

.....
```

ALl the experiments' outputs are saved in the  <a href='checkpoints'>checkpoints</a> folder.
```text
batch_size_8.txt
batch_size_16.txt
batch_size_32.txt

.....
```

Most of the file names are intuitive and indicative of the experiments. The files that require description are:

```text
main_exp_1.py to main_exp_10.py --> experiments with different version of 1D CNNs. Description is included in each file.
```

## Run time
The code requires a Colab GPU machine around 10 hours to run the default 15 epochs settings.

## Acknowledgements
This project was build using the help from the following sources and repositories: <br>
https://github.com/fxia22/pointnet.pytorch  --> PointNet Implementation <br>
https://github.com/nikitakaraevv/pointnet  --> PointNet Implementation <br>
https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/22  --> Focul loss <br>

