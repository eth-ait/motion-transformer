# Structured Prediction Helps 3D Human Motion Modelling 
Code repository for [our paper](https://ait.ethz.ch/projects/2019/spl/) presented at ICCV '19. 

We provide data preprocessing scripts, training pipeline, evaluation and visualization tools. Model implementation and pre-trained models will come soon. 

### Required packages
We recommend creating a virtual environment and install the required packages by running:
```
pip install -r requirements.txt
```
We used `Tensorflow 1.12.0` in our experiments. We also check that Tensorflow versions until `1.14.0` are also okay, but gives too many warnings due to TF 2.0. 
Please note that having `numpy 1.14.5` is important. If you use other versions of TF or other packages, make sure that you have the correct numpy version.  

### Preparing the Data
Download the data from the [DIP website](http://dip.is.tue.mpg.de/) and unzip it into a folder of your choice. Let's call that folder `<RAW_DATA>`. Create a folder where you want to store the processed data, `<SPL_DATA>`. In the folder [`preprocessing`](./preprocessing), run the script

```
cd preprocessing
python preprocess_dip.py --input_dir <RAW_DATA> --output_dir <SPL_DATA>
```

By default the script generates the data using rotation matrix representations. If you want to convert the data to angle-axis or quaternions, use the `--as_aa` or `--as_quat` flags.

This script creates the training, validation and test splits used to produce the results in the paper. Note that data split is deterministic and determined by the files `training_fnames.txt`, `validation_fnames.txt`, and `test_fnames.txt` under [`preprocessing`](./preprocessing).

When running the script it creates two versions of the validation and test split: One where we split each motion sequence into subsequences of size 180 (3 seconds) using a sliding window and one where we do not split the sequence (referred to as `dynamic` split). When we load data during training or evaluation, we always only extract one window of size `W` from each sequence. Hence, the splitting with a sliding window ''blows up'' the number of samples. Thus, the dynamic split has effectively less samples, which is sometimes convenient (for debugging, visualization etc.).

A note on the data: The data published on the DIP website is an early version of the official AMASS dataset. When we submitted the paper, the official [AMASS dataset](https://amass.is.tue.mpg.de/) was not published yet. We are planning to evaluate our model and baseline models on the official AMASS dataset and report results here. 
If you plan to use the latest version of AMASS, we are happy to provide assistance if required. However, it shouldn't be too hard to adapt `preprocess_dip.py` to parse the AMASS data. 

### Training
You can pass data and save directory via command-line arguments everytime you run an experiment. Alternatively, you can set `AMASS_DATA` and `AMASS_EXPERIMENTS` environment variables. You can run the following commands:
```
export AMASS_DATA=<SPL_DATA>
export AMASS_EXPERIMENTS=<path-to-experiment-directory>
export PYTHONPATH=$PYTHONPATH:<path-to-this-repository>
```
Please note that updating `PYTHONPATH` is required while `AMASS_DATA` and `AMASS_EXPERIMENTS` are optional.

You can train `zero_velocity` model by using rotation matrix representation as follows: 
```
cd <path-to-this-repository>
python spl/training.py --model_type zero_velocity --data_type rotmat
```
With a unique timestamp, the experiment is stored under `AMASS_EXPERIMENTS` or the given target directory if you run the training command with `--data_dir` flag.
See flags and possible choices in `spl/training.py`. We will add our model and the baselines soon.

You can easily extend this repo by implementing a new model. Please see the docstring of `spl.model.base_model.py` to read about the interface.

### Evaluation
You can evaluate and/or visualize models after training. The following command visualizes clips of 60 frames by evaluating the model on the test dataset with full sequences.
See flags and possible choices in `spl/evaluation.py`. 
```
python spl/evaluation.py --model_id <experiment-timestamp> --visualize --seq_length_out 60 --dynamic_test_split
```

Please note that by default the visualization code displays interactive animations using matplotlib. To make interactive frame-rates possible, only the skeleton is displayed. You can also create videos of the full SMPL mesh or skeleton by adding the `--to_video` option. However, in order to get videos with SMPL mesh, you need the SMPL model, which we cannot provide due to licensing issues. If you are interested in using SMPL, the best option is to download the latest code from the AMASS repo and integrate it with our repo. Feel free to contact us if you have questions about this.

### Pre-trained Models
In `pretrained_configs` folder, you can find the configuration we used. In order to re-run an experiment you can simply run:
```
python spl/training.py --from_config <path-to-a-model-config.json>
``` 
Due to the stochastic nature of training, you many not get exactly the same results. 
However, you should get marginally better or worse models. If this is not the case, please contact us. 
The models we used in the paper can be [downloaded from here](https://ait.ethz.ch/projects/2019/spl/downloads/spl_models.zip).
You can run evaluation with them or visualize their results. Note that `QuaterNet` models are not there yet. 

### Sample scripts
Under `spl/test/`, we share sample scripts showing how to use components (i.e., metrics, visualization, tfrecord data) of this repository without requiring the entire pipeline.   

### Citation
If you use code from this repository, please cite 

```
@inproceedings{Aksan_2019_ICCV,
  title={Structured Prediction Helps 3D Human Motion Modelling},
  author={Aksan, Emre and Kaufmann, Manuel and Hilliges, Otmar},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2019},
  note={First two authors contributed equally.}
}
```

If you use data from DIP or AMASS, please cite the original papers as detailed on their website.

### Contact
Please file an issue or contact [Emre Aksan (emre.aksan@inf.ethz.ch)](mailto:emre.aksan@inf.ethz.ch) or [Manuel Kaufmann (manuel.kaufmann@inf.ethz.ch)](mailto:manuel.kaufmann@inf.ethz.ch)
