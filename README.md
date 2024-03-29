# SSL Study

# Work in progress...

# Dataset

[INFO](https://github.com/cvl-umass/semi-inat-2020) about the dataset.

We have logged the entire dataset as W&B Artifacts for building easy data pipeline for our study. This also enabled us to download the dataset on any machine easily. Here's the Kaggle kernel used to log them as W&B Artifacts: [Save the Dataset as W&B Artifacts](https://www.kaggle.com/code/ayuraj/save-the-dataset-as-w-b-artifacts/notebook).

[Add chart of the dataset with associated W&B Tables view]

# Usage

### Installations

* Clone the repo: `git clone https://github.com/cosmo3769/SSL-study`
* Move into the repo: `cd SSL-study`
* Run: `python setup.py install`. If you want to develop do: `pip install -e .`
* Run: `pip install --upgrade -r requirements.txt`

### Wandb Authorization

* Run: `bash ssl_study/utils/utils.sh`

### Supervised Pipeline

To train the supervised pipeline that trains a baseline image classifier using labeled training dataset:

`python train.py --config configs/baseline.py`

* `--wandb`: Use this flag to log the metrics to Weights and Biases
* `--log_model`: Use this flag for model checkpointing. The checkpoints are logged to W&B Artifacts.
* `--log_eval`: Use this flag to log model prediction using W&B Tables.

To test your trained model, run: 

`python test.py --config configs/test_config.py`


### Sweeps

* Run: `python sweep_train.py --config configs/baseline.py`
* Run: `wandb sweep /configs/sweep_config.yaml`
* Run: `wandb agent entity-name/project-name/sweep-id`

**NOTE**

* Change the `entity-name`, `project-name`, and `sweep-id` according to your `entity-name`, `project-name`, and `sweep-id`. 
* You will get your sweep-id by running `wandb sweep /configs/sweep_config.yaml` as mentioned above.

### Tests

To run a particular test: `python -m unittest tests/test_*.py`

### SimCLRv1 pretrain

Run: `python simclrv1_pretext.py --config configs/simclrv1_pretext_config.py --wandb --log_model`

### SimCLRv1 train

Run: `python simclrv1_downstream.py --config configs/simclrv1_downstream_config.py --model_artifact_path <path/to/model/artifact>`

# Citations

```
@misc{su2021semisupervised,
      title={The Semi-Supervised iNaturalist-Aves Challenge at FGVC7 Workshop}, 
      author={Jong-Chyi Su and Subhransu Maji},
      year={2021},
      eprint={2103.06937},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
