import wandb
import os
import glob
import json
import pandas as pd
from tqdm import tqdm

def download_dataset(dataset_name: str, dataset_type: str, version: str='latest'):
    """
    Utility function to download the data saved as W&B artifacts and return a dataframe
    with path to the dataset and associated label.

    Args:
        dataset_name (str): The name of the dataset - `train`, `val`, `out-class`, and `in-class`.
        dataset_type (str): The type of the dataset - `labelled-dataset`, `unlabelled-dataset`.
        version (str): The version of the dataset to be downloaded. By default it's `latest`,
            but you can provide different version as `vX`, where, X can be 0,1,...
            
        Note that the following combination of dataset_name and dataset_type are valid:
            - `train`, `labelled-dataset`
            - `val`, `labelled-dataset`
            - `in-class`, `unlabelled-dataset`
            - `out-class`, `unlabelled-dataset`

    Return:
        df_data (pandas.DataFrame): Dataframe with path to images with associated labels if present.
    """
    # Download the dataset.
    wandb_api = wandb.Api()
    artifact = wandb_api.artifact(f'wandb_fc/ssl-study/{dataset_name}:{version}', type=dataset_type)
    artifact_dir = artifact.download()

    # Open the W&B table downloaded as a json file.
    json_file = glob.glob(artifact_dir+'/*.json')
    assert len(json_file) == 1
    with open(json_file[0]) as f:
        data = json.loads(f.read())
        assert data['_type'] == 'table'
        columns = data['columns']
        data = data['data']

    # Create a dataframe with path and label
    df_columns = ['image_id', 'image_path', 'width', 'height']
    if 'label' in columns:
        df_columns+=['label']
    data_df = pd.DataFrame(columns=df_columns)
    
    for idx, example in tqdm(enumerate(data)):
        image_id = int(example[0])
        image_dict = example[1]
        image_path = os.path.join(artifact_dir, image_dict.get('path'))
        height = image_dict.get('height')
        width = image_dict.get('width')

        df_data = [image_id, image_path, width, height]
        if 'label' in columns:
            df_data+=[example[2]]
        data_df.loc[idx] = df_data

    # Shuffle the dataframe
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    
    # Assign correct dtypes
    data_df[['image_id', 'width', 'height']] = data_df[['image_id', 'width', 'height']].apply(pd.to_numeric)
    if 'label' in columns:
        data_df[['label']] = data_df[['label']].apply(pd.to_numeric)

    return data_df

train_df = download_dataset('train', 'labelled-dataset')
valid_df = download_dataset('val', 'labelled-dataset')
test_df = download_dataset('test', 'labelled-dataset')