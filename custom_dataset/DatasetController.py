import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_dataset.CustomDataset import *

class DatasetController:
    @staticmethod
    def get_roadtracer_train_wrapper(dataset_path, batch_size, add_channel, buffer_size=None):
        print(f"📂 Loading RoadTracer train dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        train_dataset_wrapper = RTDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,
            normalize=True,
            train=True,
            add_channel=add_channel,
            thin_label=False,
            buffer_size=buffer_size
        )
        return train_dataset_wrapper

    @staticmethod
    def get_roadtracer_test_wrapper(dataset_path, batch_size, add_channel):
        print(f"📂 Loading RoadTracer test dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        test_dataset_wrapper = RTDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,
            normalize=True,
            train=False,
            thin_label=False,
            add_channel=add_channel,
        )
        return test_dataset_wrapper

    @staticmethod
    def get_massachusetts_train_wrapper(dataset_path, batch_size, add_channel, buffer_size=None): 
        print(f"📂 Loading Massachusetts train dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")
    
        train_data_wrapper = MassachusettsDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,       
            split='train',
            add_channel=add_channel,          
            normalize=True,
            buffer_size=buffer_size
        )
        return train_data_wrapper

    @staticmethod
    def get_massachusetts_test_wrapper(dataset_path, batch_size, add_channel): 
        print(f"📂 Loading Massachusetts test dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        test_data_wrapper = MassachusettsDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,       
            split='test',
            add_channel=add_channel,          
            normalize=True,
        )
        return test_data_wrapper

    @staticmethod
    def get_drive_train_wrapper(dataset_path, batch_size, add_channel, buffer_size=None): 
        print(f"📂 Loading Drive train dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")
    
        train_data_wrapper = DRIVEDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,       
            train=True,
            add_channel=add_channel,          
            normalize=True,
            buffer_size=buffer_size
        )
        return train_data_wrapper


    @staticmethod
    def get_drive_test_wrapper(dataset_path, batch_size, add_channel): 
        print(f"📂 Loading Drive test dataset from {dataset_path}")
        print(f"📦 Batch size: {batch_size}, Add channel: {add_channel}")

        test_data_wrapper = DRIVEDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,       
            train=False,
            add_channel=add_channel,          
            normalize=True,
        )
        return test_data_wrapper
