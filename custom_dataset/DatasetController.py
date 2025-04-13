from CustomDataset import *

class DatasetController:
    @staticmethod
    def get_roadtracer_train_wrapper(dataset_path, batch_size, add_channel):
        train_dataset_wrapper = RTDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,
            normalize=True,
            train=True,
            add_channel=add_channel,
            thin_label=False
        )
        return train_dataset_wrapper

    @staticmethod
    def get_roadtracer_test_wrapper(dataset_path, batch_size, add_channel):
        test_dataset_wrapper = RTDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,
            normalize=True,
            train=False,
            thin_label=False,
            add_channel=add_channel
        )
        return test_dataset_wrapper

    @staticmethod
    def get_massachusetts_train_wrapper(dataset_path, batch_size, add_channel): 
        train_data_wrapper = MassachusettsDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,       
            split='train',
            add_channel=add_channel,          
            normalize=True
        )
        return train_data_wrapper

    @staticmethod
    def get_massachusetts_test_wrapper(dataset_path, batch_size, add_channel): 
        test_data_wrapper = MassachusettsDatasetTF(
            dataset_dir=dataset_path,
            batch_size=batch_size,       
            split='test',
            add_channel=add_channel,          
            normalize=True
        )
        return test_data_wrapper
