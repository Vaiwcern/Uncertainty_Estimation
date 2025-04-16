import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
import keras
from custom_dataset.CustomDataset import RTDatasetTF, MassachusettsDatasetTF
from model.unet import StandardUNet
from utils import focal_loss, IoUMetric, F1ScoreMetric
import tensorflow as tf
import numpy as np
import imageio
from tqdm import tqdm

from seggradcam.training_write import TrainingParameters, TrainingResults
from seggradcam.training_plots import plot_predict_and_gt, plot_loss, plot_metric
from seggradcam.seggradcam import SegGradCAM, SuperRoI, ClassRoI, PixelRoI, BiasRoI
from seggradcam.visualize_sgc import SegGradCAMplot

def parse_args():
    # Tạo đối số dòng lệnh để nhận GPU
    parser = argparse.ArgumentParser(description='Train Unet model on specific GPUs.')
    parser.add_argument('--gpus', type=str, required=True, help='Comma separated list of GPU IDs to use (e.g., "5,6").')
    return parser.parse_args()

def convert_to_functional(model, input_shape=(1024, 1024, 4)):
    inputs = keras.Input(shape=input_shape)
    outputs = model(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)

def predict_and_save(model, dataset, image_files, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    idx = 0  # Chỉ số của ảnh gốc
    total = len(image_files)

    for (x, _) in tqdm(dataset, desc="Predicting"):
        if idx >= total:
            break  # Ngăn vòng lặp chạy mãi do repeat()

        x_orig = x[..., :3].numpy()  # Bỏ channel thứ 4, về numpy
        zero_channel = tf.zeros_like(x[..., :1])

        outputs = []
        batch_size = x.shape[0]
        # gradcams_by_batch = [[] for _ in range(batch_size)]

        for i in range(3):
            x_4ch = tf.concat([x[..., :3], zero_channel], axis=-1)
            y_pred = model(x_4ch, training=False)
            outputs.append(y_pred)
            zero_channel = y_pred

            # for j in range(batch_size): 
            #     prop_from_layer = model.layers[-1].name
            #     prop_to_layer = 'center_block'
            #     cls = 0

            #     clsroi = ClassRoI(model=model, image=x_4ch[j], cls=cls)
            #     newsgc = SegGradCAM(model, x_4ch[j], cls, prop_to_layer, prop_from_layer, roi=clsroi,
            #                         normalize=True, abs_w=False, posit_w=False)
            #     mymap = newsgc.SGC()  # Heatmap với shape (H, W)
            #     gradcams_by_batch[j].append(mymap)

        for b in range(batch_size):
            if idx >= len(image_files):
                break

            image_name = os.path.basename(image_files[idx])
            name_without_ext = os.path.splitext(image_name)[0]

            # Lưu ảnh gốc (convert về [0,255])
            ori_image = (x_orig[b] * 255).astype("uint8")
            ori_path = os.path.join(save_dir, f"{name_without_ext}_input.png")
            imageio.imwrite(ori_path, ori_image)

            # Lưu từng output mà không threshold
            for i in range(3):
                pred = outputs[i][b]  # (H, W, 1)
                pred = tf.squeeze(pred, axis=-1)
                pred = (pred * 255).numpy().astype("uint8")  # scale lên [0, 255] mà không threshold
                output_path = os.path.join(save_dir, f"{name_without_ext}_output_{i}.png")
                imageio.imwrite(output_path, pred)

                # grad = gradcams_by_batch[b][i]
                # if isinstance(grad, tf.Tensor):
                #     grad = grad.numpy()
                # grad = (grad * 255).astype("uint8")
                # grad_path = os.path.join(save_dir, f"{name_without_ext}_grad_{i}.png")
                # imageio.imwrite(grad_path, grad)

            idx += 1

if __name__ == "__main__":
    # Lấy đối số GPU từ dòng lệnh
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES để chọn GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    BATCH_SIZE = 16
    LR = 1e-3
    EPOCHS = 100

    # Ensure steps_per_epoch and validation_steps are integers
    trainparam = SimpleNamespace(
        dataset_name="RoadTracer",
        learning_rate=LR,
        n_classes=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        input_shape=(1024, 1024, 4),
        save_path="/home/ltnghia02/MEDICAL_ITERATIVE/model/RTdata_iterative_model"
    )

    # TRAIN
    log_file_path = os.path.join(trainparam.save_path, "log.txt")
    os.makedirs(trainparam.save_path, exist_ok=True)  # Đảm bảo thư mục tồn tại

    sys.stdout = open(log_file_path, "w")
    sys.stderr = sys.stdout

    train_dataset_wrapper = RTDatasetTF(
        dataset_dir="/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop",
        batch_size=trainparam.batch_size,
        normalize=True,
        train=True,
        thin_label=False
    )

    train_dataset = train_dataset_wrapper.dataset

    print("Total images:", len(train_dataset_wrapper.image_files))
    print("Steps per epoch:", train_dataset_wrapper.steps_per_epoch)

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = StandardUNet(input_channels=4, dropout_rate=0.0)
        optim = keras.optimizers.Adam(learning_rate=trainparam.learning_rate)
        model.compile(
            optimizer=optim,
            loss=focal_loss(),
            metrics=[
                'accuracy',
                IoUMetric(),
                F1ScoreMetric()
            ]
        )
        
        model.build(input_shape=(None, 1024, 1024, 4))

        dummy_x = tf.random.normal((1, 1024, 1024, 4))
        _ = model(dummy_x)


    model.fit(
        train_dataset,  
        epochs=trainparam.epochs,
        steps_per_epoch=train_dataset_wrapper.steps_per_epoch,
        callbacks=[
            PrintLossCallback(),
            SaveEveryNEpoch(save_path=trainparam.save_path, interval=1)
        ]
    )

    # PREDICT 
    # log_file_path = os.path.join(trainparam.save_path, "log_predict_20.txt")
    # os.makedirs(trainparam.save_path, exist_ok=True)  # Đảm bảo thư mục tồn tại

    # sys.stdout = open(log_file_path, "w")
    # sys.stderr = sys.stdout

    # test_dataset_wrapper = RTDatasetTF(
    #     dataset_dir="/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/RTdata_Crop",
    #     batch_size=trainparam.batch_size,
    #     normalize=True,
    #     train=False,
    #     thin_label=False
    # )

    # test_dataset = test_dataset_wrapper.dataset
    # image_files = [str(p) for p in test_dataset_wrapper.image_files]

    # print("Total images:", len(test_dataset_wrapper.image_files))
    # print("Steps per epoch:", test_dataset_wrapper.steps_per_epoch)

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     model = StandardUNet(input_channels=4, dropout_rate=0.0)
    #     model.build((None, 1024, 1024, 4))
    #     model.load_weights(os.path.join(trainparam.save_path, "model_epoch_20.weights.h5"))
    #     model = convert_to_functional(model, input_shape=(1024, 1024, 4))
    
    # predict_and_save(model, dataset=test_dataset, image_files=image_files, save_dir=os.path.join(trainparam.save_path, "predict_epoch_20"))


    # PREDICT massachusetts
    # log_file_path = os.path.join(trainparam.save_path, "log_predict_mass.txt")
    # os.makedirs(trainparam.save_path, exist_ok=True)  # Đảm bảo thư mục tồn tại

    # sys.stdout = open(log_file_path, "w")
    # sys.stderr = sys.stdout

    # test_dataset_wrapper = MassachusettsDatasetTF(
    #     dataset_dir="/home/ltnghia02/MEDICAL_ITERATIVE/Dataset/Massachusetts_Crop",
    #     batch_size=BATCH_SIZE,       
    #     split='test',
    #     channel=4,          
    #     normalize=True
    # )

    # test_dataset = test_dataset_wrapper.dataset
    # image_files = [str(p) for p in test_dataset_wrapper.image_files]

    # print("Total images:", len(test_dataset_wrapper.image_files))
    # print("Steps per epoch:", test_dataset_wrapper.steps_per_epoch)

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     model = StandardUNet(input_channels=4, dropout_rate=0.0)
    #     model.build((None, 512, 512, 4))
    #     model.load_weights(os.path.join(trainparam.save_path, "model_epoch_20.weights.h5"))
    #     model = convert_to_functional(model, input_shape=(512, 512, 4))
    
    # predict_and_save(model, dataset=test_dataset, image_files=image_files, save_dir=os.path.join(trainparam.save_path, "predict_epoch_20_mass"))
