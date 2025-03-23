import tensorflow as tf
import keras.backend as K
import numpy as np 
import tensorflow as tf
import keras.backend as K
import numpy as np 
import matplotlib.pyplot as plt
import os


def plot_predictions(predictions, save_path):
    """
    Vẽ ảnh RGB và GradCAM cho mỗi ảnh trong batch từ predictions.

    :param predictions: numpy array với shape (batch_size, H, W, 4)
                        3 channel đầu tiên là ảnh RGB, channel cuối cùng là output   
    """

    batch_size = predictions.shape[0]  
    num_columns = batch_size 

    num_rows = 2

    plt.figure(figsize=(num_columns * 3, num_rows * 3))  # Tạo một figure với kích thước tự động

    # Duyệt qua từng ảnh trong batch
    for i in range(batch_size):
        # Lấy ảnh RGB (3 kênh đầu tiên)
        rgb_image = predictions[i, :, :, :3]

        # Lấy Output (kênh cuối cùng)
        gradcam_image = predictions[i, :, :, 3]

        # Vẽ ảnh RGB
        plt.subplot(num_rows, num_columns, i + 1)  # RGB ảnh (hàng đầu)
        plt.imshow(rgb_image)
        plt.title(f"RGB {i+1}")
        plt.axis('off')

        # Vẽ ảnh GradCAM
        plt.subplot(num_rows, num_columns, i + 1 + batch_size)
        plt.imshow(gradcam_image, cmap='gray') 
        plt.title(f"Output {i+1}")
        plt.axis('off')

    # Hiển thị các ảnh
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()  # Tránh log(0)

        # Thay K.clip bằng tf.clip_by_value từ TensorFlow
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)  # Giữ giá trị trong (0,1)
        
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # Nếu đã sigmoid, chỉ cần tính xác suất đúng
        focal_weight = alpha * tf.pow((1 - pt), gamma)
        
        return -tf.reduce_mean(focal_weight * tf.math.log(pt))
    
    return loss
    
def iou(y_true, y_pred, threshold=0.5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Convert thành nhị phân
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)  # Tránh chia cho 0

def f1_score(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Chuyển thành nhị phân
    y_true = tf.cast(y_true, tf.float32)
    
    tp = tf.reduce_sum(y_true * y_pred)  # True Positives
    fp = tf.reduce_sum((1 - y_true) * y_pred)  # False Positives
    fn = tf.reduce_sum(y_true * (1 - y_pred))  # False Negatives

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

def inference_train(model, train_dataset, epochs, optimizer, loss_fn, save_path):
    for epoch in range(epochs):
        cnt = 0
        print(f"Epoch {epoch+1}/{epochs}")

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_iou = 0.0
        epoch_f1 = 0.0

        for batch in range(len(train_dataset)): 
            images, labels = train_dataset[batch]

            print("HEHE")

            # Khởi tạo channel thứ 4 với toàn số 0 cho loop đầu tiên
            zero_channel = np.zeros((images.shape[0], images.shape[1], images.shape[2], 1))

            total_loss = 0  # Biến lưu tổng loss
            with tf.GradientTape() as tape:
                for _ in range(3):
                    # Tạo input với channel thứ 4
                    images_4ch = np.concatenate([images, zero_channel], axis=-1)

                    if ((epoch + 1) % 3 == 0):
                        path = os.path.join(save_path, 'plots', str(epoch + 1) + '_' + str(cnt) + '.png')
                        cnt += 1 
                        plot_predictions(images_4ch, path)

                    # Forward pass
                    predictions = model(images_4ch, training=True)
                    loss = loss_fn(labels, predictions)
                    
                    # Cộng dồn loss
                    total_loss += loss

                    # Sử dụng output làm channel thứ 4 cho loop tiếp theo
                    zero_channel = predictions

                # Tính trung bình loss sau 3 lần lặp
                avg_loss = total_loss / 3

            # Backward pass và cập nhật trọng số
            gradients = tape.gradient(avg_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Tính toán các metric
            acc = tf.keras.metrics.binary_accuracy(labels, predictions)
            iou_score = iou(labels, predictions)
            f1 = f1_score(labels, predictions)

            epoch_loss += loss.numpy()
            epoch_acc += tf.reduce_mean(acc).numpy()
            epoch_iou += tf.reduce_mean(iou_score).numpy()
            epoch_f1 += tf.reduce_mean(f1).numpy()

        # Tính trung bình các giá trị
        num_batches = len(train_dataset)
        print(f"Loss: {epoch_loss / num_batches:.4f}, "
              f"Accuracy: {epoch_acc / num_batches:.4f}, "
              f"IoU: {epoch_iou / num_batches:.4f}, "
              f"F1-score: {epoch_f1 / num_batches:.4f}")
        
        if (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.weights.h5'))
            print("Epoch " + str(epoch + 1) + " saved!")
