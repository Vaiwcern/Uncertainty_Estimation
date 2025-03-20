import tensorflow as tf
import keras.backend as K
import numpy as np 

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()  # Tránh log(0)
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)  # Giữ giá trị trong (0,1)
        
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # Nếu đã sigmoid, chỉ cần tính xác suất đúng
        focal_weight = alpha * K.pow((1 - pt), gamma)
        
        return -K.mean(focal_weight * K.log(pt))
    
    return loss
    
def iou(y_true, y_pred, threshold=0.5):
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

def inference_train(model, train_dataset, epochs, optimizer, loss_fn):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_iou = 0.0
        epoch_f1 = 0.0

        for batch in range(len(train_dataset)): 
            images, labels = train_dataset[batch]

            # Khởi tạo channel thứ 4 với toàn số 0 cho loop đầu tiên
            zero_channel = np.zeros((images.shape[0], images.shape[1], images.shape[2], 1))
            
            total_loss = 0  # Biến lưu tổng loss
            with tf.GradientTape() as tape:
                for _ in range(3):
                    # Tạo input với channel thứ 4
                    images_4ch = np.concatenate([images, zero_channel], axis=-1)

                    print(images_4ch.shape)

                    # Forward pass
                    predictions = model(images_4ch, training=True)
                    loss = loss_fn(labels, predictions)
                    
                    # Cộng dồn loss
                    total_loss += loss

                    # Sử dụng output làm channel thứ 4 cho loop tiếp theo
                    zero_channel = predictions[..., np.newaxis]  

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
