import tensorflow as tf

def build_distributed_predict_step(model, strategy, training=False, iterative=1, samples=1):
    @tf.function
    def distributed_predict_step(batch):
        def step_fn(images, masks, filenames):
            images = tf.cast(images, tf.float16)
            results_per_sample = []

            for _ in tf.range(samples):
                per_iter_outputs = []
                zero_channel = tf.zeros_like(images[..., :1], dtype=tf.float16)
                x = tf.cast(images[..., :3], tf.float16)

                for _ in tf.range(iterative):
                    images_4ch = tf.concat([x, zero_channel], axis=-1)
                    y_pred = model(images_4ch, training=training)
                    per_iter_outputs.append(y_pred)
                    zero_channel = tf.cast(y_pred, tf.float16)

                results_per_sample.append(per_iter_outputs)  # shape: [iterative]

            return results_per_sample, filenames

        images, masks, filenames = batch
        return strategy.run(step_fn, args=(images, masks, filenames))

    return distributed_predict_step

def predict(data_wrapper): 
    val_dataset = data_wrapper.dataset
    print("Total images:", len(train_dataset_wrapper.image_files))
    print("Steps per epoch:", train_dataset_wrapper.steps_per_epoch)

    all_filenames = []
    all_preds = []

    for batch in tqdm(val_dataset.take(ds_loader.steps_per_epoch)):
        preds, filenames = distributed_predict_step(batch)

        # Hợp nhất từ nhiều replica
        gathered_preds = strategy.gather(preds, axis=0).numpy()
        gathered_filenames = strategy.gather(filenames, axis=0).numpy()

        all_preds.extend(gathered_preds)
        all_filenames.extend(gathered_filenames)

    return all_preds, all_filenames

