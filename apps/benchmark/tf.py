import tensorflow as tf


model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, 5, strides=2, activation="tanh"),
        tf.keras.layers.Conv2D(2, 5, strides=2, activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(32, activation="tanh"),
#         tf.keras.layers.Dense(10, activation=None),
#     ]
# )

if __name__ == '__main__':
    
    from pathlib import Path
    from random import shuffle
    import os
    import time
    from pydantic import BaseModel
    from PIL import Image
    import numpy as np
    
    
    class DataItem(BaseModel):
        label: str
        path: Path
      
    def load_dataset(path: Path):
        for label in os.listdir(dataset_path):
            for file in path.joinpath(label).iterdir():
                yield DataItem(label=label, path=file)
    
    cwd = Path(__file__).parent
    dataset_path = cwd.joinpath("../demo/MNIST")
    datasets = list(load_dataset(dataset_path))
    shuffle(datasets)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # loss = tf.keras.losses.binary_crossentropy()
  
    forward_time, backward_time = [], []
    for d in datasets[:2000]:
        st = time.time()
        img = Image.open(d.path)
        x = np.array(img).reshape(1, 28, 28, 1)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            z = model(x)
            y = np.zeros([1, 10])
            y[0, int(d.label)]=1
            ed = time.time()
            forward_time.append(ed - st)
            st = time.time()
            loss = tf.keras.losses.binary_crossentropy(y_true=y, y_pred=z)
        grads = tape.gradient(loss, model.variables)  
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        ed = time.time()
        backward_time.append(ed - st)
        
    print("forward: ", sum(forward_time) / len(forward_time) * 1000)
    print("backward: ", sum(backward_time) / len(backward_time) * 1000)
     