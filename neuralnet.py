import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import open3d as o3d
tf.config.run_functions_eagerly(False)
print(tf.__version__)
tf.random.set_seed(1234)

# Instructing to use GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(gpu)

# get the train and test split data etc etc
# then augment
# Prepare and load the data
datadir_train = 'D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\labeled\\training\\'
datadir_val = 'D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\labeled\\testing\\'

def downsample_point_cloud(point_cloud, target_points):
    # Get the total number of points in the original point cloud
    total_points = point_cloud.shape[1]

    # Generate indices for random sampling
    sampled_indices = np.random.choice(total_points, size=target_points, replace=False)

    # Use the indices to select a subset of points
    downsampled_point_cloud = point_cloud[:, sampled_indices, :]

    return downsampled_point_cloud

# Example usage:

def parse_dataset():
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}

    folders_train = [name for name in os.listdir(datadir_train) if os.path.isdir(os.path.join(datadir_train, name))]
    folders_test = [name for name in os.listdir(datadir_val) if os.path.isdir(os.path.join(datadir_val, name))]
    for i, folder_tr in enumerate(folders_train):
        print("processing class: {}".format(os.path.basename(folder_tr)))
        class_map[i] = folder_tr.split("/")[-1]
        # gather all train files
        train_files = os.listdir(os.path.join(datadir_train, folder_tr))

        for f in train_files:
            source_tr = o3d.io.read_point_cloud(os.path.join(datadir_train, folder_tr, f))
            train_points.append(np.asarray(source_tr.points))
            train_labels.append(i)

    max_len_tr = max(len(seq) for seq in train_points)
    train_points = np.asarray([np.pad(seq, ((0, max_len_tr - len(seq)), (0, 0)), 'constant') for seq in train_points])

    for i, folder_te in enumerate(folders_test):
        print("processing class: {}".format(os.path.basename(folder_te)))
        class_map[i] = folder_te.split("/")[-1]
        # gather test files
        test_files = os.listdir(os.path.join(datadir_val, folder_te))

        for f in test_files:
            source_te = o3d.io.read_point_cloud(os.path.join(datadir_train, folder_te, f))
            test_points.append(np.asarray(source_te.points))
            test_labels.append(i)

    max_len_te = max(len(seq) for seq in test_points)
    test_points = np.asarray([np.pad(seq, ((0, max_len_te - len(seq)), (0, 0)), 'constant') for seq in test_points])

    train_points = downsample_point_cloud(train_points, target_points=2100)
    test_points = downsample_point_cloud(test_points, target_points=2100)

    max_len = max(max(len(seq) for seq in test_points), max(len(seq) for seq in train_points))

    return (
        train_points,
        test_points,
        train_labels,
        test_labels,
        class_map,
        max_len,
    )

train_points, test_points, train_labels, test_labels, CLASS_MAP, max_len = parse_dataset()
num_classes = 2
num_points = max_len
batch_size = 32


# Augment train dataset
def augment(points, label):
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    points = tf.random.shuffle(points)
    return points

with tf.device('/CPU:0'):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

# Batch the datasets
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)


# Model

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


# function to create T-net Layers
def tnet(inputs, num_features):
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalAveragePooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(num_features * num_features,
                     kernel_initializer='zeros',
                     bias_initializer=bias,
                     activity_regularizer=reg,
                     )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)

    return layers.Dot(axes=(2, 1))([inputs, feat_T])


# Creating a CNN

inputs = keras.Input(shape=(num_points, 3))
x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
print(model.summary())

# compile and train
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=50, validation_data=test_dataset)

# visualize

data = test_dataset.take(1)
points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run

preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
                                       )
    )
    ax.set_axis_off()
plt.show()
