import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Reshape
from CustomLayer_v3 import SpatialSoftmax, SoftArgMaxConv
from feature_generator import FeatureDataGenerator
import json, os

import matplotlib.pyplot as plt

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

# Configuration parameters
datapath        = './Data'
general_path = "./"
# Use only reduced features (must exist)
feature_dir     = os.path.join(datapath, 'features_hrnet')
if not os.path.exists(feature_dir):
    raise FileNotFoundError(f"Reduced feature directory not found: {feature_dir}. Run extraction with reduction enabled.")
print(f"Using reduced feature directory: {feature_dir}")

annotation_file = os.path.join(datapath, 'annotations.json')
model_path      = './model/keypoint_head.keras'

if os.path.exists(feature_dir):
    feature_dir = feature_dir
    print(f"Using reduced feature directory: {feature_dir}")
elif os.path.exists(feature_dir):
    feature_dir = feature_dir
    print(f"Using raw feature directory: {feature_dir}")
else:
    raise FileNotFoundError(
        f"No feature directory found. Checked {feature_dir_reduced} and {feature_dir_raw}."
    )

annotation_file = os.path.join(datapath, 'annotations.json')
model_path      = './model/keypoint_head.keras'

BATCH_SIZE = 1
NUM_KPTS   = 1
EPOCHS     = 100

# Load and split annotations
with open(annotation_file) as f:
    data = json.load(f)
frames = data['frames']

np.random.seed(42)
indices = np.arange(len(frames))
#np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_frames = [frames[i] for i in indices[:split]]
val_frames   = [frames[i] for i in indices[split:]]

# Generators
train_gen = FeatureDataGenerator(train_frames, feature_dir, batch_size=BATCH_SIZE, num_keypoints=NUM_KPTS)
val_gen   = FeatureDataGenerator(val_frames,   feature_dir, batch_size=BATCH_SIZE, num_keypoints=NUM_KPTS)

# Infer feature map shape
def infer_shape(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.npy') or f.endswith('.npz')]
    if not files:
        raise RuntimeError(f"No feature files found in {directory}.")
    arr = np.load(os.path.join(directory, files[0]))
    feat = arr['features'] if isinstance(arr, np.lib.npyio.NpzFile) else arr
    return feat.shape

H, W, C = infer_shape(feature_dir)
print(f"Feature map shape inferred as: {H}×{W}×{C}")

# Build head-only model
def build_head_model(input_shape, num_keypoints):
    inp = layers.Input(shape=input_shape, name='feature_input')
    x = inp
    # Two residual blocks
    for _ in range(2):
        shortcut = x
        if shortcut.shape[-1] != 256:
            shortcut = Conv2D(256, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = Conv2D(256, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
    # Heatmap prediction
    x = Conv2D(num_keypoints, 1, activation=None, name='heatmap_conv')(x)
    x = layers.ReLU()(x)
    x = SpatialSoftmax(name='spatial_softmax')(x)
    x = layers.Dropout(0.3)(x)
    kp = SoftArgMaxConv(name='softargmax')(x)
    out = Reshape((num_keypoints * 2,), name='reshape_output')(kp)
    return models.Model(inputs=inp, outputs=out, name='keypoint_head')

#model = build_head_model((H, W, C), NUM_KPTS)
#model.summary()


from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


import numpy as np
from tensorflow.keras import models
from tensorflow.keras.layers import MaxPooling2D

def downsample_input(input_data):
    # Create a MaxPooling2D layer
    pooling_layer = MaxPooling2D(pool_size=(2, 2))
    
    # Apply the pooling operation
    downsampled_data = pooling_layer(input_data)
    
    return downsampled_data




# Load the best model
model = models.load_model(model_path, custom_objects={'SpatialSoftmax': SpatialSoftmax, 'SoftArgMaxConv': SoftArgMaxConv})
model.summary()

# Get a single validation sample
val_sample = val_gen[0]  # Get the first batch from the validation generator
val_features, _ = val_sample  # Unpack features and labels (assuming labels are not needed for prediction)
print(_)
val_features_downsampled = downsample_input(val_features)

# Make a prediction
prediction = model.predict(val_features_downsampled)

# Print the prediction
print("Prediction:", prediction)

# Plot several channels of the feature map
# Assuming the feature map is the output of the last Conv2D layer before the softmax
feature_map = model.layers[-3].output  # Get the output of the last Conv2D layer
#feature_model = models.Model(inputs=model.input, outputs=feature_map)
feature_model = models.Model(inputs=model.input, outputs=model.input)


# Get the feature map for the first sample
#feature_map_output = feature_model.predict(val_features_downsampled)

feature_map_output = val_sample[0]
print(val_sample)

# Plotting the feature map channels
num_channels_to_plot = 5  # Number of channels to plot
plt.figure(figsize=(15, 10))
for i in range(num_channels_to_plot):
    plt.subplot(2, 3, i + 1)
    plt.imshow(feature_map_output[0, :, :, i], cmap='jet')  # Display the i-th channel
    plt.title(f'Channel {i + 1}')
    plt.axis('off')

plt.suptitle('Feature Map Channels', fontsize=16)
plt.tight_layout()
plt.savefig(general_path + "testing15.png")
plt.close()
