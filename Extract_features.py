import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from hrnet_wrapper import HRNetBackboneLayer

hrnet = load_model("hrnet_converted.keras",
                   compile=False,
                   custom_objects={"HRNetBackboneLayer": HRNetBackboneLayer})
hrnet.trainable = False

image_dir = "./Data/images"
feature_dir = "./Data/features_hrnet"
os.makedirs(feature_dir, exist_ok=True)

for fname in os.listdir(image_dir):
    if not fname.lower().endswith((".jpg", ".png")):
        continue
    # a) Load & preprocess
    img = load_img(os.path.join(image_dir, fname), target_size=(512, 512))
    x   = img_to_array(img)[None] / 255.0  # shape (1,512,512,3)
    
    # b) Extract features
    feat = hrnet.predict(x, verbose=0)     # e.g. shape (1, H, W, C)
    feat = feat[0]                         # drop batch-dim
    
    # c) Save
    #np.save(os.path.join(feature_dir, fname.replace(".jpg", ".npy")), feat)
    # after hrnet.predict → feat (H,W,C) float32
    # 1) Spatial downsample
    feat_ds = tf.keras.layers.AveragePooling2D(pool_size=2)(feat[None])[0]  
    #    now (H/4, W/4, C)

    # 2) Channel compress
    #feat_comp = tf.keras.layers.Conv2D(64,1)(feat_ds[None])[0]             
    #    now (H/4, W/4, 64)

    # 3) Cast + compress and save
    feat16 = feat_ds.numpy().astype('float16')
    np.savez_compressed(os.path.join(feature_dir, fname.replace('.jpg','.npz')),
                        features=feat16)
