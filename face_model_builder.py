# face_model_builder.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense
)
from tensorflow.keras.models import Model


def build_face_model(num_classes=7):
    inputs = Input(shape=(224, 224, 3))

    base = ResNet50(
        include_top=False,
        weights=None,   # DO NOT load imagenet again
        input_tensor=inputs
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu", name="face_embedding")(x)
    outputs = Dense(
        num_classes,
        activation="softmax",
        name="emotion_logits"
    )(x)

    model = Model(inputs, outputs)
    return model
