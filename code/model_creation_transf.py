import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# ------------------------------------------------------------------
# Utility: simple LR scheduler (unchanged)
# ------------------------------------------------------------------

def lr_schedule(epoch, lr):
    if epoch % 50 == 0 and epoch > 0:
        return lr * 0.9
    return lr


# ------------------------------------------------------------------
# Main Model
# ------------------------------------------------------------------

class TransformerModel(Model):
    """A ViT‑style 1‑D Transformer classifier with CLS token and sinusoidal PE."""

    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        *,
        reg_type: str = "l2",
        reg_value: float = 1e-3,
        return_logits: bool = False,
        d_model: int = 256,
        num_heads: int = 8,
        dff: int = 1024,
        num_layers: int = 6,
        dropout_rate: float = 0.2,
        patch_size: int = 4,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Validate & store settings
        # ------------------------------------------------------------------
        self.d_model = d_model
        self.return_logits = return_logits

        if reg_type == "l2":
            reg = regularizers.l2(reg_value)
        elif reg_type == "l1":
            reg = regularizers.l1(reg_value)
        else:
            raise ValueError("reg_type must be 'l1' or 'l2'.")

        seq_len, feat_dim = input_shape
        num_patches = math.ceil(seq_len / patch_size)

        # ------------------------------------------------------------------
        # Stem: 1‑D patch embedding (Conv1D)
        # ------------------------------------------------------------------
        self.patch_embed = Conv1D(
            filters=d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding="same",
            kernel_regularizer=reg,
            name="patch_embed",
        )

        # CLS token (learnable)
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, d_model),
            initializer="random_normal",
            trainable=True,
        )

        # Embedding dropout
        self.embed_dropout = Dropout(dropout_rate)

        # ------------------------------------------------------------------
        # Encoder blocks
        # ------------------------------------------------------------------
        self.blocks = []
        for i in range(num_layers):
            self.blocks.append({
                "ln1": LayerNormalization(epsilon=1e-6),
                "mha": MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate),
                "dp1": Dropout(dropout_rate),
                "ln2": LayerNormalization(epsilon=1e-6),
                "ff1": Dense(dff, activation="gelu", kernel_regularizer=reg),
                "ff2": Dense(d_model, kernel_regularizer=reg),
                "dp2": Dropout(dropout_rate),
            })

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        self.head_norm = LayerNormalization(epsilon=1e-6)
        self.head_dense = Dense(d_model, activation="gelu", kernel_regularizer=reg)
        self.head_dropout = Dropout(dropout_rate)
        self.head_out = (
            Dense(num_classes, kernel_regularizer=reg)
            if return_logits
            else Dense(num_classes, activation="softmax", kernel_regularizer=reg)
        )

    # ----------------------------------------------------------------------
    # Positional encoding (sinusoidal, computed on‑the‑fly to avoid shape bugs)
    # ----------------------------------------------------------------------
    def _pos_encoding(self, length: tf.Tensor) -> tf.Tensor:
        pos = tf.cast(tf.range(length)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(self.d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        return tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]  # (1, length, d_model)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, inputs, training=False):
        # Patchify + linear projection
        x = self.patch_embed(inputs)  # (batch, n_patches, d_model)

        # Append CLS token
        batch_size = tf.shape(x)[0]
        cls = tf.broadcast_to(self.cls_token, [batch_size, 1, self.d_model])
        x = tf.concat([cls, x], axis=1)  # (batch, 1 + n_patches, d_model)

        # Add positional encodings
        x += self._pos_encoding(tf.shape(x)[1])
        x = self.embed_dropout(x, training=training)

        # Encoder blocks
        for blk in self.blocks:
            y = blk["ln1"](x)
            y = blk["mha"](y, y, y, training=training)
            y = blk["dp1"](y, training=training)
            x = x + y  # residual 1

            y = blk["ln2"](x)
            y = blk["ff1"](y)
            y = blk["ff2"](y)
            y = blk["dp2"](y, training=training)
            x = x + y  # residual 2

        # Classification head on CLS token
        cls_token_final = x[:, 0]  # (batch, d_model)
        x = self.head_norm(cls_token_final)
        x = self.head_dense(x)
        x = self.head_dropout(x, training=training)
        return self.head_out(x)

    # ------------------------------------------------------------------
    # Compile helper
    # ------------------------------------------------------------------
    def compile_model(self):
        loss = (
            SparseCategoricalCrossentropy(from_logits=True)
            if self.return_logits
            else "sparse_categorical_crossentropy"
        )
        self.compile(optimizer=Adam(learning_rate=5e-3), loss=loss, metrics=["accuracy"])
