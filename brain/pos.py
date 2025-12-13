from keras import layers
import keras
import numpy as np

# Configuration
positional_emb = True
conv_layers = 2
projection_dim = 128
num_heads = 2
transformer_units = [projection_dim, projection_dim]
transformer_layers = 2
stochastic_depth_rate = 0.1
image_size = 32
num_classes = 2
input_shape = (32, 32, 3)

# Tokenizer
class CCTTokenizer(layers.Layer):
    def __init__(self,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 num_conv_layers=conv_layers,
                 num_output_channels=[64, 128],
                 **kwargs):
        super().__init__(**kwargs)
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(layers.Conv2D(
                num_output_channels[i], kernel_size, stride,
                padding="valid", use_bias=False,
                activation="relu", kernel_initializer="he_normal"
            ))
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(layers.MaxPooling2D(pooling_kernel_size, pooling_stride, "same"))

    def call(self, images):
        outputs = self.conv_model(images)
        reshaped = keras.ops.reshape(outputs, (
            -1,
            keras.ops.shape(outputs)[1] * keras.ops.shape(outputs)[2],
            keras.ops.shape(outputs)[-1],
        ))
        return reshaped

# Positional Embedding
class PositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = keras.ops.shape(inputs)
        sequence_length = shape[-2]
        feature_length = shape[-1]
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(position_embeddings, (start_index, 0), (sequence_length, feature_length))
        return keras.ops.broadcast_to(position_embeddings, shape)

# Sequence Pooling
class SequencePooling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = layers.Dense(1)

    def call(self, x):
        attention_weights = keras.ops.softmax(self.attention(x), axis=1)
        attention_weights = keras.ops.transpose(attention_weights, axes=(0, 2, 1))
        weighted_representation = keras.ops.matmul(attention_weights, x)
        return keras.ops.squeeze(weighted_representation, -2)

# Stochastic Depth
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(shape, 0, 1, seed=self.seed_generator)
            random_tensor = keras.ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

# MLP block
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.ops.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Model Creator
def create_cct_model(
    image_size=image_size,
    input_shape=input_shape,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
    num_classes=num_classes
):
    inputs = layers.Input(input_shape)
    resized_inputs = layers.Resizing(image_size, image_size)(inputs)

    # Tokenize image into patch sequence
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(resized_inputs)

    # Add positional embeddings
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(encoded_patches)

    # Transformer layers
    dpr = np.linspace(0, stochastic_depth_rate, transformer_layers)
    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
        x3 = mlp(x3, transformer_units, dropout_rate=0.1)
        x3 = StochasticDepth(dpr[i])(x3)

        encoded_patches = layers.Add()([x3, x2])

    # Sequence pooling and final classification
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    pooled = SequencePooling()(representation)
    logits = layers.Dense(num_classes)(pooled)

    return keras.Model(inputs=inputs, outputs=logits)
