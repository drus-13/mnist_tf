import numpy as np
import os
import tensorflow as tf

IMG_SIZE = 28

class Model(tf.Module):
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE), name='flatten'),
            tf.keras.layers.Dense(
                units=10,
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                bias_initializer=tf.keras.initializers.Ones(),
                name='dense'
            ),
        ])

        opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    # The `train` function takes a batch of input images and labels.
    @tf.function(input_signature=[
        tf.TensorSpec([32, IMG_SIZE, IMG_SIZE], tf.float32),
        tf.TensorSpec([32, 10], tf.float32),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x)
            loss = self.model.loss(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        return result

    @tf.function(input_signature=[
        tf.TensorSpec([1, IMG_SIZE, IMG_SIZE], tf.float32),
    ])
    def infer(self, x):
        logits = self.model(x)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "output": probabilities,
            "logits": logits
        }
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path
        }

m = Model()

def save_model():
    model_path = R"<insert your path>"

    tf.saved_model.save(
        m,
        model_path,
        signatures={
            'train':
                m.train.get_concrete_function(),
            'infer':
                m.infer.get_concrete_function(),
            'save':
            	m.save.get_concrete_function(),
        })

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    with open(os.path.join(model_path, "model.tflite"), 'wb') as model_writer:
        model_writer.write(tflite_model)

save_model()