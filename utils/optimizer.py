# utils/optimizer.py

import tensorflow as tf
import os

# Optional: Pruning
try:
    import tensorflow_model_optimization as tfmot
except ImportError:
    tfmot = None


def apply_pruning(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32):
    """
    Applies weight pruning to the model using TensorFlow Model Optimization Toolkit.
    """
    if tfmot is None:
        raise ImportError("TensorFlow Model Optimization Toolkit not installed. Run: pip install tensorflow-model-optimization")

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.2,
            final_sparsity=0.8,
            begin_step=0,
            end_step=np.ceil(len(X_train) / batch_size).astype(np.int32) * epochs
        )
    }

    # Apply pruning
    model_pruned = prune_low_magnitude(model, **pruning_params)

    # Compile pruned model
    model_pruned.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    # Pruning callback
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    # Fine-tune pruned model
    model_pruned.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     batch_size=batch_size,
                     epochs=epochs,
                     callbacks=callbacks)

    # Strip pruning wrappers
    model_final = tfmot.sparsity.keras.strip_pruning(model_pruned)
    return model_final


def convert_to_tflite(model, output_path="model_quant.tflite"):
    """
    Converts a trained model to a quantized TFLite format.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    # Save the model
    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)

    return output_path
