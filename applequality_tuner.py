from typing import NamedTuple, Dict, Any, Text
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from keras_tuner.engine import base_tuner
from keras_tuner.tuners import Hyperband
from keras_tuner.engine.base_tuner import BaseTuner
import tensorflow_transform as tft
import tensorflow as tf

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', BaseTuner), ('fit_kwargs', Dict[Text, Any])])

LABEL_KEY = 'Quality'

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs=None,
             batch_size=64) -> tf.data.Dataset:
    """Get post_transform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset.repeat()
    
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.

    Args:
        fn_args: Holds args used to tune models as name/value pairs.

    Returns:
        A TunerFnResult that contains the tuner and fit_kwargs.
    """
    # Load the transformed data
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output)

    def model_builder(hp):
        hp_units = hp.Int('units', min_value=128, max_value=512, step=32)
        hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
        """Build machine learning model"""
        inputs = [
            tf.keras.Input(shape=(1,), name=transformed_name(f), dtype=tf.float64)
            for f in [
                "Acidity", "Crunchiness", "Juiciness", "Ripeness", "Size", "Sweetness", "Weight"
            ]
        ]
        concatenated_features = tf.keras.layers.concatenate(inputs)
        x = layers.Dense(64, activation='relu')(concatenated_features)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
    
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
    
    tuner = Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=fn_args.working_dir,
                     project_name='kt_hyperband')
        
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=5, verbose=1)
    
    fit_kwargs = {
        "callbacks": [early_stopping],
        'x': train_set,
        'validation_data': val_set,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)
