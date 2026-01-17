"""
Center Loss Implementation for Keras/TensorFlow.

Center Loss learns a center for each class and penalizes the distance between
deep features and their corresponding class centers. This encourages intra-class
compactness in the feature space.

Reference:
    Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition"
    ECCV 2016, https://ydwen.github.io/papers/WenECCV16.pdf
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
from keras import backend as K


class CenterLossLayer(Layer):
    """
    Center Loss Layer that computes the center loss and updates class centers.
    
    The center loss is defined as:
        L_center = (1/2N) * sum_i ||f_i - c_{y_i}||^2
    
    where f_i is the feature of sample i, and c_{y_i} is the center of class y_i.
    
    Args:
        num_classes: Number of classes
        feature_dim: Dimension of the feature vector (from the layer before softmax)
        alpha: Learning rate for center updates (0 to 1). Higher = faster updates.
        
    Inputs:
        [features, labels]: features is (batch, feature_dim), labels is (batch, num_classes) one-hot
        
    Returns:
        center_loss: Scalar tensor representing the center loss
    """
    
    def __init__(self, num_classes, feature_dim, alpha=0.5, **kwargs):
        super(CenterLossLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
    def build(self, input_shape):
        # Initialize class centers as trainable weights
        # Shape: (num_classes, feature_dim)
        self.centers = self.add_weight(
            name='centers',
            shape=(self.num_classes, self.feature_dim),
            initializer='zeros',
            trainable=False  # We update centers manually, not via gradients
        )
        super(CenterLossLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Compute center loss and update centers during training.
        
        Args:
            inputs: List of [features, labels]
                - features: (batch_size, feature_dim)
                - labels: (batch_size, num_classes) one-hot encoded
        """
        features, labels = inputs
        
        # Convert one-hot labels to class indices
        # Shape: (batch_size,)
        label_indices = tf.argmax(labels, axis=1)
        
        # Get the centers for each sample in the batch
        # Shape: (batch_size, feature_dim)
        centers_batch = tf.gather(self.centers, label_indices)
        
        # Compute the difference between features and their centers
        diff = features - centers_batch
        
        # Compute center loss: mean of squared L2 distances
        # L_center = (1/2N) * sum ||f_i - c_{y_i}||^2
        center_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1)) / 2.0
        
        # Update centers during training
        if training:
            # Compute the delta for each center
            # We want to move each center toward the mean of its class samples
            
            # Compute the sum of (feature - center) for each class
            # We use unsorted_segment_sum for efficiency
            # Shape: (num_classes, feature_dim)
            diff_by_center = tf.math.unsorted_segment_sum(
                diff, 
                label_indices, 
                num_segments=self.num_classes
            )
            
            # Compute counts per class (need to handle classes not in batch)
            counts_per_class = tf.math.unsorted_segment_sum(
                tf.ones_like(label_indices, dtype=tf.float32),
                label_indices,
                num_segments=self.num_classes
            )
            # Add small epsilon to avoid division by zero
            counts_per_class = tf.maximum(counts_per_class, 1.0)
            
            # Compute the update: alpha * (sum of diffs) / count
            # Shape: (num_classes, feature_dim)
            center_updates = self.alpha * diff_by_center / tf.expand_dims(counts_per_class, 1)
            
            # Update centers
            self.centers.assign_add(center_updates)
        
        return center_loss
    
    def compute_output_shape(self, input_shape):
        return ()  # Scalar output
    
    def get_config(self):
        config = super(CenterLossLayer, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'alpha': self.alpha
        })
        return config


def combined_loss_with_center(center_loss_weight=0.01):
    """
    Factory function that creates a combined loss (cross-entropy + center loss).
    
    Note: This is meant to be used when the model has two outputs:
        1. Softmax predictions
        2. Center loss value (from CenterLossLayer)
    
    Args:
        center_loss_weight: Lambda (λ) weight for center loss term
        
    Returns:
        Dictionary of losses for each output
    """
    def identity_loss(y_true, y_pred):
        """
        Identity loss - just returns the predicted value.
        Used for center loss output where loss is computed in the layer.
        """
        return center_loss_weight * y_pred
    
    return {
        'softmax': 'categorical_crossentropy',
        'center_loss': identity_loss
    }


def get_center_loss_model_losses(center_loss_weight=0.01):
    """
    Get the loss configuration for a model with center loss.
    
    Args:
        center_loss_weight: Lambda (λ) weight for center loss term
        
    Returns:
        Tuple of (loss_dict, loss_weights_dict)
    """
    def dummy_loss(y_true, y_pred):
        """Returns the precomputed center loss from the model output."""
        return y_pred
    
    losses = {
        'softmax': 'categorical_crossentropy',
        'center_loss_output': dummy_loss  # Must match model output name
    }
    
    loss_weights = {
        'softmax': 1.0,
        'center_loss_output': center_loss_weight  # Must match model output name
    }
    
    return losses, loss_weights
