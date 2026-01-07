# %%
"""
TCPL Modules: Task-Conditioned Prompt Learning for Few-Shot Cross-Subject EEG Decoding

Based on: Wang et al. 2025, "TCPL: Task-Conditioned Prompt Learning for 
Few-Shot Cross-Subject Motor Imagery EEG Decoding"
DOI: 10.3389/fnins.2025.1689286

This module implements:
1. SupportEncoder: Encodes few-shot support samples into embeddings
2. PromptGenerator: Generates subject-specific prompt tokens from embeddings
3. TCPModule: Combines encoder + generator for end-to-end prompt generation
4. prompt_augmented_mha: Modified MHA that accepts prompt tokens
5. TCPL_Backbone: 4-layer Transformer with Deep Prompt Injection
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Conv1D, Dense, Dropout, LayerNormalization, 
    GlobalAveragePooling1D, MultiHeadAttention, Add, Reshape, Concatenate, Lambda
)
from keras import backend as K


class SupportEncoder(keras.layers.Layer):
    """
    Encodes few-shot support samples into a compact embedding.
    
    Architecture (from paper):
    - 2x Conv1D layers (kernel_size=3, padding='same')
    - Global Average Pooling
    - Linear projection to embedding dimension
    
    Input shape: (batch_size, n_channels, n_timepoints) or (batch_size, n_timepoints, n_features)
    Output shape: (batch_size, embedding_dim)
    """
    
    def __init__(self, embedding_dim=32, conv_filters=64, kernel_size=3, **kwargs):
        super(SupportEncoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        
        # Conv layers for feature extraction
        self.conv1 = Conv1D(conv_filters, kernel_size, padding='same', activation='relu',
                           kernel_initializer='he_uniform')
        self.conv2 = Conv1D(conv_filters, kernel_size, padding='same', activation='relu',
                           kernel_initializer='he_uniform')
        
        # Global pooling and projection
        self.gap = GlobalAveragePooling1D()
        self.projection = Dense(embedding_dim, kernel_initializer='glorot_uniform')
    
    def call(self, x, training=None):
        """
        Args:
            x: Support samples, shape (n_shot, seq_len, features) or (n_shot, 1, channels, timepoints)
        Returns:
            Embedding of shape (n_shot, embedding_dim)
        """
        # Handle DB-ATCNet input format: (batch, 1, channels, timepoints)
        original_shape = tf.shape(x)
        if len(x.shape) == 4:
            # Reshape from (batch, 1, C, T) to (batch, T, C)
            x = tf.squeeze(x, axis=1)  # (batch, C, T)
            x = tf.transpose(x, [0, 2, 1])  # (batch, T, C)
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Pool and project
        x = self.gap(x)
        x = self.projection(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'conv_filters': self.conv_filters,
            'kernel_size': self.kernel_size
        })
        return config


class PromptGenerator(keras.layers.Layer):
    """
    Generates subject-specific prompt tokens from a subject embedding.
    
    Architecture (from paper):
    - 2-layer MLP with ReLU activation
    - Output reshaped to (n_prompts, prompt_dim)
    
    Input shape: (embedding_dim,) - mean-pooled subject embedding
    Output shape: (n_prompts, prompt_dim)
    """
    
    def __init__(self, n_prompts=10, prompt_dim=32, hidden_dim=None, **kwargs):
        super(PromptGenerator, self).__init__(**kwargs)
        self.n_prompts = n_prompts
        self.prompt_dim = prompt_dim
        self.hidden_dim = hidden_dim or prompt_dim
        
        # 2-layer MLP
        self.dense1 = Dense(self.hidden_dim, activation='relu', 
                           kernel_initializer='glorot_uniform')
        self.dense2 = Dense(n_prompts * prompt_dim, 
                           kernel_initializer='glorot_uniform',
                           kernel_regularizer=keras.regularizers.l2(1e-4))
    
    def call(self, x, training=None):
        """
        Args:
            x: Subject embedding, shape (embedding_dim,) or (1, embedding_dim)
        Returns:
            Prompt tokens, shape (n_prompts, prompt_dim)
        """
        # Ensure 2D input
        if len(x.shape) == 1:
            x = tf.expand_dims(x, 0)
        
        # Generate prompts through MLP
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Reshape to (n_prompts, prompt_dim)
        x = tf.reshape(x, (-1, self.n_prompts, self.prompt_dim))
        
        # Return first batch (single subject)
        return x[0] if x.shape[0] == 1 else x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_prompts': self.n_prompts,
            'prompt_dim': self.prompt_dim,
            'hidden_dim': self.hidden_dim
        })
        return config


class TCPModule(keras.layers.Layer):
    """
    Task-Conditioned Prompt (TCP) Module.
    
    Combines SupportEncoder and PromptGenerator to produce subject-specific
    prompt tokens from a few-shot support set.
    """
    
    def __init__(self, n_prompts=10, prompt_dim=32, embedding_dim=32, **kwargs):
        super(TCPModule, self).__init__(**kwargs)
        self.n_prompts = n_prompts
        self.prompt_dim = prompt_dim
        self.embedding_dim = embedding_dim
        
        self.encoder = SupportEncoder(embedding_dim=embedding_dim)
        self.generator = PromptGenerator(n_prompts=n_prompts, prompt_dim=prompt_dim)
    
    def call(self, support_set, training=None):
        """
        Args:
            support_set: Few-shot support samples, shape (n_shot * n_classes, 1, C, T)
        Returns:
            Prompt tokens, shape (n_prompts, prompt_dim)
        """
        # Encode each support sample
        embeddings = self.encoder(support_set, training=training)  # (n_shot*n_classes, embed_dim)
        
        # Mean-pool across support samples to get subject embedding
        subject_embedding = tf.reduce_mean(embeddings, axis=0, keepdims=True)  # (1, embed_dim)
        
        # Generate prompts from subject embedding
        prompts = self.generator(subject_embedding, training=training)  # (n_prompts, prompt_dim)
        
        return prompts
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_prompts': self.n_prompts,
            'prompt_dim': self.prompt_dim,
            'embedding_dim': self.embedding_dim
        })
        return config


class PromptAugmentedMHA(keras.layers.Layer):
    """
    Keras Layer wrapper for prompt-augmented multi-head attention.
    
    This allows the prompt-augmented MHA to be used within a Keras model
    with proper weight management. Compatible with Keras 3.
    """
    
    def __init__(self, key_dim=8, num_heads=2, dropout=0.5, n_prompts=10, **kwargs):
        super(PromptAugmentedMHA, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.n_prompts = n_prompts
        
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = Dropout(0.3)
        self.add = Add()
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: Tuple/list of (input_feature, prompts)
                - input_feature: (batch, seq_len, features)
                - prompts: (batch, n_prompts, prompt_dim) 
        """
        input_feature, prompts = inputs
        
        # prompts comes in as (batch, n_prompts, dim) from model input
        # Concatenate prompts with input along sequence dimension
        augmented_input = Concatenate(axis=1)([prompts, input_feature])
        
        # Layer norm + attention
        x = self.layer_norm(augmented_input)
        attn_output = self.mha(x, x, training=training)
        x = self.dropout(attn_output, training=training)
        
        # Skip connection - but attention output includes prompt positions!
        # The Add() requires shapes to match.
        # augmented_input is (batch, k+T, dim)
        # attn_output is (batch, k+T, dim)
        x = self.add([augmented_input, x])
        
        # Remove prompt tokens - retain only the original sequence length T
        # Since n_prompts is known at build time, we can slice directly
        output = Lambda(lambda t: t[:, self.n_prompts:, :])(x)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'key_dim': self.key_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate,
            'n_prompts': self.n_prompts
        })
        return config


class TransformerBlock(keras.layers.Layer):
    """
    Single Transformer Block with Deep Prompt Injection.
    Structure:
    1. Prompt Injection + MHA (Residual)
    2. Feed Forward Network (Residual)
    """
    def __init__(self, key_dim=8, num_heads=2, dropout=0.5, n_prompts=10, ffn_dim=32, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.mha_block = PromptAugmentedMHA(key_dim, num_heads, dropout, n_prompts)
        
        # FFN part
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.dense1 = Dense(ffn_dim, activation='gelu') # GELU typical for transformers
        self.dropout1 = Dropout(dropout)
        self.dense2 = Dense(key_dim * num_heads) # Output dim must match input dim for residual
        self.dropout2 = Dropout(dropout)
        self.add2 = Add()
        
    def call(self, inputs, training=None):
        x, prompts = inputs
        
        # 1. Prompt-Augmented MHA
        # Returns x with same shape as input x (prompts removed)
        x_attn = self.mha_block([x, prompts], training=training)
        
        # 2. Feed Forward Network
        x_norm = self.ln2(x_attn)
        x_ffn = self.dense1(x_norm)
        x_ffn = self.dropout1(x_ffn, training=training)
        x_ffn = self.dense2(x_ffn)
        x_ffn = self.dropout2(x_ffn, training=training)
        
        # Residual connection
        output = self.add2([x_attn, x_ffn])
        
        return output
        
    def get_config(self):
        config = super().get_config()
        # Note: ffn_dim might not be exactly preserved if not stored, 
        # but key_dim*num_heads is the implicit model_dim
        return config


def get_sinusoidal_encoding(seq_len, d_model):
    """
    Generates sinusoidal positional encoding.
    Args:
        seq_len: Length of sequence
        d_model: Embedding dimension
    Returns:
        Tensor of shape (1, seq_len, d_model)
    """
    position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
    
    pe = tf.zeros((seq_len, d_model))
    
    # Use array slicing for sin/cos assignment (TensorFlow requires updates via indices)
    # We construct sin/cos separately and concat
    sin_part = tf.sin(position * div_term)
    cos_part = tf.cos(position * div_term)
    
    # Interleave sin and cos
    # shape: (seq_len, d_model/2, 2) -> flatten to (seq_len, d_model)
    pe = tf.stack([sin_part, cos_part], axis=2)
    pe = tf.reshape(pe, (seq_len, -1))
    
    # Provide shape (1, seq_len, d_model) for broadcasting
    return pe[tf.newaxis, :, :]


class TCPL_Backbone(keras.layers.Layer):
    """
    4-Layer Transformer Backbone with Deep Prompt Tuning.
    
    Replaces the TCFN + Sliding Window logic.
    Accepts sequence of tokens (windows) and prompts.
    Injects prompts at *every* transformer layer.
    """
    def __init__(self, n_layers=4, key_dim=8, num_heads=2, dropout=0.5, n_prompts=10, ffn_dim=None, **kwargs):
        super(TCPL_Backbone, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.model_dim = key_dim * num_heads # Assuming model_dim = key_dim * heads is enforced by previous layers
        
        if ffn_dim is None:
            ffn_dim = self.model_dim * 4 # Standard transformer ratio
            
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(key_dim, num_heads, dropout, n_prompts, ffn_dim)
            )
            
        self.final_norm = LayerNormalization(epsilon=1e-6)
        
        # Positional Encoding will be computed dynamically or fixed
        
    def call(self, inputs, training=None):
        x, prompts = inputs
        
        # Add Positional Encoding to x
        # x shape: (batch_size, seq_len, model_dim)
        seq_len = tf.shape(x)[1]
        pe = get_sinusoidal_encoding(seq_len, self.model_dim)
        # Cast PE to match x dtype (float32)
        pe = tf.cast(pe, dtype=x.dtype)
        
        x = x + pe
        
        # Pass through each layer, re-injecting prompts
        for layer in self.layers:
            x = layer([x, prompts], training=training)
            
        x = self.final_norm(x)
        return x
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_layers': self.n_layers
        })
        return config
