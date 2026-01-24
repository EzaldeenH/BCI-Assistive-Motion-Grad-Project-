import numpy as np
import tensorflow as tf
from mixup_generator import MixupGenerator

def test_mixup():
    print("Testing Linear Mixup...")
    # Mock data: 10 samples, 1 channel dim, 6 channels, 100 timepoints
    # We use 6 channels to simulate small setup: Fp1(Odd), Fp2(Even), Fz(Center), C3(Odd), C4(Even), Oz(Center)
    X = np.random.rand(10, 1, 6, 100).astype(np.float32)
    y = np.eye(4)[np.random.choice(4, 10)] # 10 samples, 4 classes
    
    metrics_names = ['Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Oz']

    gen_linear = MixupGenerator(X, y, batch_size=5, alpha=0.2, mixup_type='linear')
    X_batch, y_batch = gen_linear[0]
    
    assert X_batch.shape == (5, 1, 6, 100), f"Linear X shape mismatch: {X_batch.shape}"
    assert y_batch.shape == (5, 4), f"Linear y shape mismatch: {y_batch.shape}"
    print("Linear Mixup Output Shape OK")

    print("\nTesting Binary Channel Mixup...")
    gen_binary = MixupGenerator(X, y, batch_size=5, mixup_type='channel', ch_names=metrics_names)
    X_batch, y_batch = gen_binary[0]
    
    assert X_batch.shape == (5, 1, 6, 100), f"Binary X shape mismatch: {X_batch.shape}"
    assert y_batch.shape == (5, 4), f"Binary y shape mismatch: {y_batch.shape}"
    
    indices_left = gen_binary.left_indices
    indices_right = gen_binary.right_indices
    print(f"Left Indices (Odd+Z): {indices_left} (Expected: 0, 2, 3, 5)")
    print(f"Right Indices (Even): {indices_right} (Expected: 1, 4)")
    
    assert 0 in indices_left # Fp1
    assert 1 in indices_right # Fp2
    assert 2 in indices_left # Fz (Center -> Left)
    
    print("Binary Channel Mixup Logic OK")

    print("\nAll mixup tests passed!")

if __name__ == "__main__":
    test_mixup()
