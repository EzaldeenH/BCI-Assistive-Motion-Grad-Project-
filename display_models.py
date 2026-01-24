"""
Display all model summaries using TensorFlow's model.summary()
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import models


def display_all_models():
    """Display model.summary() for all available models."""
    
    model_configs = [
        ("DB_ATCNet", lambda: models.DB_ATCNet(
            n_classes=4, in_chans=64, in_samples=640,
            eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
            n_windows=5, attention='mha',
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation='elu', drop1=0.35, drop2=0.1, drop3=0.15, drop4=0.15,
            depth1=2, depth2=4
        )),
        ("DB_ATCNet_MultiScale", lambda: models.DB_ATCNet_MultiScale(
            n_classes=4, in_chans=64, in_samples=640,
            eegn_kernelSizes=(8, 16, 32, 64, 128),
            eegn_F1=16, eegn_D=2, eegn_poolSize=7, eegn_dropout=0.3,
            n_windows=5, attention='mha',
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation='elu', drop1=0.35, drop2=0.1, drop3=0.15, drop4=0.15,
            depth1=1, depth2=2
        )),
        ("DB_ATCNet_EfficientMultiScale", lambda: models.DB_ATCNet_EfficientMultiScale(
            n_classes=4, in_chans=64, in_samples=640,
            dilation_rates=(1, 2, 4), se_ratio=4,
            eegn_F1=16, eegn_D=2, eegn_kernelSize=32, eegn_poolSize=7, eegn_dropout=0.3,
            n_windows=5, attention='mha',
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation='elu', drop1=0.35, drop2=0.1, drop3=0.15, drop4=0.15,
            depth1=2, depth2=4
        )),
        ("ATCNet", lambda: models.ATCNet(
            n_classes=4, in_chans=64, in_samples=640,
            n_windows=5, attention='mha',
            eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation='elu'
        )),
        ("TCNet_Fusion", lambda: models.TCNet_Fusion(n_classes=4)),
        ("EEGTCNet", lambda: models.EEGTCNet(n_classes=4)),
        ("EEGNet", lambda: models.EEGNet_classifier(n_classes=4)),
        ("EEGNeX", lambda: models.EEGNeX_8_32(n_timesteps=640, n_features=64, n_outputs=4)),
        ("DeepConvNet", lambda: models.DeepConvNet(nb_classes=4, Chans=64, Samples=640)),
        ("ShallowConvNet", lambda: models.ShallowConvNet(nb_classes=4, Chans=64, Samples=640)),
    ]
    
    for name, build_fn in model_configs:
        print("\n" + "=" * 80)
        print(f"MODEL: {name}")
        print("=" * 80)
        try:
            model = build_fn()
            model.summary()
        except Exception as e:
            print(f"Error building model: {e}")


if __name__ == "__main__":
    display_all_models()
