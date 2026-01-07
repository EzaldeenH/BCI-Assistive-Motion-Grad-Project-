# %%
"""
TCPL Training Script: Task-Conditioned Prompt Learning for Few-Shot Cross-Subject EEG Decoding

This script implements meta-learning training for DB_ATCNet_TCPL.
Based on: Wang et al. 2025, "TCPL: Task-Conditioned Prompt Learning for 
Few-Shot Cross-Subject Motor Imagery EEG Decoding"

Usage:
    python Physionet_main_TCPL.py --mode train --n_shot 5 --n_folds 10
    python Physionet_main_TCPL.py --mode test --n_shot 10 --checkpoint results_tcpl/fold_1/best_model.h5
"""

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/ezzo/anaconda3/lib/python3.13/site-packages/nvidia/cuda_nvcc'
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay

import models
from tcpl_modules import TCPModule
from Physionet_DataLoad_TCPL import (
    load_physionet_by_subject, 
    EpisodicDataGenerator,
    split_subjects_kfold
)


def get_data_path():
    """Resolve the Physionet dataset directory."""
    env_path = os.environ.get('PHYSIONET_DATA_DIR')
    if env_path and os.path.isdir(env_path):
        print(f"Using PHYSIONET_DATA_DIR={env_path}")
        return env_path

    repo_files = os.path.join(os.getcwd(), 'files')
    if os.path.isdir(repo_files):
        if os.path.exists(os.path.join(repo_files, 'S001')) or os.path.exists(os.path.join(repo_files, 'RECORDS')):
            print(f"Using local dataset folder: {repo_files}")
            return repo_files

    fallback = "/root/autodl-tmp/physionet"
    if os.path.isdir(fallback):
        return fallback

    raise FileNotFoundError("Physionet dataset not found.")


class TCPLTrainer:
    """
    Meta-learning trainer for TCPL (Task-Conditioned Prompt Learning).
    
    Implements Algorithm 1 from the paper:
    1. Sample subject (task) from training subjects
    2. Split into support/query sets
    3. Generate prompts from support set
    4. Forward pass on query with prompts
    5. Loss = CrossEntropy + λ * ||prompts||²
    """
    
    def __init__(self, model, tcp_module, n_classes=4, lr=0.001, 
                 prompt_reg_lambda=1e-4):
        """
        Parameters
        ----------
        model : keras.Model
            DB_ATCNet_TCPL model with two inputs (EEG, prompts)
        tcp_module : TCPModule
            Module to generate prompts from support set
        n_classes : int
            Number of classes
        lr : float
            Learning rate
        prompt_reg_lambda : float
            Regularization weight for prompt L2 norm
        """
        self.model = model
        self.tcp_module = tcp_module
        self.n_classes = n_classes
        self.prompt_reg_lambda = prompt_reg_lambda
        self.initial_lr = lr
        
        # Cosine decay learning rate schedule
        # Decays from lr to 0.1*lr over total_steps
        total_steps = 2000  # Will be updated if n_episodes is different
        self.lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=total_steps,
            alpha=0.1  # Final LR = 0.1 * initial_lr
        )
        
        # Optimizer with learning rate schedule
        self.optimizer = Adam(learning_rate=self.lr_schedule)
        
        # Build TCP module by calling it once
        dummy_support = tf.zeros((4, 1, 64, 640))  # 4 samples for build
        _ = self.tcp_module(dummy_support)
        
    @tf.function
    def train_step(self, support_X, support_y, query_X, query_y):
        """
        One meta-training step (Algorithm 1).
        
        Parameters
        ----------
        support_X : tf.Tensor
            Support set EEG data (n_shot * n_classes, 1, C, T)
        support_y : tf.Tensor
            Support set labels (n_shot * n_classes, n_classes)
        query_X : tf.Tensor
            Query set EEG data
        query_y : tf.Tensor
            Query set labels
            
        Returns
        -------
        loss : tf.Tensor
            Total loss (CE + prompt regularization)
        accuracy : tf.Tensor
            Query set accuracy
        """
        with tf.GradientTape() as tape:
            # Generate subject-specific prompts from support set
            prompts = self.tcp_module(support_X, training=True)
            
            # Forward pass on query set with prompts
            # prompts shape: (n_prompts, prompt_dim)
            # We need to tile prompts for batch processing
            batch_size = tf.shape(query_X)[0]
            prompts_batched = tf.tile(tf.expand_dims(prompts, 0), [batch_size, 1, 1])
            
            predictions = self.model([query_X, prompts_batched], training=True)
            
            # Cross-entropy loss
            ce_loss = tf.reduce_mean(categorical_crossentropy(query_y, predictions))
            
            # Prompt regularization (L2 norm)
            prompt_reg = self.prompt_reg_lambda * tf.reduce_mean(tf.square(prompts))
            
            # Total loss
            total_loss = ce_loss + prompt_reg
        
        # Compute gradients for both model and TCP module
        all_trainable = self.model.trainable_variables + self.tcp_module.trainable_variables
        gradients = tape.gradient(total_loss, all_trainable)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, all_trainable))
        
        # Compute accuracy
        pred_labels = tf.argmax(predictions, axis=1)
        true_labels = tf.argmax(query_y, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_labels, true_labels), tf.float32))
        
        return total_loss, ce_loss, accuracy
    
    def evaluate(self, support_X, query_X, query_y):
        """
        Evaluate on query set given support set.
        
        This simulates few-shot inference: generate prompts from support,
        then classify query samples.
        """
        # Generate prompts from support
        prompts = self.tcp_module(support_X, training=False)
        
        # Batch prompts for query - convert to tensor to avoid mixed types
        batch_size = query_X.shape[0]
        prompts_np = prompts.numpy() if hasattr(prompts, 'numpy') else prompts
        prompts_batched = np.tile(np.expand_dims(prompts_np, 0), [batch_size, 1, 1])
        prompts_batched = tf.cast(prompts_batched, tf.float32)
        
        # Forward pass
        predictions = self.model([query_X, prompts_batched], training=False)
        
        # Compute metrics
        pred_labels = np.argmax(predictions.numpy(), axis=1)
        true_labels = np.argmax(query_y, axis=1)
        
        acc = accuracy_score(true_labels, pred_labels)
        kappa = cohen_kappa_score(true_labels, pred_labels)
        
        return acc, kappa, predictions.numpy()


def train_tcpl(config):
    """
    Main TCPL meta-training function with k-fold cross-validation.
    """
    print("=" * 60)
    print("TCPL Meta-Learning Training")
    print("=" * 60)
    
    # Create results directory
    results_path = config['results_path']
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Load data by subject
    print("\n[1] Loading data by subject...")
    subject_data, n_channels, n_classes = load_physionet_by_subject(
        config['data_path'], 
        n_timepoints=config['n_timepoints']
    )
    
    # Split subjects into folds
    subject_ids = list(subject_data.keys())
    folds = split_subjects_kfold(subject_ids, n_folds=config['n_folds'], seed=42)
    
    print(f"\n[2] {config['n_folds']}-fold cross-validation across {len(subject_ids)} subjects")
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1}: {len(fold)} subjects ({fold[:3]}...)")
    
    # Track metrics
    all_fold_results = []
    
    # Log file
    log_file = open(os.path.join(results_path, 'training_log.txt'), 'w')
    
    for fold_idx in range(config['n_folds']):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{config['n_folds']}")
        print(f"{'='*60}")
        
        fold_start = time.time()
        
        # Split subjects into train and test
        test_subjects = folds[fold_idx]
        train_subjects = [s for s in subject_ids if s not in test_subjects]
        
        # Separate validation subjects from training subjects to prevent leakage
        val_subjects = train_subjects[:10]  # Use 10 subjects for validation
        train_subjects_final = train_subjects[10:]  # Remaining for training
        
        print(f"Train subjects: {len(train_subjects_final)}, Val subjects: {len(val_subjects)}, Test subjects: {len(test_subjects)}")
        
        # Create fold results directory
        fold_path = os.path.join(results_path, f'fold_{fold_idx + 1}')
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        
        # Create episodic data generator for training (PURE training set)
        train_subject_data = {s: subject_data[s] for s in train_subjects_final}
        train_generator = EpisodicDataGenerator(
            train_subject_data, 
            n_shot=config['n_shot'],
            seed=42 + fold_idx
        )
        
        # Create fresh model and TCP module for each fold
        model = models.DB_ATCNet_TCPL(
            n_classes=n_classes,
            in_chans=n_channels,
            in_samples=config['n_timepoints'],
            n_windows=config['n_windows'],
            n_prompts=config['n_prompts'],
            prompt_dim=config['prompt_dim'],
            eegn_F1=config['eegn_F1'],
            eegn_D=config['eegn_D'],
            eegn_kernelSize=config['eegn_kernelSize'],
            eegn_poolSize=config['eegn_poolSize'],
            tcn_depth=config['tcn_depth'],
            tcn_filters=config['tcn_filters'],
        )
        
        tcp_module = TCPModule(
            n_prompts=config['n_prompts'],
            prompt_dim=config['prompt_dim'],
            embedding_dim=config['prompt_dim']
        )
        
        # Create trainer
        trainer = TCPLTrainer(
            model=model,
            tcp_module=tcp_module,
            n_classes=n_classes,
            lr=config['lr'],
            prompt_reg_lambda=config['prompt_reg_lambda']
        )
        
        # Create validation generator (using the separate validation subjects)
        # Note: val_subjects and train_subjects_final were separated at the start of the loop
        val_generator = EpisodicDataGenerator(
            {s: subject_data[s] for s in val_subjects},
            n_shot=config['n_shot'],
            seed=99
        )
        
        # Training loop with early stopping
        print(f"\\n[Training] up to {config['n_episodes']} episodes (early stopping enabled)...")
        best_val_acc = 0
        patience_counter = 0
        patience = config.get('patience', 300)  # Stop if no improvement for 300 episodes
        val_interval = config.get('val_interval', 100)  # Validate every 100 episodes
        
        train_losses = []
        train_accs = []
        val_accs = []
        best_weights = None
        
        for episode in range(config['n_episodes']):
            # Sample episode from training subjects
            support_X, support_y, query_X, query_y = train_generator.sample_episode()
            
            # Convert to tensors
            support_X = tf.cast(support_X, tf.float32)
            support_y = tf.cast(support_y, tf.float32)
            query_X = tf.cast(query_X, tf.float32)
            query_y = tf.cast(query_y, tf.float32)
            
            # Train step
            loss, ce_loss, acc = trainer.train_step(support_X, support_y, query_X, query_y)
            
            train_losses.append(float(loss))
            train_accs.append(float(acc))
            
            # Log progress
            if (episode + 1) % config['log_interval'] == 0:
                avg_loss = np.mean(train_losses[-config['log_interval']:])
                avg_acc = np.mean(train_accs[-config['log_interval']:])
                print(f"  Episode {episode+1}/{config['n_episodes']}: "
                      f"loss={avg_loss:.4f}, acc={avg_acc:.4f}")
            
            # Validation check
            if (episode + 1) % val_interval == 0:
                # Evaluate on validation subjects
                val_acc_list = []
                for val_subj in val_subjects:
                    s_X, s_y, q_X, q_y = val_generator.sample_episode(val_subj)
                    s_X = tf.cast(s_X, tf.float32)
                    q_X = tf.cast(q_X, tf.float32)
                    v_acc, _, _ = trainer.evaluate(s_X, q_X, q_y)
                    val_acc_list.append(v_acc)
                
                current_val_acc = np.mean(val_acc_list)
                val_accs.append(current_val_acc)
                
                print(f"  [Val] Episode {episode+1}: val_acc={current_val_acc:.4f} (best={best_val_acc:.4f})")
                
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    patience_counter = 0
                    # Save best weights
                    best_weights = [w.numpy() for w in model.trainable_variables]
                else:
                    patience_counter += val_interval
                    
                if patience_counter >= patience:
                    print(f"  Early stopping at episode {episode+1} (no improvement for {patience} episodes)")
                    break
        
        # Restore best weights if we have them
        if best_weights is not None:
            for w, best_w in zip(model.trainable_variables, best_weights):
                w.assign(best_w)
            print(f"  Restored best weights (val_acc={best_val_acc:.4f})")
        
        # Save model weights
        model_path = os.path.join(fold_path, 'model.weights.h5')
        model.save_weights(model_path)
        
        # Save TCP module weights (wrap in model for saving)
        tcp_path = os.path.join(fold_path, 'tcp_weights.npz')
        tcp_weights = {f'w{i}': w.numpy() for i, w in enumerate(tcp_module.trainable_variables)}
        np.savez(tcp_path, **tcp_weights)
        
        # Evaluate on test subjects
        print(f"\n[Evaluation] Testing on {len(test_subjects)} subjects...")
        fold_accs = []
        fold_kappas = []
        
        test_generator = EpisodicDataGenerator(
            {s: subject_data[s] for s in test_subjects},
            n_shot=config['n_shot'],
            seed=42
        )
        
        for test_subj in test_subjects:
            support_X, support_y, query_X, query_y = test_generator.sample_episode(test_subj)
            
            support_X = tf.cast(support_X, tf.float32)
            query_X = tf.cast(query_X, tf.float32)
            
            acc, kappa, _ = trainer.evaluate(support_X, query_X, query_y)
            fold_accs.append(acc)
            fold_kappas.append(kappa)
            print(f"  Subject {test_subj}: acc={acc:.4f}, kappa={kappa:.4f}")
        
        # Fold summary
        fold_time = (time.time() - fold_start) / 60
        fold_acc_mean = np.mean(fold_accs)
        fold_acc_std = np.std(fold_accs)
        fold_kappa_mean = np.mean(fold_kappas)
        
        print(f"\nFold {fold_idx+1} Results: acc={fold_acc_mean:.4f}±{fold_acc_std:.4f}, "
              f"kappa={fold_kappa_mean:.4f}, time={fold_time:.1f}min")
        
        all_fold_results.append({
            'fold': fold_idx + 1,
            'acc_mean': fold_acc_mean,
            'acc_std': fold_acc_std,
            'kappa_mean': fold_kappa_mean,
            'per_subject_acc': fold_accs
        })
        
        # Log to file
        log_file.write(f"Fold {fold_idx+1}: acc={fold_acc_mean:.5f}±{fold_acc_std:.5f}, "
                      f"kappa={fold_kappa_mean:.5f}\n")
        
        # Plot training curve
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title(f'Fold {fold_idx+1} - Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs)
        plt.title(f'Fold {fold_idx+1} - Training Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fold_path, 'training_curves.png'))
        plt.close()
        
        # Clear session
        keras.backend.clear_session()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    all_accs = [r['acc_mean'] for r in all_fold_results]
    all_kappas = [r['kappa_mean'] for r in all_fold_results]
    
    final_summary = f"""
    Model: DB_ATCNet_TCPL
    N-shot: {config['n_shot']}
    Episodes per fold: {config['n_episodes']}
    
    Per-fold Accuracies: {[f"{a:.4f}" for a in all_accs]}
    Per-fold Kappas: {[f"{k:.4f}" for k in all_kappas]}
    
    Average Accuracy: {np.mean(all_accs):.5f} ± {np.std(all_accs):.5f}
    Average Kappa: {np.mean(all_kappas):.5f} ± {np.std(all_kappas):.5f}
    """
    
    print(final_summary)
    log_file.write(f"\n{'='*60}\nFINAL RESULTS\n{final_summary}")
    log_file.close()
    
    # Save results
    np.savez(
        os.path.join(results_path, 'tcpl_results.npz'),
        all_fold_results=all_fold_results,
        config=config
    )
    
    # Summary plot
    plt.figure(figsize=(10, 5))
    x = np.arange(len(all_accs))
    plt.bar(x - 0.15, all_accs, 0.3, label='Accuracy', color='steelblue')
    plt.bar(x + 0.15, all_kappas, 0.3, label='Kappa', color='coral')
    plt.axhline(np.mean(all_accs), color='steelblue', linestyle='--', 
                label=f'Avg Acc: {np.mean(all_accs):.3f}')
    plt.axhline(np.mean(all_kappas), color='coral', linestyle='--',
                label=f'Avg Kappa: {np.mean(all_kappas):.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title(f'TCPL {config["n_shot"]}-shot Cross-Validation Results')
    plt.xticks(x, [f'Fold {i+1}' for i in range(len(all_accs))])
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'tcpl_summary.png'), dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='TCPL Training for DB-ATCNet')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--n_shot', type=int, default=10, help='Number of shots per class')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of cross-validation folds')
    parser.add_argument('--n_episodes', type=int, default=20000, help='Max training episodes per fold')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=50, help='Log every N episodes')
    parser.add_argument('--patience', type=int, default=5000, help='Early stopping patience (episodxfes)')
    parser.add_argument('--val_interval', type=int, default=100, help='Validation check interval')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data_path': get_data_path(),
        'results_path': os.path.join(os.getcwd(), 'results_tcpl'),
        'n_timepoints': 640,
        'n_classes': 4,
        
        # Meta-learning params
        'n_shot': args.n_shot,
        'n_folds': args.n_folds,
        'n_episodes': args.n_episodes,
        'lr': args.lr,
        'prompt_reg_lambda': 1e-4,
        'log_interval': args.log_interval,
        'patience': args.patience,
        'val_interval': args.val_interval,
        
        # Model params (same as DB_ATCNet)
        'n_windows': 5,
        'n_prompts': 10,
        'prompt_dim': 32,  # Should match F2 = F1 * D = 16 * 2 = 32
        'eegn_F1': 16,
        'eegn_D': 2,
        'eegn_kernelSize': 64,
        'eegn_poolSize': 7,
        'tcn_depth': 2,
        'tcn_filters': 32,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    if args.mode == 'train':
        train_tcpl(config)
    else:
        print("Test mode not yet implemented. Use train mode first.")


if __name__ == "__main__":
    main()