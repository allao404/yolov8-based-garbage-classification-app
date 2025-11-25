"""
YOLOv8 Training Script for Garbage Classification Dataset
Dataset: GARBAGE CLASSIFICATION 3 from Roboflow (10,464 images, 7 classes)
Trains YOLOv8 model for garbage classification
"""

import os
import sys
from pathlib import Path
import torch
import datetime
from ultralytics import YOLO


def check_gpu():
    """Check GPU availability and information"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"GPU Information:")
        print(f"{'='*60}")
        print(f"GPU Available: Yes")
        print(f"GPU Count: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name}")
            print(f"  Total Memory: {gpu_memory:.2f} GB")

        print(f"{'='*60}\n")
        return True
    else:
        print("\nWarning: No GPU available. Training will be slow on CPU.")
        return False


def train_model(
    model_size='m',
    data_config='configs/garbage.yaml',
    epochs=150,
    imgsz=640,
    batch=32,
    device=0,
    project='models',
    name='garbage_yolov8m',
    resume=False
):
    """
    Train YOLOv8 model on Garbage Classification dataset

    Args:
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        data_config: Path to data configuration YAML file
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size (adjust based on GPU memory)
        device: GPU device ID (0, 1, etc.) or 'cpu'
        project: Project directory for saving results
        name: Experiment name
        resume: Resume training from last checkpoint
    """

    # Get project root directory
    project_root = Path(__file__).parent.parent
    data_config_path = project_root / data_config
    project_path = project_root / project

    print(f"\n{'='*60}")
    print("YOLOv8 Training Configuration")
    print(f"{'='*60}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: Garbage Classification (7 classes)")
    print(f"Data Config: {data_config_path}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Device: {device}")
    print(f"Project: {project_path}")
    print(f"Experiment: {name}")
    print(f"{'='*60}\n")

    # Check if data config exists
    if not data_config_path.exists():
        print(f"Error: Data config file not found: {data_config_path}")
        print("Please run download_garbage_dataset.py first to download the dataset.")
        return None

    # Initialize model
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading YOLOv8 model: {model_name}")

    if resume:
        # Resume from last checkpoint
        checkpoint_path = project_path / name / 'weights' / 'last.pt'
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            model = YOLO(str(checkpoint_path))
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}")
            print("Starting training from pretrained weights...")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)

    print("Model loaded successfully!\n")

    # Training hyperparameters
    train_args = {
        'data': str(data_config_path),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': str(project_path),
        'name': name,
        'exist_ok': True,
        'patience': 50,  # Early stopping patience
        'cos_lr': True,
        'pretrained': True, # !
        'freeze': 0,  #
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': False,  # Don't cache images (uses less RAM)
        'workers': 8,  # Number of dataloader workers
        'optimizer': 'AdamW',  # Optimizer
        'lr0': 0.01,  # Initial learning rate (increased for better convergence)
        'lrf': 0.001,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 5.0,  # Warmup epochs (increased)
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        'box': 10.0,  # Box loss gain
        'cls': 1.5,  # Class loss gain (increased for better classification)
        'dfl': 1.5,  # DFL loss gain
        'label_smoothing': 0.1,  # Label smoothing
        'hsv_h': 0.05,  # Image HSV-Hue augmentation (increased)
        'hsv_s': 1.0,  # Image HSV-Saturation augmentation (increased)
        'hsv_v': 0.8,  # Image HSV-Value augmentation (increased)
        'degrees': 15.0,  # Image rotation (+/- deg)
        'translate': 0.2,  # Image translation (+/- fraction)
        'scale': 0.9,  # Image scale (+/- gain)
        'shear': 5.0,  # Image shear (+/- deg)
        'perspective': 0.0005,  # Image perspective (+/- fraction)
        'flipud': 0.5,  # Image flip up-down (probability)
        'fliplr': 0.5,  # Image flip left-right (probability)
        'mosaic': 1.0,  # Image mosaic (probability)
        'mixup': 0.2,  # Image mixup (probability, added)
        'copy_paste': 0.4,  # Segment copy-paste (probability, added)
        'verbose': True,  # Verbose output
        'plots': True,  # Generate training plots
    }

    print("Starting training...\n")
    start_time = datetime.datetime.now()

    # Train the model
    results = model.train(**train_args)

    end_time = datetime.datetime.now()
    training_time = end_time - start_time

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Training Time: {training_time}")
    print(f"Best model saved to: {project_path / name / 'weights' / 'best.pt'}")
    print(f"Last model saved to: {project_path / name / 'weights' / 'last.pt'}")
    print(f"Results saved to: {project_path / name}")
    print(f"{'='*60}\n")

    return model, training_time


def save_training_report(results_dir, config, training_time, val_results=None):
    """
    Save a text report of training results

    Args:
        results_dir: Directory where results are saved
        config: Training configuration dictionary
        training_time: Training duration timedelta
        val_results: Validation results (optional)
    """
    report_path = results_dir / 'training_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("YOLOv8 Garbage Classification Training Report\n")
        f.write("="*80 + "\n\n")

        # Training Configuration
        f.write("Training Configuration:\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: YOLOv8{config['model_size']}\n")
        f.write(f"Dataset: Garbage Classification (7 classes)\n")
        f.write(f"Data Config: {config['data_config']}\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"Image Size: {config['imgsz']}\n")
        f.write(f"Batch Size: {config['batch']}\n")
        f.write(f"Device: {config['device']}\n")
        f.write(f"Training Time: {training_time}\n")
        f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        # Training Results (read from results.csv if available)
        results_csv = results_dir / 'results.csv'
        if results_csv.exists():
            f.write("Training Progress:\n")
            f.write("-"*80 + "\n")
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()

                # Get final epoch metrics
                final_row = df.iloc[-1]
                f.write(f"Final Epoch: {int(final_row['epoch']) if 'epoch' in df.columns else len(df)}\n")

                # Box Loss
                if 'train/box_loss' in df.columns:
                    f.write(f"Final Train Box Loss: {final_row['train/box_loss']:.4f}\n")
                if 'val/box_loss' in df.columns:
                    f.write(f"Final Val Box Loss: {final_row['val/box_loss']:.4f}\n")

                # Class Loss
                if 'train/cls_loss' in df.columns:
                    f.write(f"Final Train Cls Loss: {final_row['train/cls_loss']:.4f}\n")
                if 'val/cls_loss' in df.columns:
                    f.write(f"Final Val Cls Loss: {final_row['val/cls_loss']:.4f}\n")

                # DFL Loss
                if 'train/dfl_loss' in df.columns:
                    f.write(f"Final Train DFL Loss: {final_row['train/dfl_loss']:.4f}\n")
                if 'val/dfl_loss' in df.columns:
                    f.write(f"Final Val DFL Loss: {final_row['val/dfl_loss']:.4f}\n")

                f.write("\n")

                # Metrics
                f.write("Final Validation Metrics:\n")
                f.write("-"*80 + "\n")
                if 'metrics/precision(B)' in df.columns:
                    f.write(f"Precision: {final_row['metrics/precision(B)']:.4f}\n")
                if 'metrics/recall(B)' in df.columns:
                    f.write(f"Recall: {final_row['metrics/recall(B)']:.4f}\n")
                if 'metrics/mAP50(B)' in df.columns:
                    f.write(f"mAP50: {final_row['metrics/mAP50(B)']:.4f}\n")
                if 'metrics/mAP50-95(B)' in df.columns:
                    f.write(f"mAP50-95: {final_row['metrics/mAP50-95(B)']:.4f}\n")

            except Exception as e:
                f.write(f"Note: Could not parse results.csv ({e})\n")
            f.write("\n")

        # Validation Results on Test Set
        if val_results is not None:
            f.write("Test Set Validation Results:\n")
            f.write("-"*80 + "\n")
            f.write(f"mAP50: {val_results.box.map50:.4f}\n")
            f.write(f"mAP50-95: {val_results.box.map:.4f}\n")
            f.write(f"Precision: {val_results.box.mp:.4f}\n")
            f.write(f"Recall: {val_results.box.mr:.4f}\n")
            f.write("\n")

            # Per-class metrics if available
            if hasattr(val_results.box, 'ap_class_index'):
                f.write("Per-Class mAP50:\n")
                f.write("-"*80 + "\n")
                class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'e-waste']
                for idx, ap in enumerate(val_results.box.ap50):
                    if idx < len(class_names):
                        f.write(f"{class_names[idx]:15s}: {ap:.4f}\n")
                f.write("\n")

        # Model Paths
        f.write("Saved Models:\n")
        f.write("-"*80 + "\n")
        f.write(f"Best Model: {results_dir / 'weights' / 'best.pt'}\n")
        f.write(f"Last Model: {results_dir / 'weights' / 'last.pt'}\n")
        f.write("\n")

        f.write("="*80 + "\n")

    print(f"Training report saved to: {report_path}")


def validate_model(model_path, data_config='configs/garbage.yaml'):
    """
    Validate trained model on test set

    Args:
        model_path: Path to trained model weights
        data_config: Path to data configuration YAML file
    """
    project_root = Path(__file__).parent.parent
    data_config_path = project_root / data_config
    model_path = project_root / model_path

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return None

    print(f"\n{'='*60}")
    print("Model Validation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data Config: {data_config_path}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(str(model_path))

    # Validate on test set
    print("Running validation on test set...\n")
    results = model.val(data=str(data_config_path), split='test')

    print(f"\n{'='*60}")
    print("Validation Results:")
    print(f"{'='*60}")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"{'='*60}\n")

    return results


def main():
    """Main training function"""
    # Check GPU
    has_gpu = check_gpu()

    # Training configuration
    config = {
        'model_size': 's',  # YOLOv8s for faster training and good accuracy
        'data_config': 'configs/garbage.yaml',
        'epochs': 200,  # Sufficient for convergence with larger dataset
        'imgsz': 640,  # Standard YOLO size, good balance
        'batch': 128,  # Increased batch size for 4 GPUs (32 per GPU)
        'device': [0, 1, 2, 3],  # Use 4 GPUs
        'project': 'models',
        'name': 'garbage_yolov8s',
        'resume': False  # Set to True to resume training
    }

    # Adjust batch size based on available GPU memory
    if has_gpu:
        # Check available memory (not total memory)
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', '0'],
                                  capture_output=True, text=True)
            free_memory = int(result.stdout.strip())
            print(f"GPU 0 free memory: {free_memory} MiB")

            # If less than 30GB free, reduce batch size further
            if free_memory < 30000:
                print(f"Warning: Limited free memory ({free_memory/1024:.1f}GB)")
                print("Using smaller batch size...")
                config['batch'] = 8
        except:
            print("Could not check GPU memory, using conservative batch size")

    # Train model
    print("\n" + "="*60)
    print("Starting YOLOv8 Training on Garbage Classification Dataset")
    print("="*60 + "\n")

    result = train_model(**config)

    if result is not None:
        model, training_time = result

        # Validate on test set
        best_model_path = f"models/{config['name']}/weights/best.pt"
        val_results = validate_model(best_model_path, data_config=config['data_config'])

        # Generate and save training report
        project_root = Path(__file__).parent.parent
        results_dir = project_root / 'models' / config['name']
        save_training_report(results_dir, config, training_time, val_results)


if __name__ == "__main__":
    main()
