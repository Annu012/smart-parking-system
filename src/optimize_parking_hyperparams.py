"""Ultra-Lightweight CPU Hyperparameter Optimization
For CPU-only training with <8GB RAM
"""
from ultralytics import YOLO
import optuna
import torch
import json
import traceback
from pathlib import Path
import gc

class ParkingHyperparamOptimizer:
    def __init__(self, n_trials=3, project_root=None):
        self.n_trials = n_trials
        self.best_params = {}
        self.best_mAP50 = 0
        
        if project_root is None:
            project_root = Path.cwd()
        else:
            project_root = Path(project_root)
        
        self.project_root = project_root
        self.data_yaml = project_root / 'config' / 'parking_dataset.yaml'
        self.runs_dir = project_root / 'runs'
        
        print(f"📁 Project: {self.project_root}")
        print(f"📄 Data YAML: {self.data_yaml}")
        
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"❌ {self.data_yaml} not found!")
        
        print("✅ Setup verified!\n")
    
    def objective(self, trial):
        # **ULTRA-LIGHT HYPERPARAMETERS FOR CPU**
        lr = trial.suggest_float('lr0', 0.001, 0.01, log=False)  # Narrower range
        momentum = trial.suggest_float('momentum', 0.8, 0.95)     # Narrower range
        weight_decay = trial.suggest_float('weight_decay', 0, 0.0005)  # Smaller max
        batch_size = trial.suggest_categorical('batch', [2, 4])  # Tiny batches only
        
        try:
            print(f"\n{'='*70}")
            print(f"🔄 Trial {trial.number}: lr={lr:.4f}, momentum={momentum:.4f}, batch={batch_size}")
            print(f"{'='*70}")
            
            # Force garbage collection to free memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # **NANO model only - 2.6M parameters**
            model = YOLO('yolo11n.pt')
            
            print(f"   Starting ultra-lightweight training...")
            results = model.train(
                data=str(self.data_yaml),
                epochs=3,  # **ONLY 3 EPOCHS** for optimization phase
                imgsz=320,  # **TINY images** - 320px
                batch=batch_size,  # **TINY batch** - 2 or 4
                device='cpu',
                lr0=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                patience=2,  # Early stop if no improvement
                verbose=True,  # Show progress so we can see epochs
                project=str(self.runs_dir),
                name=f'trial_{trial.number}',
                exist_ok=True,
                save=False,  # **DON'T SAVE** weights to reduce I/O
                val=True,
                workers=0,
                cache=False,  # **DISABLE image caching** - saves memory
                amp=False,  # Disable mixed precision on CPU
                mosaic=0.0,  # **DISABLE mosaic augmentation** - memory hog
                mixup=0.0,  # **DISABLE mixup** - memory hog
                augment=False,  # **DISABLE all augmentation**
                rect=False,
                fraction=1.0,
                hsv_h=0,
                hsv_s=0,
                hsv_v=0,
                flipud=0.0,
                fliplr=0.0,
                degrees=0,
                translate=0,
                shear=0,
                perspective=0,
                erasing=0,
                copy_paste=0,
                cutmix=0,
            )
            
            print(f"\n   ✅ Trial {trial.number} completed")
            
            if not hasattr(results, 'results_dict'):
                print(f"   ⚠️  No results found")
                return 0.0
            
            mAP50 = float(results.results_dict.get('metrics/mAP50', 0))
            print(f"   📊 mAP50: {mAP50:.4f}")
            
            if mAP50 > self.best_mAP50:
                self.best_mAP50 = mAP50
                self.best_params = {
                    'lr0': float(lr),
                    'momentum': float(momentum),
                    'weight_decay': float(weight_decay),
                    'batch': int(batch_size)
                }
                print(f"   🏆 NEW BEST! mAP50: {self.best_mAP50:.4f}")
            
            # Cleanup
            del model, results
            gc.collect()
            
            return mAP50
        
        except Exception as e:
            print(f"\n   ❌ Trial {trial.number} FAILED!")
            print(f"   {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            gc.collect()
            return 0.0
    
    def optimize(self):
        print(f"\n{'='*70}")
        print(f"🚀 ULTRA-LIGHTWEIGHT HYPERPARAMETER TUNING (CPU-ONLY)")
        print(f"   Model: YOLO11n (Nano - 2.6M params)")
        print(f"   Image Size: 320px")
        print(f"   Batch Size: 2-4")
        print(f"   Epochs/Trial: 3")
        print(f"   Augmentation: DISABLED")
        print(f"{'='*70}\n")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, gc_after_trial=True)
        
        print(f"\n{'='*70}")
        print(f"✅ OPTIMIZATION COMPLETE!")
        print(f"   Best mAP50: {self.best_mAP50:.4f}")
        print(f"   Best params: {json.dumps(self.best_params, indent=2)}")
        print(f"{'='*70}\n")
        
        return self.best_params


if __name__ == "__main__":
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}\n")
    
    if not (current_dir / 'config' / 'parking_dataset.yaml').exists():
        print(f"❌ ERROR: parking_dataset.yaml not found")
        exit(1)
    
    # Start with just 2 trials to test
    optimizer = ParkingHyperparamOptimizer(n_trials=2, project_root=current_dir)
    best_params = optimizer.optimize()
    
    # Train final model with full settings
    print(f"\n{'='*70}")
    print(f"🎯 FINAL TRAINING (FULL SETTINGS)")
    print(f"{'='*70}\n")
    
    try:
        # Use best params if found, otherwise use defaults
        if not best_params:
            print("⚠️  No best params found from optimization, using defaults...")
            best_params = {'lr0': 0.01, 'momentum': 0.937, 'weight_decay': 0.0, 'batch': 4}
        
        model = YOLO('yolo11n.pt')
        data_yaml = current_dir / 'config' / 'parking_dataset.yaml'
        runs_dir = current_dir / 'runs'
        
        print(f"Starting final training with params:")
        print(f"  Batch: {best_params['batch']}")
        print(f"  LR: {best_params['lr0']:.4f}")
        print(f"  Momentum: {best_params['momentum']:.4f}\n")
        
        results = model.train(
            data=str(data_yaml),
            epochs=50,  # More epochs for final training
            imgsz=416,  # Slightly larger for final model
            batch=best_params['batch'],
            device='cpu',
            lr0=best_params['lr0'],
            momentum=best_params['momentum'],
            weight_decay=best_params['weight_decay'],
            patience=10,
            save=True,
            verbose=True,
            project=str(runs_dir),
            name='final_model_optimized',
            exist_ok=True,
            workers=0,
            cache=False,
            amp=False,
            mosaic=0.5,  # Re-enable some augmentation for final training
            mixup=0.1,
            augment=True,
        )
        
        # Save best params
        results_dir = current_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'best_hyperparams.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n✅ Final training complete!")
        print(f"💾 Best hyperparams saved to: {results_dir / 'best_hyperparams.json'}")
        
    except Exception as e:
        print(f"\n❌ Final training failed!")
        print(f"{type(e).__name__}: {str(e)}")
        traceback.print_exc()