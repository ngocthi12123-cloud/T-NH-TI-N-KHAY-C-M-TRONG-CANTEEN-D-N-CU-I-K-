# Scripts

## Apps

- `apps/01_demo_checkout_app.py`: browser checkout using the fixed tray grid and CNN classifier.
- `apps/02_data_ide.py`: inspect and curate classification images with CNN predictions.

## CLI

- `cli/01_crop_tray.py`: crop a tray from a region JSON, an interactive selection, or the fixed template.
- `cli/02_demo_checkout.py`: run classifier-only checkout and write bill JSON.

## Data

- `data/01_build_classification_dataset.py`: build classification splits.
- `data/02_audit_dataset_conflicts.py`: audit duplicate and conflicting images.
- `data/03_package_classification_dataset.py`: package the dataset for cloud training.
- `data/07_collect_cookpad_candidates.py`: collect candidates with quality, duplicate, text, and CNN gates.
- `data/09_replace_reviewed_class.py`: replace a reviewed class safely.

## Training and evaluation

- `train/01_train_classifier.py`: train or fine-tune the CNN classifier.
- `debug/01_gradcam_debug.py`: inspect classifier attention with Grad-CAM.

## Cloud

- `cloud/01_sync_drive_artifacts.py`: synchronize classifier packages, checkpoints, reports, and notebook files.
