"""Shared constants. Every notebook and the app should import from here
so the whole team uses identical settings."""

SEED = 42                      # DO NOT change: keeps the train/val/test split identical for everyone
IMG_SIZE = 224                 # CNN input size
SPLIT = (0.70, 0.15, 0.15)     # train / val / test
BATCH_SIZE = 32

# Target classes from the project objectives
CLASSES = [
    "crack",
    "pothole",
    "damaged asphalt",
    "water-filled pothole",
    "open manhole",
    "normal",
]

# Shared files on Google Drive (as seen from Colab)
DRIVE_ROOT = "/content/drive/MyDrive/road_project"
