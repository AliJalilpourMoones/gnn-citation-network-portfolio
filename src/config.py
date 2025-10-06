# src/config.py

class Config:
    # --- Training Hyperparameters ---
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 5e-4
    EPOCHS = 200
    
    # --- Model Architecture ---
    HIDDEN_CHANNELS = 16
    DROPOUT_RATE = 0.5