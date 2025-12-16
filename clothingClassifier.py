import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

print("\n" + "="*80)
print("🧥 CAROUSELL CLOTHING CLASSIFIER - COMPLETE SYSTEM")
print("="*80)

IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 25
INITIAL_LR = 0.001
FINE_TUNE_LR = 0.0001

TRAIN_DIR = 'Clothes_Dataset_Train'
VAL_DIR = 'Clothes_Dataset_Val'
MODEL_PATH = 'models/carousell_clothing_model.keras'
BEST_MODEL_PATH = 'models/carousell_clothing_model_best.keras'
CLASS_NAMES_PATH = 'models/class_names.json'
HISTORY_PATH = 'outputs/training_history.json'
PLOT_PATH = 'outputs/training_history.png'

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

CLASS_NAMES_MAPPING = {
    'Blazer': 'Blazer 🧥',
    'Celana_Panjang': 'Long Pants 👖',
    'Celana_Pendek': 'Shorts 🩳',
    'Gaun': 'Dress 👗',
    'Hoodie': 'Hoodie 🧥',
    'Jaket': 'Jacket 🧥',
    'Jaket_Denim': 'Denim Jacket 🧥',
    'Jaket_Olahraga': 'Sports Jacket 🏃',
    'Jeans': 'Jeans 👖',
    'Kaos': 'T-Shirt 👕',
    'Kemeja': 'Shirt 👔',
    'Mantel': 'Coat 🧥',
    'Polo': 'Polo Shirt 👕',
    'Rok': 'Skirt 👗',
    'Sweter': 'Sweater 🧶'
}

def load_trained_model():
    model_to_load = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    
    if not os.path.exists(model_to_load):
        return None, None
    
    if not os.path.exists(CLASS_NAMES_PATH):
        return None, None
    
    try:
        print(f"\n🔄 Loading model from: {model_to_load}")
        model = keras.models.load_model(model_to_load)
        
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {len(class_names)}")
        print(f"   Model: {os.path.basename(model_to_load)}")
        
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def predict_image(model, class_names, image_path, show_plot=True):
    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Image not found: {image_path}")
        return None
    
    try:
        print(f"\n🔍 Analyzing image: {os.path.basename(image_path)}")
        
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=(IMG_SIZE, IMG_SIZE)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = predictions[predicted_idx]
        
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        
        print("\n" + "="*80)
        print(f"📸 PREDICTION RESULTS")
        print("="*80)
        
        display_name = CLASS_NAMES_MAPPING.get(predicted_class, predicted_class)
        print(f"\n🎯 THIS IS: {display_name}")
        print(f"📊 Confidence: {confidence:.2%}")
        
        print(f"\n📋 Top 5 Predictions:")
        print("-" * 80)
        for i, idx in enumerate(top_5_indices, 1):
            cls = class_names[idx]
            display = CLASS_NAMES_MAPPING.get(cls, cls)
            conf = predictions[idx]
            bar = "█" * int(conf * 40)
            print(f"  {i}. {display:<30} {conf:>6.2%}  {bar}")
        print("="*80 + "\n")
        
        if show_plot:
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            
            title_text = f"{display_name}\nConfidence: {confidence:.2%}"
            plt.title(title_text, fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            output_path = os.path.join('outputs', f'prediction_{os.path.basename(image_path)}')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Prediction image saved: {output_path}")
            
            try:
                plt.show()
            except:
                pass
            
            plt.close()
        
        return {
            'image': os.path.basename(image_path),
            'prediction': predicted_class,
            'display_name': display_name,
            'confidence': float(confidence),
            'top_5': [(class_names[idx], float(predictions[idx])) for idx in top_5_indices]
        }
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_prediction_mode(model, class_names):
    print("\n" + "="*80)
    print("🎭 INTERACTIVE PREDICTION MODE")
    print("="*80)
    print("\nAvailable classes:")
    for i, cls in enumerate(class_names, 1):
        display = CLASS_NAMES_MAPPING.get(cls, cls)
        print(f"  {i:2d}. {display}")
    
    print("\n" + "="*80)
    print("📝 INSTRUCTIONS:")
    print("  • Enter the path to an image file")
    print("  • Type 'quit' or 'exit' to stop")
    print("  • Type 'batch' to predict multiple images")
    print("="*80 + "\n")
    
    while True:
        user_input = input("📁 Enter image path (or command): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting prediction mode. Goodbye!")
            break
        
        elif user_input.lower() == 'batch':
            print("\n📦 Batch prediction mode")
            print("Enter image paths (one per line). Type 'done' when finished.\n")
            
            image_paths = []
            while True:
                path = input(f"  Image {len(image_paths)+1}: ").strip()
                if path.lower() == 'done':
                    break
                if path:
                    image_paths.append(path)
            
            if image_paths:
                print(f"\n🔄 Processing {len(image_paths)} images...\n")
                results = []
                for i, img_path in enumerate(image_paths, 1):
                    print(f"\n[{i}/{len(image_paths)}]")
                    result = predict_image(model, class_names, img_path, show_plot=False)
                    if result:
                        results.append(result)
                
                if results:
                    print("\n" + "="*80)
                    print("📊 BATCH PREDICTION SUMMARY")
                    print("="*80)
                    for idx, res in enumerate(results, 1):
                        print(f"{idx}. {res['image']:<40} → {res['display_name']} ({res['confidence']:.2%})")
                    print("="*80 + "\n")
        
        elif user_input:
            predict_image(model, class_names, user_input, show_plot=True)
        
        else:
            print("⚠️  Please enter a valid image path or command")

def check_dataset_exists():
    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ ERROR: Training directory not found: {TRAIN_DIR}")
        return False
    
    if not os.path.exists(VAL_DIR):
        print(f"\n❌ ERROR: Validation directory not found: {VAL_DIR}")
        return False
    
    return True

def train_model():
    print("\n" + "="*80)
    print("🏋️ STARTING TRAINING PROCESS")
    print("="*80)
    
    if not check_dataset_exists():
        print("\nPlease ensure your dataset is organized as:")
        print(f"  {TRAIN_DIR}/")
        print(f"    ├── Blazer/")
        print(f"    ├── Celana_Panjang/")
        print(f"    └── ... (other classes)")
        print(f"\n  {VAL_DIR}/")
        print(f"    ├── Blazer/")
        print(f"    ├── Celana_Panjang/")
        print(f"    └── ... (other classes)")
        return None, None
    
    print("\n[CREATE] Data Generators...")
    
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✓ Training samples:   {train_generator.samples:,}")
    print(f"✓ Validation samples: {validation_generator.samples:,}")
    
    class_names = sorted(list(train_generator.class_indices.keys()))
    num_classes = len(class_names)
    
    print(f"✓ Classes: {num_classes}")
    
    print("\n[BUILD] Model...")
    
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    print("✓ Model built")
    
    print("\n" + "="*80)
    print("🚀 PHASE 1: Initial Training (Frozen Base)")
    print("="*80)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    callbacks_phase1 = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history_phase1 = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    print("\n" + "="*80)
    print("🚀 PHASE 2: Fine-Tuning")
    print("="*80)
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    callbacks_phase2 = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history_phase2 = model.fit(
        train_generator,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    combined_history = {
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
        'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    }
    
    print("\n[SAVE] Saving results...")
    
    model.save(MODEL_PATH)
    print(f"✓ Model saved: {MODEL_PATH}")
    
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✓ Class names saved: {CLASS_NAMES_PATH}")
    
    history_serializable = {k: [float(v) for v in vals] for k, vals in combined_history.items()}
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"✓ History saved: {HISTORY_PATH}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(combined_history['accuracy']) + 1)
    
    axes[0].plot(epochs_range, combined_history['accuracy'], 'b-', label='Training')
    axes[0].plot(epochs_range, combined_history['val_accuracy'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_range, combined_history['loss'], 'b-', label='Training')
    axes[1].plot(epochs_range, combined_history['val_loss'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"✓ Plot saved: {PLOT_PATH}")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()
    
    final_val_acc = combined_history['val_accuracy'][-1]
    print(f"\n" + "="*80)
    print(f"✅ TRAINING COMPLETE!")
    print(f"="*80)
    print(f"Final Validation Accuracy: {final_val_acc:.2%}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    
    return model, class_names

def main():
    print("\n🔍 Checking for existing trained model...")
    
    model, class_names = load_trained_model()
    
    if model is not None and class_names is not None:
        print("\n✅ Found trained model! Skipping training.")
        print("\n" + "="*80)
        print("MENU OPTIONS:")
        print("="*80)
        print("  1. Start prediction mode (classify images)")
        print("  2. Retrain model from scratch")
        print("  3. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            interactive_prediction_mode(model, class_names)
        elif choice == '2':
            print("\n⚠️  This will retrain the model from scratch.")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == 'yes':
                model, class_names = train_model()
                if model is not None:
                    print("\n✅ Training completed! Starting prediction mode...")
                    interactive_prediction_mode(model, class_names)
        else:
            print("\n👋 Exiting. Goodbye!")
    
    else:
        print("\n⚠️  No trained model found. Training required.")
        print("\n" + "="*80)
        print("OPTIONS:")
        print("="*80)
        print("  1. Train model now")
        print("  2. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            model, class_names = train_model()
            if model is not None:
                print("\n✅ Training completed! Starting prediction mode...")
                interactive_prediction_mode(model, class_names)
        else:
            print("\n👋 Exiting. Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("👋 Exiting gracefully...")
    except Exception as e:
        print(f"\n\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()