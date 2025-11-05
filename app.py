import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import json
import math
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import io

st.set_page_config(
    page_title="Art Style Classifier",
    page_icon="🎨",
    layout="wide"
)

if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = []
if 'is_training' not in st.session_state:
    st.session_state.is_training = False

IMG_SIZE = 128
DATASET_DIR = 'dataset'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset(dataset_path, img_size=IMG_SIZE):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))])
    
    if not class_names:
        raise ValueError(f"No class directories found in {dataset_path}. Please add image folders for each art style.")
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        class_images = [f for f in os.listdir(class_path) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not class_images:
            st.warning(f"No images found in {class_name} folder")
            continue
            
        for img_file in class_images:
            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size))
                img_array = np.array(img, dtype=np.float32)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                st.warning(f"Error loading {img_path}: {e}")
    
    if len(images) == 0:
        raise ValueError("No images could be loaded from the dataset. Please check your image files.")
    
    return np.array(images, dtype=np.float32), np.array(labels), class_names

def create_model(num_classes, img_size=IMG_SIZE):
    """Transfer Learning model - art style classification with pre-trained EfficientNetB0"""
    
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3),
        pooling='avg'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    inputs = keras.Input(shape=(img_size, img_size, 3))
    
    x = keras.applications.efficientnet.preprocess_input(inputs)
    
    x = base_model(x, training=True)
    
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    loss = keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def predict_image(model, image, class_names, use_tta=True):
    """optional Test-Time Augmentation"""
    if isinstance(image, Image.Image):
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32)
    else:
        img_array = image
    
    if use_tta:
        predictions_list = []
        
        # original image
        img_batch = np.expand_dims(img_array, axis=0)
        predictions_list.append(model.predict(img_batch, verbose=0)[0])
        
        # horizontal flip
        img_flipped = np.flip(img_array, axis=1)
        img_batch = np.expand_dims(img_flipped, axis=0)
        predictions_list.append(model.predict(img_batch, verbose=0)[0])
        
        # brightness variations
        img_bright = np.clip(img_array * 1.1, 0, 255)
        img_batch = np.expand_dims(img_bright, axis=0)
        predictions_list.append(model.predict(img_batch, verbose=0)[0])
        
        img_dark = np.clip(img_array * 0.9, 0, 255)
        img_batch = np.expand_dims(img_dark, axis=0)
        predictions_list.append(model.predict(img_batch, verbose=0)[0])
        
        # Alaverage all predictions
        predictions = np.mean(predictions_list, axis=0)
    else:
        img_batch = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_batch, verbose=0)[0]
    
    return predictions

# sidebar
st.sidebar.title("🎨 Art Style Classifier")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📁 Dataset", "🎓 Train Model", "🔍 Classify", "📊 Metrics"]
)

# HOME PAGE
if page == "🏠 Home":
    st.title("🎨 Art Style Classifier")
    st.markdown("### Identify Art Styles with Deep Learning")
    
    st.markdown("""
    This application uses **Transfer Learning with EfficientNetB0** to achieve high-accuracy art style classification.
    The model is pre-trained on ImageNet and fine-tuned on artistic styles for superior performance.
    
    **Supported Art Styles:**
    - 🌅 **Impressionist**: Soft, blended colors and light effects
    - 📐 **Cubist**: Geometric shapes and abstract forms
    - 💭 **Surrealist**: Dreamlike, unusual combinations
    - 🏛️ **Renaissance**: Warm tones and classical composition
    - 🎭 **Abstract**: Bold shapes and vibrant colors
    
    **Features:**
    - ✅ Transfer learning for maximum accuracy
    - ✅ Class balancing to prevent prediction bias
    - ✅ Adaptive learning rate for optimal training
    - ✅ Real-time training progress with fast visualization
    - ✅ Upload and classify your own artwork
    - ✅ Save and load trained models
    - ✅ Batch image classification
    
    **Get Started:**
    1. 📁 View the dataset (pre-loaded with sample images)
    2. 🎓 Train the model on the dataset
    3. 🔍 Classify new artwork
    4. 📊 Analyze model performance
    """)
    
    # dataset info
    if os.path.exists(DATASET_DIR):
        st.markdown("### 📊 Current Dataset")
        cols = st.columns(5)
        class_dirs = sorted([d for d in os.listdir(DATASET_DIR) 
                           if os.path.isdir(os.path.join(DATASET_DIR, d))])
        
        for idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(DATASET_DIR, class_name)
            img_count = len([f for f in os.listdir(class_path) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            with cols[idx % 5]:
                st.metric(class_name, f"{img_count} images")

# DATASET PAGE
elif page == "📁 Dataset":
    st.title("📁 Dataset Management")
    
    tab1, tab2 = st.tabs(["📖 View Dataset", "📤 Upload Images"])
    
    with tab1:
        st.markdown("### Current Dataset")
        
        if os.path.exists(DATASET_DIR):
            class_dirs = sorted([d for d in os.listdir(DATASET_DIR) 
                               if os.path.isdir(os.path.join(DATASET_DIR, d))])
            
            if class_dirs:
                selected_style = st.selectbox("Select Art Style", class_dirs)
                class_path = os.path.join(DATASET_DIR, selected_style)
                
                image_files = [f for f in os.listdir(class_path) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                st.markdown(f"**{selected_style}**: {len(image_files)} images")
                
                # images - grid
                cols_per_row = 5
                for i in range(0, len(image_files), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(image_files):
                            img_path = os.path.join(class_path, image_files[i + j])
                            with cols[j]:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
            else:
                st.warning("No dataset found. Please upload images.")
        else:
            st.error("Dataset directory not found.")
    
    with tab2:
        st.markdown("### Upload New Images")
        st.info("Upload images to expand the training dataset")
        
        if os.path.exists(DATASET_DIR):
            class_dirs = sorted([d for d in os.listdir(DATASET_DIR) 
                               if os.path.isdir(os.path.join(DATASET_DIR, d))])
            
            new_style = st.text_input("Or create new style:", "")
            if new_style:
                class_dirs.append(new_style)
            
            selected_class = st.selectbox("Select style for upload:", class_dirs)
            uploaded_files = st.file_uploader(
                "Choose images", 
                type=['png', 'jpg', 'jpeg'], 
                accept_multiple_files=True
            )
            
            if uploaded_files and st.button("Upload Images"):
                class_path = os.path.join(DATASET_DIR, selected_class)
                os.makedirs(class_path, exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    img = Image.open(uploaded_file)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"upload_{timestamp}.png"
                    img.save(os.path.join(class_path, filename))
                
                st.success(f"Uploaded {len(uploaded_files)} images to {selected_class}")
                st.rerun()

# TRAINING PAGE
elif page == "🎓 Train Model":
    st.title("🎓 Train Art Style Classifier")
    
    st.markdown("### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", 5, 100, 20)
        batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
    
    with col2:
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        use_augmentation = st.checkbox("Use Data Augmentation", value=True)
    
    if st.button("🚀 Start Training", type="primary"):
        try:
            with st.spinner("Loading dataset..."):
                X, y, class_names = load_dataset(DATASET_DIR)
                st.session_state.class_names = class_names
                
                st.success(f"Loaded {len(X)} images across {len(class_names)} classes")
            
            # shuffle data
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            # split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            st.info(f"Training: {len(X_train)} | Validation: {len(X_val)}")
            
            # calculate class weights to handle imbalance
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            st.info(f"✓ Class balancing enabled - weights: {', '.join([f'{class_names[i]}: {w:.2f}' for i, w in class_weight_dict.items()])}")
            
            # create model
            with st.spinner("Creating model..."):
                model = create_model(len(class_names))
                st.session_state.model = model
            
            # training progress
            st.markdown("### 📈 Training Progress")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # pre-create metric columns outside loop
            metric_cols = st.columns(4)
            metric_loss = metric_cols[0].empty()
            metric_val_loss = metric_cols[1].empty()
            metric_acc = metric_cols[2].empty()
            metric_val_acc = metric_cols[3].empty()
            
            # create containers for charts
            col1, col2 = st.columns(2)
            with col1:
                loss_chart = st.empty()
            with col2:
                acc_chart = st.empty()
            
            # training history
            history = {
                'loss': [],
                'val_loss': [],
                'accuracy': [],
                'val_accuracy': []
            }
            
            # convert to tf.data.Dataset for better performance
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            
            # setup data augmentation if enabled
            if use_augmentation:
                st.info("🔄 Data augmentation enabled: rotation, flip, zoom, shift")
                if len(X_train) < batch_size:
                    st.warning(f"⚠️ Training set ({len(X_train)} images) is smaller than batch size ({batch_size}). Consider reducing batch size or adding more images.")
                
                # use tf.data augmentation for better performance
                data_augmentation = keras.Sequential([
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.15),
                    layers.RandomZoom(0.15),
                    layers.RandomTranslation(0.1, 0.1),
                    layers.RandomContrast(0.2),
                    layers.RandomBrightness(0.2),
                ])
                
                def augment(image, label):
                    return data_augmentation(image, training=True), label
                
                train_dataset = train_dataset.shuffle(1000).map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            else:
                train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # train model with learning rate scheduling
            update_interval = max(1, epochs // 10)
            best_val_loss = float('inf')
            best_val_acc = 0.0
            patience_counter = 0
            reduce_lr_patience = 3
            best_model_weights = None
            
            for epoch in range(epochs):
                # train one epoch with class balancing
                h = model.fit(
                    train_dataset,
                    epochs=1,
                    validation_data=val_dataset,
                    class_weight=class_weight_dict,
                    verbose=0
                )
                
                # update history
                history['loss'].append(h.history['loss'][0])
                history['val_loss'].append(h.history['val_loss'][0])
                history['accuracy'].append(h.history['accuracy'][0])
                history['val_accuracy'].append(h.history['val_accuracy'][0])
                
                # save best model weights
                current_val_acc = history['val_accuracy'][-1]
                current_val_loss = history['val_loss'][-1]
                
                if current_val_acc > best_val_acc or (current_val_acc == best_val_acc and current_val_loss < best_val_loss):
                    best_val_acc = current_val_acc
                    best_val_loss = current_val_loss
                    best_model_weights = model.get_weights()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= reduce_lr_patience and epoch < epochs - 5:
                    old_lr = float(keras.backend.get_value(model.optimizer.learning_rate))
                    new_lr = old_lr * 0.5
                    keras.backend.set_value(model.optimizer.learning_rate, new_lr)
                    patience_counter = 0
                    st.info(f"📉 Reducing learning rate to {new_lr:.6f}")
                
                # update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {history['loss'][-1]:.4f} - Acc: {history['accuracy'][-1]:.4f}")
                
                # update metrics (lightweight)
                metric_loss.metric("Loss", f"{history['loss'][-1]:.4f}")
                metric_val_loss.metric("Val Loss", f"{history['val_loss'][-1]:.4f}")
                metric_acc.metric("Accuracy", f"{history['accuracy'][-1]:.4f}")
                metric_val_acc.metric("Val Accuracy", f"{history['val_accuracy'][-1]:.4f}")
                
                # update plots only every N epochs or on last epoch (much faster)
                if (epoch + 1) % update_interval == 0 or epoch == epochs - 1:
                    import pandas as pd
                    
                    # use streamlit native charts (much faster than matplotlib)
                    loss_df = pd.DataFrame({
                        'Epoch': list(range(1, len(history['loss']) + 1)),
                        'Training Loss': history['loss'],
                        'Validation Loss': history['val_loss']
                    })
                    
                    acc_df = pd.DataFrame({
                        'Epoch': list(range(1, len(history['accuracy']) + 1)),
                        'Training Accuracy': history['accuracy'],
                        'Validation Accuracy': history['val_accuracy']
                    })
                    
                    with col1:
                        loss_chart.line_chart(loss_df.set_index('Epoch'))
                    
                    with col2:
                        acc_chart.line_chart(acc_df.set_index('Epoch'))
            
            # restore best model weights
            if best_model_weights is not None:
                model.set_weights(best_model_weights)
                st.info(f"✨ Restored best model weights (Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f})")
            
            # save history
            st.session_state.training_history = history
            
            # final evaluation with best weights
            val_loss, val_acc = model.evaluate(val_dataset, verbose=0)
            
            st.success(f"✅ Training Complete! Best Validation Accuracy: {val_acc:.4f}")
            
            # save model
            model_path = os.path.join(MODEL_DIR, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
            model.save(model_path)
            
            # save class names
            with open(os.path.join(MODEL_DIR, 'class_names.json'), 'w') as f:
                json.dump(class_names, f)
            
            st.info(f"💾 Model saved to: {model_path}")
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
    
    # existing model load
    st.markdown("---")
    st.markdown("### 📂 Load Existing Model")
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    
    if model_files:
        selected_model = st.selectbox("Select model to load:", model_files)
        
        if st.button("Load Model"):
            try:
                model_path = os.path.join(MODEL_DIR, selected_model)
                st.session_state.model = keras.models.load_model(model_path)
                
                # load class names
                class_names_path = os.path.join(MODEL_DIR, 'class_names.json')
                if os.path.exists(class_names_path):
                    with open(class_names_path, 'r') as f:
                        st.session_state.class_names = json.load(f)
                
                st.success(f"✅ Model loaded: {selected_model}")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    else:
        st.info("No saved models found. Train a model first.")

# CLASSIFICATION PAGE
elif page == "🔍 Classify":
    st.title("🔍 Classify Artwork")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train or load a model first!")
    else:
        st.success("✅ Model loaded and ready")
        
        tab1, tab2 = st.tabs(["Single Image", "Batch Classification"])
        
        with tab1:
            st.markdown("### Upload Image for Classification")
            st.info("✨ Using Test-Time Augmentation (TTA) for more robust predictions on 4 image variations")
            
            uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    with st.spinner("Classifying..."):
                        predictions = predict_image(st.session_state.model, image, 
                                                   st.session_state.class_names)
                        
                        predicted_class = st.session_state.class_names[np.argmax(predictions)]
                        confidence = np.max(predictions) * 100
                        
                        st.markdown(f"### Predicted Style: **{predicted_class}**")
                        st.markdown(f"### Confidence: **{confidence:.2f}%**")
                        
                        st.markdown("### All Predictions:")
                        
                        # bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['#FF6B6B' if i == np.argmax(predictions) else '#4ECDC4' 
                                 for i in range(len(predictions))]
                        ax.barh(st.session_state.class_names, predictions * 100, color=colors)
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Prediction Confidence by Style')
                        ax.grid(True, alpha=0.3, axis='x')
                        st.pyplot(fig)
                        plt.close()
        
        with tab2:
            st.markdown("### Batch Image Classification")
            
            uploaded_files = st.file_uploader(
                "Upload multiple images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.markdown(f"**{len(uploaded_files)} images uploaded**")
                
                if st.button("Classify All"):
                    results = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        image = Image.open(uploaded_file).convert('RGB')
                        predictions = predict_image(st.session_state.model, image,
                                                   st.session_state.class_names)
                        
                        predicted_class = st.session_state.class_names[np.argmax(predictions)]
                        confidence = np.max(predictions) * 100
                        
                        results.append({
                            'image': image,
                            'filename': uploaded_file.name,
                            'predicted_class': predicted_class,
                            'confidence': confidence
                        })
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    st.markdown("### Results")
                    
                    # rewukts
                    cols_per_row = 3
                    for i in range(0, len(results), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j < len(results):
                                result = results[i + j]
                                with cols[j]:
                                    st.image(result['image'], use_container_width=True)
                                    st.markdown(f"**{result['predicted_class']}**")
                                    st.markdown(f"Confidence: {result['confidence']:.1f}%")

# METRICS PAGE
elif page == "📊 Metrics":
    st.title("📊 Model Performance Metrics")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train or load a model first!")
    elif st.session_state.training_history is None:
        st.info("No training history available. Train a model to see metrics.")
    else:
        history = st.session_state.training_history
        
        # history plots
        st.markdown("### Training History")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history['loss'], label='Training Loss', linewidth=2, marker='o')
            ax.plot(history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training & Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
            ax.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training & Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # final metrics
        st.markdown("### Final Metrics")
        
        cols = st.columns(4)
        cols[0].metric("Final Training Loss", f"{history['loss'][-1]:.4f}")
        cols[1].metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
        cols[2].metric("Final Training Accuracy", f"{history['accuracy'][-1]:.4f}")
        cols[3].metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]:.4f}")
        
        # confusion matrix
        st.markdown("### Confusion Matrix")
        
        if st.button("Generate Confusion Matrix"):
            with st.spinner("Evaluating model on dataset..."):
                try:
                    X, y, _ = load_dataset(DATASET_DIR)
                    
                    # predictions
                    y_pred = []
                    for img in X:
                        pred = predict_image(st.session_state.model, img, 
                                           st.session_state.class_names)
                        y_pred.append(np.argmax(pred))
                    
                    y_pred = np.array(y_pred)
                    
                    # confusion matrix
                    cm = confusion_matrix(y, y_pred)
                    
                    # plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=st.session_state.class_names,
                               yticklabels=st.session_state.class_names,
                               ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                    plt.close()
                    
                    # classification report
                    st.markdown("### Classification Report")
                    report = classification_report(y, y_pred, 
                                                  target_names=st.session_state.class_names)
                    st.text(report)
                    
                except Exception as e:
                    st.error(f"Error generating confusion matrix: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Art Style Classifier**

A deep learning application for identifying artistic styles in paintings.

Built with:
- Streamlit
- TensorFlow/Keras
- Scikit-learn
""")
