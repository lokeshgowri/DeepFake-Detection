import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os
import requests
import base64
import pandas as pd

# ---------- AI Generated Check ----------
def check_ai_generated(image_path):
    url = "https://api.thehive.ai/api/v2/task/sync"
    api_key = "YOUR_API_KEY_HERE"

    with open(image_path, "rb") as f:
        img_data = f.read()

    img_base64 = base64.b64encode(img_data).decode("utf-8")

    payload = {
        "model": "deepfake",
        "input": img_base64
    }

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    result = response.json()

    label = result["output"][0]["classes"][0]["class"]
    return label


# ---------- Load Model ----------
model = tf.keras.models.load_model('my_model.keras')


# ---------- Video Prediction ----------
def img_pred(video_path, model, batch_size=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (150, 150))
        img = img.reshape(1, 150, 150, 3)
        frames.append(img)

        if len(frames) >= batch_size:
            batch = np.vstack(frames)
            predictions = model.predict(batch, verbose=0)

            for p in predictions:
                if np.argmax(p) == 0:
                    fake_count += 1
            frames = []

    cap.release()
    fake_percentage = (fake_count / frame_count) * 100 if frame_count else 0
    return fake_percentage, frame_count, fake_count


# ---------- Image Prediction ----------
def photo_pred(image, model):
    img = Image.open(image).resize((150, 150))
    img = np.array(img)

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    img = img.reshape(1, 150, 150, 3)
    prediction = model.predict(img, verbose=0)[0]

    total = prediction.sum()
    fake_conf = (prediction[0] / total) * 100
    real_conf = (prediction[1] / total) * 100

    fake_conf = min(fake_conf, 99.9)
    real_conf = min(real_conf, 99.9)

    if fake_conf > real_conf:
        return "Fake", fake_conf
    else:
        return "Real", real_conf


# ---------- Face Detection (Bounding Boxes) ----------
def detect_faces(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    return image, faces


# ---------- Heatmap Visualization ----------
def generate_heatmap(image_path, model):
    try:
        orig_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img_array = np.array(Image.fromarray(orig_img).resize((150, 150)))[..., :3].reshape(1, 150, 150, 3).astype('float32')
        last_conv = next(l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D))
        grad_model = tf.keras.Model(model.inputs, [last_conv.output, model.output])
        with tf.GradientTape() as tape:
            inputs = {grad_model.input_names[0]: img_array} if grad_model.input_names else img_array
            conv_out, preds = grad_model(inputs)
            loss = preds[:, tf.argmax(preds[0])]
        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_out), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = cv2.resize(heatmap[0].numpy(), (orig_img.shape[1], orig_img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    except Exception:
        return None

# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title="Deepfake Detection", layout="centered")
    
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.markdown("<h1 style='text-align: center;'>Deepfake Detection</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    choice = st.radio("Select file type to upload", ["Video", "Image"], horizontal=True)

    # ---------- VIDEO ----------
    if choice == "Video":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

        if video_file:
            video_path = "uploaded_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())

            st.video(video_file)

            with st.spinner("Analyzing video..."):
                fake_percentage, total_frames, fake_frames = img_pred(video_path, model)

            real_percentage = 100 - fake_percentage
            real_frames = total_frames - fake_frames

            label = "Fake" if fake_percentage > 50 else "Real"
            color = "#ff4d4d" if label == "Fake" else "#4CAF50"

            if label == "Fake":
                confidence_text = f"Fake Confidence: {fake_percentage:.2f}%"
            else:
                confidence_text = f"Real Confidence: {real_percentage:.2f}%"
                
            st.session_state.history.append({"Type": "Video", "Label": label, "Confidence": f"{fake_percentage if label == 'Fake' else real_percentage:.2f}%"})

            st.markdown(f"""
            <div style="padding:1rem; background:{color}; color:white; border-radius:10px;">
                <h4>{label} Video</h4>
                <p>{confidence_text}</p>
                <p>Fake Frames: {fake_frames}/{total_frames}</p>
                <p>Real Frames: {real_frames}/{total_frames}</p>
            </div>
            """, unsafe_allow_html=True)

    # ---------- IMAGE ----------
    else:
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if image_file:
            st.image(image_file, caption="Uploaded Image", width="stretch")

            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())

            # Overall image prediction
            with st.spinner("Analyzing image..."):
                prediction, confidence = photo_pred(image_file, model)

            # Face detection only (boxes will follow overall prediction)
            with st.spinner("Detecting faces and drawing bounding boxes..."):
                image_with_boxes, faces = detect_faces(image_path)

                # Determine box color based on overall image prediction
                if prediction == "Fake":
                    box_color = (0, 0, 255)  # Red
                else:
                    box_color = (0, 255, 0)  # Green

                for (x, y, w, h) in faces:
                    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(
                        image_with_boxes,
                        f"{prediction} {confidence:.2f}%",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2
                    )

                st.image(image_with_boxes, caption="Face-Level Detection", width="stretch")

            # AI-generated check
            with st.spinner("Checking AI-generated..."):
                try:
                    ai_label = check_ai_generated(image_path)
                except:
                    ai_label = None

            if ai_label == "ai-generated":
                color = "#ff9933"
                text = "AI-Generated Image"
                final_label = "AI-Generated"
            elif prediction == "Fake":
                color = "#ff4d4d"
                text = f"Deepfake Image — Model confidence: {confidence:.2f}%"
                final_label = "Fake"
            else:
                color = "#4CAF50"
                text = f"Real Image — Model confidence: {confidence:.2f}%"
                final_label = "Real"

            st.session_state.history.append({"Type": "Image", "Label": final_label, "Confidence": f"{confidence:.2f}%"})

            st.markdown(f"""
            <div style="padding:1rem; background:{color}; color:white; border-radius:10px;">
                <h4>{text}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Generating Explanation Heatmap..."):
                heatmap_img = generate_heatmap(image_path, model)
                if heatmap_img is not None:
                    st.image(heatmap_img, caption="Explanation Heatmap (Highlights what influenced the prediction)", width="stretch")

            os.remove(image_path)

            st.caption("⚠️ Confidence is based on model prediction and is not absolute proof.")

    st.sidebar.title("Dashboard & Analytics")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.sidebar.write("Detection History")
        st.sidebar.dataframe(df)
        st.sidebar.write("Label Distribution")
        st.sidebar.bar_chart(df['Label'].value_counts())


if __name__ == "__main__":
    main()