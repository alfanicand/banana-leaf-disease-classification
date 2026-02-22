import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================
# Config
# =============================
st.set_page_config(
    page_title="Perbandingan Model Penyakit Daun Pisang",
    layout="centered"
)

CLASS_NAMES = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']

CLASS_INFO = {
    "cordana": {
        "deskripsi": "Cordana merupakan penyakit daun yang ditandai dengan bercak coklat hingga keabu-abuan pada permukaan daun.",
        "penanganan": "Penanganan dapat dilakukan dengan pemangkasan daun terinfeksi dan penggunaan fungisida sesuai anjuran."
    },
    "healthy": {
        "deskripsi": "Daun dalam kondisi sehat tanpa gejala penyakit yang terlihat.",
        "penanganan": "Lakukan perawatan rutin, pemupukan seimbang, dan pemantauan berkala untuk menjaga kesehatan tanaman."
    },
    "pestalotiopsis": {
        "deskripsi": "Pestalotiopsis ditandai dengan bercak tidak beraturan berwarna coklat dengan tepi lebih gelap.",
        "penanganan": "Pengendalian dapat dilakukan dengan sanitasi kebun, mengurangi kelembaban berlebih, dan aplikasi fungisida bila diperlukan."
    },
    "sigatoka": {
        "deskripsi": "Sigatoka ditandai dengan bercak kecil memanjang berwarna kuning hingga coklat yang dapat menyatu dan mengering.",
        "penanganan": "Pengendalian dilakukan dengan pemangkasan daun terinfeksi dan penyemprotan fungisida secara teratur sesuai rekomendasi."
    }
}

# =============================
# Load models (Fixed Feature)
# =============================
@st.cache_resource
def load_models():
    model_mobilenet = tf.keras.models.load_model(
        "mobilenetv2_fixedfeature.keras", compile=False
    )
    model_efficientnet = tf.keras.models.load_model(
        "efficientnetb0_fixedfeature.keras", compile=False
    )
    return model_mobilenet, model_efficientnet

model_mn, model_ef = load_models()

# =============================
# Preprocessing (SESUAI SKRIPSI)
# - resize 224x224
# - TANPA normalisasi manual
# - preprocess_input ADA DI DALAM MODEL
# =============================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype="float32")
    img = np.expand_dims(img, axis=0)
    return img

# =============================
# UI
# =============================
st.title("Perbandingan MobileNetV2 vs EfficientNetB0 (Fixed Feature)")
st.write(
    "Aplikasi ini membandingkan hasil klasifikasi penyakit daun pisang "
    "menggunakan **MobileNetV2 dan EfficientNetB0 pada skenario Fixed Feature** "
)

uploaded_file = st.file_uploader(
    "Upload gambar daun pisang",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # ===== GAMBAR DI TENGAH (TIDAK MELEBAR / TIDAK MEMANJANG) =====
    st.subheader("Gambar Input")

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.image(
            image,
            use_container_width=True,
            caption="Citra daun pisang"
        )

    x = preprocess_image(image)

    # =============================
    # Prediction
    # =============================
    pred_mn = model_mn.predict(x, verbose=0)[0]
    pred_ef = model_ef.predict(x, verbose=0)[0]

    idx_mn = int(np.argmax(pred_mn))
    idx_ef = int(np.argmax(pred_ef))

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### MobileNetV2 (Fixed Feature)")
        st.write(f"**Prediksi:** {CLASS_NAMES[idx_mn]}")
        st.write(f"**Confidence:** {pred_mn[idx_mn]*100:.2f}%")

        st.bar_chart(
            {CLASS_NAMES[i]: float(pred_mn[i]) for i in range(len(CLASS_NAMES))}
        )
        st.markdown("**Deskripsi Penyakit:**")
        st.write(CLASS_INFO[CLASS_NAMES[idx_mn]]["deskripsi"])

        st.markdown("**Saran Penanganan:**")
        st.write(CLASS_INFO[CLASS_NAMES[idx_mn]]["penanganan"])

    with col2:
        st.markdown("### EfficientNetB0 (Fixed Feature)")
        st.write(f"**Prediksi:** {CLASS_NAMES[idx_ef]}")
        st.write(f"**Confidence:** {pred_ef[idx_ef]*100:.2f}%")

        st.bar_chart(
            {CLASS_NAMES[i]: float(pred_ef[i]) for i in range(len(CLASS_NAMES))}
        )
        st.markdown("**Deskripsi Penyakit:**")
        st.write(CLASS_INFO[CLASS_NAMES[idx_ef]]["deskripsi"])

        st.markdown("**Saran Penanganan:**")
        st.write(CLASS_INFO[CLASS_NAMES[idx_ef]]["penanganan"])
