import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Photo booth ")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ================= SIDEBAR =================
    st.sidebar.header("Resize")
    scale = st.sidebar.slider("Scale (%)", 10, 200, 100)

    st.sidebar.header("Adjustments")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)

    st.sidebar.header("Filters")
    grayscale = st.sidebar.checkbox("Grayscale")
    blur_strength = st.sidebar.slider("Blur", 0, 25, 0)
    sharpen = st.sidebar.checkbox("Sharpen")
    warm = st.sidebar.checkbox("Warm Filter")

    st.sidebar.header("Portrait Mode")
    portrait = st.sidebar.checkbox("Enable Portrait Blur")
    portrait_blur = st.sidebar.slider("Background Blur Strength", 0, 51, 15)

    # ================= PIPELINE =================

    # Resize
    h, w = image.shape[:2]
    new_w = int(w * scale / 100)
    new_h = int(h * scale / 100)
    resized = cv2.resize(image, (new_w, new_h))

    # Adjustments
    adjusted = cv2.convertScaleAbs(resized, alpha=contrast, beta=brightness)

    # Filters
    filtered = adjusted.copy()

    if grayscale:
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if blur_strength > 0:
        k = blur_strength if blur_strength % 2 != 0 else blur_strength + 1
        filtered = cv2.GaussianBlur(filtered, (k, k), 0)

    if sharpen:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        filtered = cv2.filter2D(filtered, -1, kernel)

    if warm:
        b, g, r = cv2.split(filtered)
        r = cv2.add(r, 20)
        b = cv2.subtract(b, 10)
        filtered = cv2.merge((b, g, r))

    # ================= PORTRAIT =================
    final = filtered.copy()

    if portrait and portrait_blur > 0:
        h, w = final.shape[:2]

        k = portrait_blur if portrait_blur % 2 != 0 else portrait_blur + 1
        blurred = cv2.GaussianBlur(final, (k, k), 0)

        mask = np.zeros((h, w), dtype=np.uint8)

        cx, cy = w // 2, h // 2
        radius = min(h, w) // 3

        cv2.circle(mask, (cx, cy), radius, 255, -1)

        mask = cv2.GaussianBlur(mask, (51, 51), 0)

        mask = mask / 255.0
        mask = np.stack([mask]*3, axis=2)

        final = (final * mask + blurred * (1 - mask)).astype(np.uint8)

    # ================= DOWNLOAD =================
    _, buffer = cv2.imencode(".png", final)
    img_bytes = buffer.tobytes()

    st.sidebar.header("Download")
    st.sidebar.download_button(
        "Download Edited Image",
        data=img_bytes,
        file_name="edited.png",
        mime="image/png"
    )

    # ================= DISPLAY =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Edited")
        st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), width=new_w)