import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Mask Compliance & Infection Risk Assessment",
    page_icon="😷",
    layout="wide",
)

# โหลดโมเดล YOLOv8 ที่เทรนมาสำหรับ 3 คลาส
@st.cache_resource
def load_model():
    # ใช้ไฟล์ best.pt ที่อยู่ในโฟลเดอร์เดียวกัน
    return YOLO("best.pt")

model = load_model()

st.title("🏥 Pulmonology Clinic: Mask Compliance & Risk Assessment")
st.markdown(
    "ระบบตรวจสอบการสวมหน้ากากอนามัย (3 คลาส: `with_mask`, `no_mask`, `mask_weared_incorrect`) และประเมินความเสี่ยงโรคทางเดินหายใจ"
)

# Sidebar สำหรับตั้งค่าและคำอธิบาย
st.sidebar.header("⚙️ การตั้งค่าระบบ")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 คำอธิบายคลาส")
st.sidebar.markdown("- **with_mask**: สวมหน้ากากถูกต้อง ✅")
st.sidebar.markdown("- **mask_weared_incorrect**: สวมไม่ถูกต้อง (ใต้คาง/จมูกโผล่) ⚠️")
st.sidebar.markdown("- **no_mask**: ไม่สวมหน้ากาก ❌")

# อัปโหลดรูปภาพ
uploaded_file = st.file_uploader(
    "อัปโหลดรูปภาพผู้มารับบริการ...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # เปิดภาพต้นฉบับ
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ภาพต้นฉบับ")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # แปลงภาพ PIL เป็น OpenCV format (numpy array)
    img_array = np.array(image)

    # ประมวลผลด้วยโมเดล YOLOv8
    results = model(img_array, conf=confidence_threshold)

    # ดึงภาพที่มีการวาด Bounding Box (ค่าเริ่มต้นของ YOLO จะเป็น BGR)
    res_plotted = results[0].plot()

    # **จุดที่แก้ไข:** แปลงสีจาก BGR เป็น RGB เพื่อให้สีภาพกลับมาถูกต้องบน Streamlit
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("ผลการวิเคราะห์")
        st.image(
            res_plotted_rgb,
            caption="ผลการตรวจจับการสวมหน้ากากอนามัย",
            use_container_width=True,
        )

    # ดึงข้อมูล Box และ Classes ที่ตรวจพบเพื่อคำนวณความเสี่ยง
    boxes = results[0].boxes
    class_names = model.names

    counts = {"with_mask": 0, "mask_weared_incorrect": 0, "no_mask": 0}

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        if cls_name in counts:
            counts[cls_name] += 1

    total_detected = sum(counts.values())

    # แสดงผลสรุปสถิติ
    st.markdown("---")
    st.subheader("📋 สรุปผลการตรวจจับ (Compliance Metrics)")

    m1, m2, m3 = st.columns(3)
    m1.metric("สวมถูกต้อง (With Mask)", counts["with_mask"])
    m2.metric(
        "สวมไม่ถูกต้อง (Incorrect)", counts["mask_weared_incorrect"]
    )
    m3.metric("ไม่สวมหน้ากาก (No Mask)", counts["no_mask"])

    # คำนวณความเสี่ยงโรคทางเดินหายใจ
    # กำหนดน้ำหนักความเสี่ยง: no_mask = 1.0, incorrect = 0.5, with_mask = 0.0
    if total_detected > 0:
        risk_score = (
            (counts["no_mask"] * 1.0)
            + (counts["mask_weared_incorrect"] * 0.5)
        ) / total_detected

        st.markdown("---")
        st.subheader("⚠️ การประเมินความเสี่ยงโรคทางเดินหายใจ (Infection Risk)")

        if risk_score > 0.6:
            st.error(
                f"**ระดับความเสี่ยง: สูงมาก (High Risk)** (คะแนนความเสี่ยง: {risk_score:.2f})"
            )
            st.markdown(
                "คำแนะนำ: พื้นที่หนาแน่นมีความเสี่ยงสูงในการแพร่กระจายเชื้อ ควรกวดขันการสวมหน้ากากอนามัยทันที"
            )
        elif risk_score > 0.2:
            st.warning(
                f"**ระดับความเสี่ยง: ปานกลาง (Moderate Risk)** (คะแนนความเสี่ยง: {risk_score:.2f})"
            )
            st.markdown(
                "คำแนะนำ: พบผู้สวมหน้ากากไม่ถูกต้องหรือ Loei บางส่วน ควรให้คำแนะนำเพิ่มเติม"
            )
        else:
            st.success(
                f"**ระดับความเสี่ยง: ต่ำ (Low Risk)** (คะแนนความเสี่ยง: {risk_score:.2f})"
            )
            st.markdown(
                "คำแนะนำ: มาตรการควบคุมการสวมหน้ากากอยู่ในเกณฑ์ดีเยี่ยม"
            )