import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# โหลดโมเดลที่คุณเทรนเสร็จแล้ว (ใช้แคชเพื่อความรวดเร็ว)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.warning("ยังไม่พบไฟล์โมเดล 'best.pt' กรุณาเทรนโมเดลและนำไฟล์มาใส่ก่อนครับ")

st.title("🏥 Mask Compliance Detection in High-Risk Hospital")
st.write("ระบบตรวจสอบการสวมหน้ากากอนามัยสำหรับแผนกความเสี่ยงสูง")

# เมนูด้านข้างสำหรับเลือกโหมดการทำงาน
option = st.sidebar.selectbox(
    "เลือกวิธีการทดสอบระบบ", ("ถ่ายภาพจากกล้อง", "อัปโหลดรูปภาพ")
)

if option == "อัปโหลดรูปภาพ":
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        if 'model' in globals():
            # ให้โมเดล YOLOv8 ทำนายผล
            results = model(image_np)
            res_plotted = results[0].plot() # วาดกล่อง Bounding Box ลงบนภาพ
            
            # แปลงสีจาก BGR เป็น RGB สำหรับแสดงผลบน Streamlit
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_plotted_rgb, caption="ผลการตรวจจับการสวมหน้ากาก", use_container_width=True)
            st.success("ประมวลผลสำเร็จ!")
        else:
            st.image(image_np, caption="ภาพที่อัปโหลด (ยังไม่ได้โหลดโมเดล)", use_container_width=True)

elif option == "ถ่ายภาพจากกล้อง":
    # Widget ถ่ายภาพจากกล้อง
    camera_image = st.camera_input("กดปุ่มเพื่อถ่ายภาพตรวจสอบ")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if 'model' in globals():
            # ประมวลผลด้วยโมเดล YOLOv8
            results = model(cv2_img)
            res_plotted = results[0].plot()
            
            # แปลงสีแสดงผล
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_plotted_rgb, caption="ผลการตรวจจับจากกล้อง", use_container_width=True)
            st.success("ตรวจจับสำเร็จ!")
        else:
            st.image(cv2_img, caption="ภาพจากกล้อง (ยังไม่ได้โหลดโมเดล)", use_container_width=True)