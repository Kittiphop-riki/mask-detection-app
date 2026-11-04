import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import plotly.graph_objects as go

# โหลดโมเดล (ใช้แคชเพื่อความเร็ว)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.warning("ยังไม่พบไฟล์โมเดล 'best.pt' กรุณาตรวจสอบไฟล์โมเดลอีกครั้งครับ")

st.title("🏥 Mask Compliance & Disease Risk Assessment")
st.write("ระบบประเมินความเสี่ยงโรคติดต่อทางเดินหายใจจากพฤติกรรมการสวมหน้ากากอนามัย")

option = st.sidebar.selectbox(
    "เลือกวิธีการทดสอบระบบ", ("ถ่ายภาพจากกล้อง", "อัปโหลดรูปภาพ")
)

def create_donut_chart(percentage, title_text):
    fig = go.Figure(data=[go.Pie(
        values=[percentage, 100 - percentage],
        labels=["ความเสี่ยง", "ปลอดภัย"],
        hole=0.75,
        marker=dict(colors=['#00a8cc', '#e4e4e4']),
        textinfo='none',
        hoverinfo='skip'
    )])
    
    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        title_font_size=14,
        annotations=[dict(
            text=f"<b>{percentage}%</b>",
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False,
            font_color="#1f77b4"
        )],
        showlegend=False,
        margin=dict(t=40, b=10, l=10, r=10),
        height=180,
        width=180
    )
    return fig

def process_and_display(image_np):
    if 'model' in globals():
        # ปรับ conf=0.4 และ iou=0.45 เพื่อป้องกันการจับซ้อนกันและกรองผลลัพธ์ที่ผิดพลาดออก
        results = model(image_np, conf=0.4, iou=0.45, imgsz=640)
        
        # ใช้ฟังก์ชัน plot() ของ YOLO โดยตรง ซึ่งจัดการเรื่อง NMS มาให้เรียบร้อยแล้ว ภาพจะไม่ซ้อนกันรกครับ
        res_plotted = results[0].plot()
        
        # นับจำนวนและวิเคราะห์คลาสจาก boxes ที่ผ่านการกรองแล้ว
        boxes = results[0].boxes
        class_names = results[0].names
        
        mask_count = 0
        no_mask_count = 0
        
        if boxes is not None and len(boxes) > 0:
            for cls_id in boxes.cls:
                name = class_names[int(cls_id)].lower()
                if 'no' in name or 'without' in name or 'not' in name:
                    no_mask_count += 1
                else:
                    mask_count += 1
                    
        total_people = mask_count + no_mask_count
        
        # แสดงผลภาพที่ผ่านการกรอบและไม่ซ้อนกัน
        st.image(res_plotted, caption="ผลการตรวจจับการสวมหน้ากาก", use_container_width=True)
        
        # แสดงผลสรุปจำนวนคน
        st.markdown("---")
        st.subheader("📊 สรุปผลการตรวจจับ")
        col1, col2, col3 = st.columns(3)
        col1.metric("👥 คนทั้งหมด", f"{total_people} คน")
        col2.metric("✅ สวมหน้ากาก", f"{mask_count} คน")
        col3.metric("❌ ไม่สวมหน้ากาก", f"{no_mask_count} คน")
        
        # ประเมินความเสี่ยงโรคทางเดินหายใจ
        if total_people > 0:
            unmasked_ratio = no_mask_count / total_people
            base_risk = int(unmasked_ratio * 100)
            
            risk_flu = min(base_risk + (no_mask_count * 5), 100)       
            risk_rsv = min(base_risk + (no_mask_count * 4), 100)       
            risk_tb = min(int(base_risk * 0.9 + (no_mask_count * 6)), 100)  
            risk_pneumonia = min(base_risk + (no_mask_count * 4), 100) 
            risk_pertussis = min(int(base_risk * 0.8 + (no_mask_count * 4)), 100) 
        else:
            risk_flu = risk_rsv = risk_tb = risk_pneumonia = risk_pertussis = 0

        st.markdown("---")
        st.subheader("🦠 ระดับความเสี่ยงการแพร่ระบาดของโรค")
        
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            st.plotly_chart(create_donut_chart(risk_flu, "ไข้หวัดใหญ่ (Influenza)"), use_container_width=True)
        with r_col2:
            st.plotly_chart(create_donut_chart(risk_rsv, "ไวรัส RSV"), use_container_width=True)
        with r_col3:
            st.plotly_chart(create_donut_chart(risk_tb, "วัณโรคปอด (TB)"), use_container_width=True)
            
        r_col4, r_col5 = st.columns(2)
        with r_col4:
            st.plotly_chart(create_donut_chart(risk_pneumonia, "ปอดอักเสบ / ปอดบวม"), use_container_width=True)
        with r_col5:
            st.plotly_chart(create_donut_chart(risk_pertussis, "โรคไอกรน (Pertussis)"), use_container_width=True)
            
        st.success("ประมวลผลและวิเคราะห์ความเสี่ยงสำเร็จ!")
    else:
        st.error("ยังไม่ได้โหลดโมเดล กรุณาตรวจสอบไฟล์ best.pt")

if option == "อัปโหลดรูปภาพ":
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        process_and_display(image_np)

elif option == "ถ่ายภาพจากกล้อง":
    camera_image = st.camera_input("กดปุ่มเพื่อถ่ายภาพตรวจสอบ")
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        process_and_display(cv2_img)