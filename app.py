import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import plotly.graph_objects as go

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Mask Compliance & Disease Risk Assessment",
    page_icon="🏥",
    layout="wide",
)

# โหลดโมเดล (ใช้แคชเพื่อความเร็ว)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.warning("ยังไม่พบไฟล์โมเดล 'best.pt' กรุณาตรวจสอบไฟล์โมเดลอีกครั้งครับ")

st.title("🏥 Pulmonology Clinic: Mask Compliance & Risk Assessment")
st.write("ระบบตรวจสอบการสวมหน้ากากอนามัย (3 คลาส) และประเมินความเสี่ยงโรคติดต่อทางเดินหายใจ")

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
        # ประมวลผลด้วยโมเดล YOLOv8
        results = model(image_np, conf=0.4, iou=0.45, imgsz=640)
        
        # ดึงภาพที่มีการวาด Bounding Box (ค่าเริ่มต้นของ YOLO จะเป็น BGR)
        res_plotted = results[0].plot()
        
        # แปลงสีจาก BGR เป็น RGB เพื่อให้สีภาพกลับมาถูกต้องบน Streamlit (แก้ปัญหาภาพติดสีฟ้า)
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # ดึงข้อมูล Box และ Classes เพื่อแยกนับ 3 คลาส
        boxes = results[0].boxes
        class_names = model.names
        
        counts = {
            "with_mask": 0,
            "mask_weared_incorrect": 0,
            "no_mask": 0
        }
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                if cls_name in counts:
                    counts[cls_name] += 1
                else:
                    name_lower = cls_name.lower()
                    if 'incorrect' in name_lower or 'weared_incorrect' in name_lower:
                        counts["mask_weared_incorrect"] += 1
                    elif 'no' in name_lower or 'without' in name_lower:
                        counts["no_mask"] += 1
                    else:
                        counts["with_mask"] += 1
                        
        total_people = sum(counts.values())
        
        # แสดงผลภาพที่ผ่านการกรองและแปลงสี RGB แล้ว
        st.image(res_plotted_rgb, caption="ผลการตรวจจับการสวมหน้ากากอนามัย", use_container_width=True)
        
        # แสดงผลสรุปจำนวนคนแยกตามคลาส
        st.markdown("---")
        st.subheader("📊 สรุปผลการตรวจจับ (Compliance Metrics)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 คนทั้งหมด", f"{total_people} คน")
        col2.metric("✅ สวมถูกต้อง", f"{counts['with_mask']} คน")
        col3.metric("⚠️ สวมไม่ถูกต้อง", f"{counts['mask_weared_incorrect']} คน")
        col4.metric("❌ ไม่สวมหน้ากาก", f"{counts['no_mask']} คน")
        
        # ประเมินความเสี่ยงโรคทางเดินหายใจ (รวมน้ำหนักสวมไม่ถูกต้อง = 0.5)
        if total_people > 0:
            risk_score_ratio = (
                (counts["no_mask"] * 1.0) + 
                (counts["mask_weared_incorrect"] * 0.5)
            ) / total_people
            
            base_risk = int(risk_score_ratio * 100)
            unmasked_total = counts["no_mask"] + counts["mask_weared_incorrect"]
            
            risk_flu = min(base_risk + (unmasked_total * 5), 100)      
            risk_rsv = min(base_risk + (unmasked_total * 4), 100)      
            risk_tb = min(int(base_risk * 0.9 + (unmasked_total * 6)), 100)  
            risk_pneumonia = min(base_risk + (unmasked_total * 4), 100) 
            risk_pertussis = min(int(base_risk * 0.8 + (unmasked_total * 4)), 100) 
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