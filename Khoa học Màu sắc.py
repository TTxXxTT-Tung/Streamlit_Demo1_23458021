from pathlib import Path
import streamlit as st

from utils.ui import show_optional_logos, footer_note

st.set_page_config(page_title="PRINT COLOUR SCIENCE LAB", page_icon="🎨", layout="wide")

show_optional_logos(show_banner=True)
st.title("PROJECT QUÁ TRÌNH – KHOA HỌC MÀU TRONG CÔNG NGHỆ IN")
st.subheader("Ứng dụng Streamlit trong Công nghệ in")

st.markdown("""
**Tên đề tài:** KHOA HỌC MÀU SẮC.  
**Sinh viên thực hiện:** Quách Thanh Tùng  
**MSSV:** 23458021  
**Lớp:** 23VL58CT1 – Công nghệ kỹ thuật In  
**Giảng viên hướng dẫn:** ThS. Trần Tiến Đức  
""")
