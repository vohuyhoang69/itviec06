import  streamlit as st
import  pandas as pd
import matplotlib as plt
import seaborn as sns
import os
from PIL import Image



st.sidebar.markdown("""
    <style>
    .sidebar-info {
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-info">
👤 Họ và tên: <b>Võ Huy Hoàng</b><br>
📧 Email: <b>2356210012@hcmussh.edu.vn</b>
</div>
""", unsafe_allow_html=True)


st.title("Đồ án TN DS - ML - CSC - k304")
st.subheader("Chào mừng các bạn đến với đồ án tốt nghiệp Data Science - Machine Learning của Trung tâm Tin học CSC")
st.write("### Có 2 chủ đề trong project2: Content-Based Company Similarity Recommendation and 'Recommend or Not' Classification for Candidates")
import os
col1, col2 = st.columns(2)

with col1:
    st.image("images/CLASSIFICATION_IN_MACHINE_LEARNING.jpg", width=500)
    st.caption("Chủ đề 1")

with col2:
    st.image("images/CONTENT-BASED_FILTERING.jpg", width=500)
    st.caption("Chủ đề 2")

st.markdown("""
    <style>
    .custom-text {
        font-size: 25px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-text">

### Giới thiệu Project Recommendation & Classification ITviec

**1. Bối cảnh & Ý nghĩa**  
ITviec là nền tảng việc làm lớn trong lĩnh vực Công nghệ Thông tin tại Việt Nam.  
Trên ITviec, ứng viên có thể xem đánh giá, nhận xét về các công ty trước khi quyết định ứng tuyển.  
Tuy nhiên, số lượng đánh giá rất lớn, thông tin rời rạc → cần hệ thống hỗ trợ phân tích, gợi ý.  

**2. Mục tiêu Project**  
 Xây dựng hệ thống giúp:  
- Đề xuất các công ty tương tự dựa trên thông tin mô tả của công ty (Content-Based Filtering).  
- Dự đoán khả năng "Recommend" của công ty dựa trên phân tích đánh giá của nhân viên/ứng viên.  

**3. Lợi ích của hệ thống**  
*Đối với doanh nghiệp:*  
→ Biết được công ty mình được đánh giá ra sao, thuộc nhóm nào.  
→ Tham khảo các công ty tương tự để cải thiện môi trường làm việc.  

*Đối với ứng viên:*  
→ Tham khảo các công ty nên ứng tuyển.  
→ Có thêm công cụ hỗ trợ chọn lọc công ty phù hợp.  

</div>
""", unsafe_allow_html=True)








