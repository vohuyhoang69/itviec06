
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
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
st.title("Trang dành cho Doanh nghiệp")
st.write("Tại đây, doanh nghiệp có thể xem kết quả phân tích và gợi ý cải thiện.")

import pandas as pd

# Đọc dữ liệu công ty từ file
df = pd.read_excel("Overview_Companies.xlsx")

# Tạo danh sách hiển thị dạng "id - Tên công ty"
display_list = [f"{row['id']} - {row['Company Name']}" for _, row in df.iterrows()]

# Ô chọn
selected_display = st.selectbox("Chọn công ty (hiện id và Tên)", display_list)

# Tách id từ chuỗi đã chọn
selected_id = int(selected_display.split(" - ")[0])

# Lấy tên công ty theo id
selected_name = df[df['id'] == selected_id]['Company Name'].values[0]
st.write(f"✅ Bạn đã chọn công ty: **{selected_name}**, ID: {selected_id}")


## Tìm kiếm các công ty tương đồng
# Load embedding đã lưu
with open("bert_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)
# Tìm chỉ số dòng tương ứng với selected_id
idx = df[df['id'] == selected_id].index[0]

# Tính similarity
sim_scores = cosine_similarity([embeddings[idx]], embeddings).flatten()
# Lấy top 3 công ty gần giống nhất (trừ chính nó ra)
top_indices = sim_scores.argsort()[-4:-1][::-1]

import streamlit as st

# Giả sử có DataFrame df chứa thông tin các công ty
# Và top_indices chứa chỉ số 3 công ty gần giống nhất

st.write("### 🔎 Top 3 công ty gần giống nhất:")

for idx in top_indices:
    row = df.iloc[idx]

    with st.container():
     st.markdown(f'### {row["Company Name"]}')
     st.caption(f'**Ngành nghề:** {row["Company industry"]} | **Quy mô:** {row["Company size"]} | **Địa điểm:** {row["Location"]}')
    
     st.markdown(f'**Điểm nổi bật trong mô tả tuyển dụng:** {row["Company overview"]}')
     st.markdown(f'**Kỹ năng nổi bật:** {row["Our key skills"]}')
     st.markdown(f'**Vì sao nên làm ở đây:** {row["Why you will love working here"]}')
    
     st.warning("💡 Doanh nghiệp của bạn đã đủ hấp dẫn hơn đối thủ này chưa?")
     st.divider()

#Phân tích các reviews của nhân viên đánh giá 
# Đọc dữ liệu review
df_reviews = pd.read_excel("Reviews.xlsx")

# Chuyển về datetime, những giá trị lỗi sẽ thành NaT
df_reviews['Cmt_day'] = pd.to_datetime(df_reviews['Cmt_day'], errors='coerce')

# Chỉ giữ những dòng có ngày hợp lệ
df_reviews = df_reviews.dropna(subset=['Cmt_day'])

# Tạo cột Năm, đảm bảo toàn bộ là số nguyên
df_reviews['Năm'] = df_reviews['Cmt_day'].dt.year.astype(int)

# Đếm số lượng review theo id và năm
summary = df_reviews.groupby(['id', 'Năm']).size().reset_index(name='Số lượng review')

# Lọc theo id của công ty đã chọn
summary_selected = summary[summary['id'] == selected_id]


import matplotlib.pyplot as plt
import seaborn as sns

if not summary_selected.empty:
    st.write(f"Biểu đồ số lượng review theo năm của công ty **{selected_name}**")

    fig, ax = plt.subplots()
    sns.barplot(data=summary_selected, x='Năm', y='Số lượng review', ax=ax)
    ax.set_xlabel("Năm")
    ax.set_ylabel("Số lượng review")
    ax.set_title(f"Số lượng review theo năm - {selected_name}")

    st.pyplot(fig)
else:
    st.warning("Công ty này chưa có dữ liệu review hoặc dữ liệu thiếu thông tin ngày.")

import streamlit as st
import pandas as pd
import pickle
from bertopic import BERTopic

# Load dữ liệu review
df = pd.read_excel("Reviews.xlsx")

# Xử lý ngày tháng và tạo cột Năm
df['Cmt_day'] = pd.to_datetime(df['Cmt_day'], format='%B %Y', errors='coerce')
df['Năm'] = df['Cmt_day'].dt.year

# Ghép nội dung review
df['Full_Review'] = (
    df['Title'].astype(str) + ' ' +
    df['What I liked'].astype(str) + ' ' +
    df['Suggestions for improvement'].astype(str)
)

# Load mô hình BERTopic
topic_model = BERTopic.load("bertopic_model")

# Lọc những dòng có năm hợp lệ
df = df.dropna(subset=['Năm'])
df['Năm'] = df['Năm'].astype(int)

# Lọc danh sách 3 năm gần nhất của công ty đã chọn
years = sorted(df[df['id'] == selected_id]['Năm'].unique(), reverse=True)[:3]
years = sorted(years)

for year in years:
    st.write(f"## 🔥 Top 5 chủ đề được bàn luận nhiều nhất năm {year} của công ty **{selected_name}**")

    df_year = df[(df['Năm'] == year) & (df['id'] == selected_id)]

    topics, _ = topic_model.transform(df_year['Full_Review'].tolist())
    df_year['Topic'] = topics

    df_year = df_year[df_year['Topic'] != -1]

    if df_year.empty:
        st.warning("Không có dữ liệu chủ đề rõ ràng cho năm này.")
        continue

    top_topic_counts = df_year['Topic'].value_counts().head(5).reset_index()
    top_topic_counts.columns = ['Topic', 'Số lượng']

    def extract_keywords(topic_id):
        keywords = topic_model.get_topic(topic_id)
        if isinstance(keywords, list):
            return ", ".join([word for word, _ in keywords[:5]])
        return "Không rõ"

    top_topic_counts['Nội dung chủ đề'] = top_topic_counts['Topic'].apply(extract_keywords)

    st.dataframe(top_topic_counts[['Topic', 'Nội dung chủ đề', 'Số lượng']])
    st.bar_chart(top_topic_counts.set_index('Nội dung chủ đề')['Số lượng'])

    st.divider()


