
import streamlit as st
import pandas as pd
import pickle
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic

# ===== Sidebar thông tin cá nhân =====
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

# ===== Tiêu đề chính =====
st.title("Trang dành cho Người xin việc")
st.write("Tham khảo danh sách công ty phù hợp và được Recommend cao.")

# ===== Load dữ liệu công ty =====
df = pd.read_excel("Overview_Companies.xlsx")
df['Full_Description'] = (
    df['Company Name'].astype(str) + ' ' +
    df['Company Type'].astype(str) + ' ' +
    df['Company industry'].astype(str) + ' ' +
    df['Company size'].astype(str) + ' ' +
    df['Country'].astype(str) + ' ' +
    df['Working days'].astype(str) + ' ' +
    df['Overtime Policy'].astype(str) + ' ' +
    df['Company overview'].astype(str) + ' ' +
    df['Our key skills'].astype(str) + ' ' +
    df["Why you will love working here"].astype(str)
)

# ===== Load embedding đã tính sẵn (phiên bản mở rộng) =====
with open("bert_embeddings_enriched_stopword.pkl", "rb") as f:
    embeddings = pickle.load(f)

# ===== Khởi tạo model BERT và dịch =====
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
translator = Translator()

# ===== Hàm làm sạch dữ liệu người dùng =====
def clean_user_input(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

# ===== Giao diện tìm kiếm =====
st.title("🎯 Tìm Công ty Phù hợp với Mong muốn của Bạn")

query_vi = st.text_input("Nhập từ khóa mô tả mong muốn (bằng tiếng Việt hoặc tiếng Anh)")

if query_vi:
    try:
        query_en = translator.translate(query_vi, src='vi', dest='en').text
    except:
        query_en = query_vi

    st.write(f"🔄 Từ khóa sau khi dịch (nếu cần): **{query_en}**")

    cleaned_query = clean_user_input(query_en)

    if len(cleaned_query.split()) < 0:
        st.warning("Vui lòng nhập mô tả cụ thể hơn (tối thiểu 3 từ có nghĩa).")
    else:
        query_vec = model.encode([cleaned_query])
        sim_scores = cosine_similarity(query_vec, embeddings).flatten()
        top_indices = sim_scores.argsort()[-5:][::-1]

        high_threshold = 0.3
        low_threshold = 0.1

        if sim_scores[top_indices[0]] >= high_threshold:
            st.success("✅ Tìm thấy những công ty thực sự phù hợp nhất:")
        elif sim_scores[top_indices[0]] >= low_threshold:
            st.info("ℹ️ Không có công ty hoàn toàn khớp, nhưng dưới đây là những công ty gần nhất với mong muốn của bạn:")
        else:
            st.warning("⚠️ Không có công ty nào thực sự phù hợp với từ khóa bạn nhập.")
            st.stop()

        sorted_indices = sim_scores.argsort()[::-1]
        top_n = min(3, len(df))
        cols = st.columns(top_n)

        if "selected_idx" not in st.session_state:
            st.session_state.selected_idx = None

        for col, idx in zip(cols, sorted_indices[:top_n]):
            score = sim_scores[idx]
            company_name = df.iloc[idx]['Company Name']
            description = df.iloc[idx]['Full_Description']

            with col:
                with st.container():
                    st.markdown(f"### 🏢 {company_name}")
                    st.caption(f"Similarity: {score:.2f}")

                    if st.button(f"Xem thông tin mô tả", key=f"btn_{idx}"):
                        st.session_state.selected_idx = idx

                    href = df.iloc[idx].get('Href', '')
                    if href and href != 'nan':
                        st.markdown(f"[🔗 Xem chi tiết trên ITViec]({href})", unsafe_allow_html=True)

        if st.session_state.selected_idx is not None:
            idx = st.session_state.selected_idx
            st.write("---")
            with st.expander("📝 Thông tin chi tiết và phân tích thực tế", expanded=True):
                st.write(f"### 🏢 Thông tin chi tiết của {df.iloc[idx]['Company Name']}:")
                st.write(f"**Loại hình:** {df.iloc[idx]['Company Type']}")
                st.write(f"**Ngành nghề:** {df.iloc[idx]['Company industry']}")
                st.write(f"**Quy mô:** {df.iloc[idx]['Company size']}")
                st.write(f"**Tổng quan công ty:** {df.iloc[idx]['Company overview']}")
                st.write(f"**Quốc gia:** {df.iloc[idx]['Country']}")
                st.write(f"**Ngày làm việc:** {df.iloc[idx]['Working days']}")
                st.write(f"**Chính sách OT:** {df.iloc[idx]['Overtime Policy']}")
                st.write(f"**Kỹ năng nổi bật:** {df.iloc[idx]['Our key skills']}")
                st.write(f"**Vì sao nên làm tại đây:** {df.iloc[idx]['Why you will love working here']}")
                st.write(f"**Địa điểm:** {df.iloc[idx]['Location']}")

                # Hỏi thêm người dùng có muốn xem phân tích review không
                if st.checkbox("Tôi muốn xem phân tích từ Review thực tế"):
                 # Load dữ liệu review
                    df_reviews = pd.read_excel("Reviews.xlsx")

                 # Xử lý ngày tháng và tạo cột Năm
                    df_reviews['Cmt_day'] = pd.to_datetime(df_reviews['Cmt_day'], format='%B %Y', errors='coerce')
                    df_reviews['Năm'] = df_reviews['Cmt_day'].dt.year


                    topic_model = BERTopic.load("bertopic_model")

                # Ghép nội dung review
                    df_reviews['Full_Review'] = (
                    df_reviews['Title'].astype(str) + ' ' +
                    df_reviews['What I liked'].astype(str) + ' ' +
                    df_reviews['Suggestions for improvement'].astype(str)
                    )

                # Lọc những dòng có năm hợp lệ
                    df_reviews = df_reviews.dropna(subset=['Năm'])
                    df_reviews['Năm'] = df_reviews['Năm'].astype(int)

                # Lọc review của công ty hiện tại
                    company_id = df.iloc[idx]['id']
                    df_company = df_reviews[df_reviews['id'] == company_id]

                    if df_company.empty:
                        st.info("⚠️ Chưa có dữ liệu review cho công ty này.")
                    else:
                        st.write("### 📊 Tổng hợp nhanh từ review của nhân viên:")

                # Làm sạch dữ liệu để dự đoán
                    df_company['Full_Review_clean'] = df_company['Full_Review'].astype(str).apply(clean_user_input)

                # Load mô hình phân loại Recommend
                    with open("combo_model.pkl", "rb") as f:
                        combo = pickle.load(f)

                    vectorizer = combo["vectorizer"]
                    model = combo["model"]


                    X = vectorizer.transform(df_company['Full_Review_clean'])
                    preds = model.predict(X)
                    total = len(preds)
                    recommend_count = sum(preds)
                    percent = recommend_count / total * 100
                    score = round(percent * 5 / 100, 1)

                    st.write(f"- Tổng số review: **{total}**")
                    st.write(f"- Tỷ lệ Recommend: **{percent:.2f}%**")
                    st.write(f"- Điểm tổng hợp: ⭐️ **{score}/5**")

                    if percent >= 60:
                        st.success("✅ Đánh giá tổng quan: Có thể cân nhắc làm việc tại đây.")
                    elif percent <= 30:
                        st.error("⚠️ Đánh giá tổng quan: Nhiều review không khuyến nghị, cần thận trọng.")
                    else:
                        st.info("ℹ️ Đánh giá tổng quan: Ý kiến trái chiều, nên tìm hiểu thêm.")

                    st.divider()

                # Phân tích chủ đề từng năm
                    years = sorted(df_company['Năm'].unique(), reverse=True)[:3]
                    years = sorted(years)

                    for year in years:
                        st.write(f"## 🔥 Top 5 chủ đề được bàn luận nhiều nhất năm {year} của công ty **{df.iloc[idx]['Company Name']}**")

                        df_year = df_company[df_company['Năm'] == year]
                        topics, _ = topic_model.transform(df_year['Full_Review'].tolist())
                        df_year['Topic'] = topics
                        df_year = df_year[df_year['Topic'] != -1]

                        if df_year.empty:
                            st.warning("Không có dữ liệu chủ đề rõ ràng cho năm này.")
                        else:
                            top_topic_counts = df_year['Topic'].value_counts().head(5).reset_index()
                            top_topic_counts.columns = ['Topic', 'Số lượng']

                            def extract_keywords(topic_id):
                                keywords = topic_model.get_topic(topic_id)
                                if isinstance(keywords, list):
                                    return " ".join([word for word, _ in keywords[:20]])
                                return ""

                            top_topic_counts['Từ khóa cho Wordcloud'] = top_topic_counts['Topic'].apply(extract_keywords)

                            for _, row in top_topic_counts.iterrows():
                                st.write(f"### 🌥️ Chủ đề {row['Topic']} - {row['Số lượng']} lượt nhắc đến")

                                wordcloud = WordCloud(width=400, height=200, background_color="white").generate(row['Từ khóa cho Wordcloud'])

                                fig, ax = plt.subplots()
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis("off")
                                st.pyplot(fig)

                    st.divider()



