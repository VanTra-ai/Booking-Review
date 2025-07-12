import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import unicodedata
from underthesea import word_tokenize
import os
import pickle
import re  # Thêm thư viện re để xử lý regex

# =============================================================================
# KHỐI TIỀN XỬ LÝ
# =============================================================================

# Định nghĩa stopwords tiếng Việt
vietnamese_stopwords = [
    "là", "và", "của", "có", "không", "được", "trong", "đã", "cho", "với", "này", "đó",
    "các", "để", "những", "một", "rất", "cũng", "khi", "như", "về", "từ", "tại", "tới",
    "thì", "vì", "nên", "lúc", "mà", "còn", "bởi", "theo", "vào", "ra", "lên", "xuống",
    "tôi", "tầm", "tạm", "nơi", "thư", "mọi", "đâu", "mình", "ai", "khi", "sẽ", "đều",
    "chỉ", "mới", "thật", "quá", "đến", "chứ", "nhất", "đủ", "chỉ", "đang", "trước", "sau",
    "đây", "đấy", "thì", "đó", "này", "nhiều", "ít", "hơn", "thôi", "tuy", "hay", "bởi",
    "ở", "đi", "làm", "nếu", "tuy", "hoặc", "đều", "cứ", "đã", "rồi", "xong", "phòng", "khách",
    "tam", "nhung", "nhieu", "duoc", "minh", "khong", "ngay", "cac", "nha", "rat", "co",
    "khach san", "can", "tai", "cua", "tot", "nay", "chi", "tren", "day", "hon",
    "yên tĩnh", "đậu xe", "thoải mái", "nhiệt tình", "thân thiện", "rất ok", "hẹn gặp lại",
    "vị trí", "gần biển", "mới xây", "trang thiết bị", "còn mới", "sạch sẽ", "bãi đậu xe",
    "ô tô", "để dọc đường", "ít cây", "khá nắng", "ngu ngốc", "tệ", "xấu", "kinh khủng"
]


# Hàm tiền xử lý văn bản (giống hệt trong notebook)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Chuẩn hóa Unicode và chuyển thành chữ thường
    text = unicodedata.normalize('NFC', text.lower())
    # Tách từ tiếng Việt
    tokens = word_tokenize(text, format="text")
    tokens_list = tokens.split()
    filtered_tokens = [token for token in tokens_list if token not in vietnamese_stopwords and len(token) > 1]
    return " ".join(filtered_tokens)


# =============================================================================
# KHỐI ĐỊNH NGHĨA MÔ HÌNH
# =============================================================================

# NUM_CLASSES và label_map này là dành cho Mô hình 2 (kết hợp các đặc trưng)
NUM_CLASSES = 6
label_map = {0: '5', 1: '6', 2: '7', 3: '8', 4: '9', 5: '10'}

# label_map riêng cho Mô hình 1 (đánh giá 1-10)
# Chỉ số 0 sẽ ánh xạ thành điểm '1', chỉ số 9 thành điểm '10'
label_map_model1 = {i: str(i + 1) for i in range(10)}


class BertTextClassifier(nn.Module):
    """
    Mô hình phân loại chỉ dựa trên BERT cho văn bản. (Mô hình 1)
    ĐƯỢC ĐIỀU CHỈNH ĐỂ KHỚP VỚI KIẾN TRÚC MÔ HÌNH ĐÃ LƯU (HIDDEN: 50, OUTPUT: 10)
    """

    def __init__(self, freeze_bert=False):  # Bỏ num_classes ở đây để sử dụng các kích thước cố định từ checkpoint
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('trituenhantaoio/bert-base-vietnamese-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 50),  # Đã thay đổi từ 128 thành 50 (dựa trên lỗi size mismatch)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 10)  # Đã thay đổi từ 128 (đầu vào) và NUM_CLASSES (đầu ra) thành 50 (đầu vào) và 10 (đầu ra)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


class BertCombinedClassifier(nn.Module):
    """
    Mô hình phân loại kết hợp BERT cho văn bản và các lớp Linear cho đặc trưng số. (Mô hình 2)
    """

    def __init__(self, freeze_bert=False, num_classes=NUM_CLASSES):
        super(BertCombinedClassifier, self).__init__()

        HOTEL_INFO_DIM = 7
        REVIEW_INFO_DIM = 3
        HOTEL_EMBED_DIM = 32
        REVIEW_EMBED_DIM = 16

        self.bert = BertModel.from_pretrained('trituenhantaoio/bert-base-vietnamese-uncased')
        self.hotel_info_embedder = nn.Linear(HOTEL_INFO_DIM, HOTEL_EMBED_DIM)
        self.review_info_embedder = nn.Linear(REVIEW_INFO_DIM, REVIEW_EMBED_DIM)

        D_in = 768 + HOTEL_EMBED_DIM + REVIEW_EMBED_DIM
        H = 128
        D_out = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H, D_out)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, hotel_info, review_info, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        hotel_embedding = self.hotel_info_embedder(hotel_info)
        review_embedding = self.review_info_embedder(review_info)
        combined_features = torch.cat([last_hidden_state_cls, hotel_embedding, review_embedding], dim=1)
        logits = self.classifier(combined_features)
        return logits


# =============================================================================
# KHỐI CÀI ĐẶT STREAMLIT VÀ LOGIC DỰ ĐOÁN
# =============================================================================

st.set_page_config(page_title="Booking Review Sentiment", layout="centered")

st.title("Ứng dụng dự đoán đánh giá đặt phòng")
st.markdown("""
    Chọn một trong hai mô hình để dự đoán điểm đánh giá dựa trên các đặc trưng khác nhau.
""")

# --- Thiết lập các hằng số đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_RESOURCES_DIR = os.path.join(BASE_DIR, "app")  # Đảm bảo thư mục 'app' nằm cùng cấp với app.py

MODEL1_PATH = os.path.join(APP_RESOURCES_DIR, "bert_classifier_booking_review.pth")
MODEL2_PATH = os.path.join(APP_RESOURCES_DIR, "MSSA_classifier_booking_review.pth")

TOKENIZER_DIR = os.path.join(APP_RESOURCES_DIR, "booking_review_tokenizer")
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALER_HOTEL_PATH = os.path.join(APP_RESOURCES_DIR, "scaler_hotel.pkl")
SCALER_REVIEW_PATH = os.path.join(APP_RESOURCES_DIR, "scaler_review.pkl")
MAPPING_DICTS_PATH = os.path.join(APP_RESOURCES_DIR, "mapping_dicts.pkl")


# Sử dụng st.cache_resource để tải tất cả tài nguyên một lần duy nhất
@st.cache_resource
def load_all_resources():
    st.info("Đang tải tài nguyên... (Chỉ lần đầu tiên)")

    tokenizer_loaded = None
    try:
        if os.path.exists(TOKENIZER_DIR):
            tokenizer_loaded = BertTokenizer.from_pretrained(TOKENIZER_DIR)
            st.success(f"Đã tải tokenizer từ thư mục local: {TOKENIZER_DIR}.")
        else:
            st.warning(f"Không tìm thấy thư mục tokenizer local tại: {TOKENIZER_DIR}. Đang thử tải từ Hugging Face...")
            tokenizer_loaded = BertTokenizer.from_pretrained('trituenhantaoio/bert-base-vietnamese-uncased')
            st.warning("Đã tải tokenizer từ Hugging Face. Có thể không khớp chính xác với tokenizer bạn đã huấn luyện.")
    except Exception as e:
        st.error(f"Lỗi khi tải tokenizer: {e}. Đang thử tải từ Hugging Face...")
        tokenizer_loaded = BertTokenizer.from_pretrained('trituenhantaoio/bert-base-vietnamese-uncased')
        st.warning("Đã tải tokenizer từ Hugging Face. Có thể không khớp chính xác với tokenizer bạn đã huấn luyện.")

    model1_loaded = None
    try:
        # Khởi tạo BertTextClassifier mà không truyền num_classes vì đã hardcode trong __init__
        model1_loaded = BertTextClassifier()
        if os.path.exists(MODEL1_PATH):
            model1_loaded.load_state_dict(torch.load(MODEL1_PATH, map_location=DEVICE))
            model1_loaded.to(DEVICE)
            model1_loaded.eval()
            st.success(f"Mô hình 1 (Chỉ văn bản) đã được tải thành công từ: {MODEL1_PATH}!")
        else:
            st.warning(f"Không tìm thấy file mô hình 1 tại: {MODEL1_PATH}. Mô hình 1 sẽ không hoạt động.")
            model1_loaded = None
    except RuntimeError as e:  # Bắt lỗi RuntimeError cụ thể cho size mismatch
        st.error(
            f"Lỗi khi tải mô hình 1 (kiến trúc không khớp): {e}. Vui lòng kiểm tra lại định nghĩa BertTextClassifier trong code so với mô hình đã lưu.")
        model1_loaded = None
    except Exception as e:
        st.error(f"Lỗi không xác định khi tải mô hình 1: {e}")
        model1_loaded = None

    model2_loaded = None
    try:
        model2_loaded = BertCombinedClassifier(num_classes=NUM_CLASSES)  # NUM_CLASSES vẫn là 6 cho mô hình này
        if os.path.exists(MODEL2_PATH):
            model2_loaded.load_state_dict(torch.load(MODEL2_PATH, map_location=DEVICE))
            model2_loaded.to(DEVICE)
            model2_loaded.eval()
            st.success(f"Mô hình 2 (Kết hợp các đặc trưng) đã được tải thành công từ: {MODEL2_PATH}!")
        else:
            st.warning(f"Không tìm thấy file mô hình 2 tại: {MODEL2_PATH}. Mô hình 2 sẽ không hoạt động.")
            model2_loaded = None
    except RuntimeError as e:  # Bắt lỗi RuntimeError cụ thể cho size mismatch
        st.error(
            f"Lỗi khi tải mô hình 2 (kiến trúc không khớp): {e}. Vui lòng kiểm tra lại định nghĩa BertCombinedClassifier trong code so với mô hình đã lưu.")
        model2_loaded = None
    except Exception as e:
        st.error(f"Lỗi không xác định khi tải mô hình 2: {e}")
        model2_loaded = None

    scaler_hotel_loaded = None
    scaler_review_loaded = None
    try:
        if os.path.exists(SCALER_HOTEL_PATH) and os.path.exists(SCALER_REVIEW_PATH):
            with open(SCALER_HOTEL_PATH, 'rb') as f:
                scaler_hotel_loaded = pickle.load(f)
            with open(SCALER_REVIEW_PATH, 'rb') as f:
                scaler_review_loaded = pickle.load(f)
            st.success("Đã tải các StandardScaler thành công (cho Mô hình 2).")
        else:
            st.warning(
                f"Không tìm thấy các file StandardScaler tại: {SCALER_HOTEL_PATH} hoặc {SCALER_REVIEW_PATH}. Mô hình 2 có thể không hoạt động đúng nếu thiếu các file này.")

    except Exception as e:
        st.warning(f"Không thể tải StandardScaler: {e}. Mô hình 2 có thể không hoạt động đúng nếu thiếu các file này.")
        scaler_hotel_loaded = None
        scaler_review_loaded = None

    mapping_dicts_loaded = None
    try:
        if os.path.exists(MAPPING_DICTS_PATH):
            with open(MAPPING_DICTS_PATH, 'rb') as f:
                mapping_dicts_loaded = pickle.load(f)
            st.success("Đã tải các dictionary ánh xạ thành công (cho Mô hình 2).")
        else:
            st.warning(
                f"Không tìm thấy file dictionary ánh xạ tại: {MAPPING_DICTS_PATH}. Mô hình 2 có thể không hoạt động đúng nếu thiếu file này.")
    except Exception as e:
        st.warning(
            f"Không thể tải các dictionary ánh xạ: {e}. Mô hình 2 có thể không hoạt động đúng nếu thiếu file này.")
        mapping_dicts_loaded = None

    return tokenizer_loaded, model1_loaded, model2_loaded, scaler_hotel_loaded, scaler_review_loaded, mapping_dicts_loaded


# Tải tất cả tài nguyên khi ứng dụng khởi chạy
tokenizer, model1, model2, scaler_hotel, scaler_review, mapping_dicts = load_all_resources()

# Kiểm tra nếu tokenizer không tải được thì dừng ứng dụng
if tokenizer is None:
    st.error("Không thể tải Tokenizer. Vui lòng kiểm tra đường dẫn và file. Ứng dụng không thể hoạt động.")
    st.stop()

# Tách các dictionary ánh xạ (nếu có tải được)
room_type_dict = {}
group_type_dict = {}

if mapping_dicts:
    room_type_dict = mapping_dicts.get('room_type_dict', {})
    group_type_dict = mapping_dicts.get('group_type_dict', {})

    room_type_options = {v: k for k, v in room_type_dict.items()} if room_type_dict else {"0": "Không có dữ liệu"}
    group_type_options = {v: k for k, v in group_type_dict.items()} if group_type_dict else {"0": "Không có dữ liệu"}

# --- Lựa chọn mô hình trên Sidebar ---
st.sidebar.header("Chọn Mô hình")
model_choice = st.sidebar.radio(
    "Bạn muốn sử dụng mô hình nào để dự đoán?",
    ("Mô hình 1: Chỉ văn bản", "Mô hình 2: Kết hợp các đặc trưng")
)

# --- Giao diện người dùng cho đầu vào văn bản chung ---
st.header("Nhập thông tin đánh giá:")

review_text = st.text_area("Bình luận về trải nghiệm (Review)", height=100, key="review_text_input")
positive_comment_text = st.text_area("Bình luận tích cực thêm (Comment Positive, không bắt buộc)", height=100,
                                     key="positive_comment_text_input")

combined_text_input = f"{review_text} {positive_comment_text}".strip()

# ====================================================
# Hiển thị Inputs bổ sung dựa trên lựa chọn mô hình
# ====================================================
hotel_info_raw = []
review_info_raw = []

if model_choice == "Mô hình 2: Kết hợp các đặc trưng":
    st.subheader("Thông tin Khách sạn (Hotel Info - 7 đặc trưng):")
    if not (scaler_hotel and scaler_review and mapping_dicts):
        st.warning("Các file Scaler hoặc Mappings không tải được. Mô hình 2 có thể không hoạt động chính xác.")
        st.info(
            f"Vui lòng đảm bảo các file 'scaler_hotel.pkl', 'scaler_review.pkl', 'mapping_dicts.pkl' nằm trong thư mục: {APP_RESOURCES_DIR}")

    # Các trường nhập liệu cho Hotel Info
    sorted_hotel_info_keys_for_model = [
        "amenities", "cleanliness", "comfort", "facilities", "location",
        "service_staff", "value_for_money"
    ]

    hotel_info_inputs = {}
    for key in sorted_hotel_info_keys_for_model:
        label = key.replace('_', ' ').title()
        hotel_info_inputs[key] = st.slider(f"Điểm {label}", min_value=1.0, max_value=10.0, value=8.0, step=0.1,
                                           key=f"hotel_slider_{key}")

    hotel_info_raw = [hotel_info_inputs[key] for key in sorted_hotel_info_keys_for_model]

    st.subheader("Thông tin Đánh giá khác (Review Info - 3 đặc trưng):")

    # --- Đã chỉnh sửa: Thiết lập trực tiếp Thời gian lưu trú từ 1 đến 31 đêm ---
    stay_duration_display_options = [f"{i} đêm" for i in range(1, 32)]  # Tạo list từ "1 đêm" đến "31 đêm"

    selected_stay_duration_display = st.selectbox(
        "Thời gian lưu trú:",
        options=stay_duration_display_options,
        index=0,  # Mặc định là "1 đêm"
        key="stay_duration_select"
    )

    # Trích xuất số đêm từ chuỗi đã chọn (ví dụ: "5 đêm" -> 5)
    # Sử dụng regex để tìm số đầu tiên trong chuỗi
    match = re.search(r'(\d+)', selected_stay_duration_display)
    if match:
        encoded_stay_duration = int(match.group(1))
    else:
        encoded_stay_duration = 1  # Mặc định là 1 đêm nếu không tìm thấy số

    if not room_type_options or not group_type_options:
        st.warning(
            "Không tìm thấy dữ liệu ánh xạ đầy đủ cho các trường danh mục (Loại phòng, Loại nhóm khách). Các giá trị mặc định sẽ được sử dụng."
        )
        selected_room_type_display = "0"
        selected_group_type_display = "0"

        encoded_room_type = room_type_dict.get(selected_room_type_display, 0)
        encoded_group_type = group_type_dict.get(selected_group_type_display, 0)

    else:
        display_room_type_options = list(room_type_options.values())
        display_group_type_options = list(group_type_options.values())

        default_room_type_index = 0
        if "Phòng đôi" in display_room_type_options:
            default_room_type_index = display_room_type_options.index("Phòng đôi")

        selected_room_type_display = st.selectbox(
            "Loại phòng:",
            options=display_room_type_options,
            index=default_room_type_index if display_room_type_options else 0,
            key="room_type_select"
        )
        selected_group_type_display = st.selectbox(
            "Loại nhóm khách:",
            options=display_group_type_options,
            index=0,
            key="group_type_select"
        )

        reverse_room_type_dict = {v: k for k, v in room_type_dict.items()}
        reverse_group_type_dict = {v: k for k, v in group_type_dict.items()}

        encoded_room_type = reverse_room_type_dict.get(selected_room_type_display, 0)
        encoded_group_type = reverse_group_type_dict.get(selected_group_type_display, 0)

    review_info_raw = [encoded_room_type, encoded_stay_duration, encoded_group_type]

# --- Nút dự đoán ---
submit_button = st.button("Dự đoán điểm đánh giá", key="predict_button")

if submit_button:
    if not combined_text_input:
        st.warning("Vui lòng nhập ít nhất một bình luận để dự đoán.")
    else:
        with st.spinner("Đang xử lý và dự đoán..."):
            try:
                processed_text = preprocess_text(combined_text_input)

                if not processed_text.strip():
                    st.warning("Bình luận quá ngắn hoặc không có ý nghĩa sau tiền xử lý. Vui lòng nhập thêm.")
                    st.stop()

                encoded_sent = tokenizer.encode_plus(
                    text=processed_text,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                input_ids = encoded_sent.get('input_ids').to(DEVICE)
                attention_mask = encoded_sent.get('attention_mask').to(DEVICE)

                logits = None
                if model_choice == "Mô hình 1: Chỉ văn bản":
                    if model1 is None:
                        st.error("Mô hình 1 không khả dụng. Vui lòng kiểm tra file mô hình.")
                        st.stop()
                    with torch.no_grad():
                        logits = model1(input_ids, attention_mask)

                    if logits is not None:
                        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
                        prediction_idx = torch.argmax(logits, dim=1).item()

                        # Sử dụng label_map_model1 mới để ánh xạ chỉ số thành điểm 1-10
                        predicted_label_model1 = label_map_model1[prediction_idx]

                        st.subheader("Kết quả dự đoán (Mô hình 1):")
                        st.metric(label="Điểm đánh giá dự đoán", value=predicted_label_model1)  # Hiện thị điểm từ 1-10

                        st.markdown("---")
                        st.subheader("Điểm tin cậy cho mỗi lớp (từ 1-10):")
                        # Sử dụng label_map_model1 để hiển thị độ tin cậy
                        confidence_data = {label_map_model1[i]: f"{prob * 100:.2f}%" for i, prob in enumerate(probs)}
                        st.json(confidence_data)

                else:  # Mô hình 2: Kết hợp các đặc trưng
                    if model2 is None or scaler_hotel is None or scaler_review is None or not mapping_dicts:
                        st.error(
                            "Mô hình 2 hoặc các StandardScaler/Mappings không khả dụng. Vui lòng kiểm tra các file."
                        )
                        st.info(
                            f"Đảm bảo các file scaler_hotel.pkl, scaler_review.pkl, mapping_dicts.pkl nằm trong thư mục: {APP_RESOURCES_DIR}")
                        st.stop()

                    hotel_info_scaled = scaler_hotel.transform(np.array(hotel_info_raw).reshape(1, -1))
                    review_info_scaled = scaler_review.transform(np.array(review_info_raw).reshape(1, -1))

                    hotel_info_tensor = torch.tensor(hotel_info_scaled, dtype=torch.float32).to(DEVICE)
                    review_info_tensor = torch.tensor(review_info_scaled, dtype=torch.float32).to(DEVICE)

                    with torch.no_grad():
                        logits = model2(hotel_info_tensor, review_info_tensor, input_ids, attention_mask)

                    if logits is not None:
                        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
                        prediction_idx = torch.argmax(logits, dim=1).item()
                        predicted_label = label_map[prediction_idx]  # Sử dụng label_map 6 lớp cho Mô hình 2

                        st.subheader("Kết quả dự đoán (Mô hình 2):")
                        st.metric(label="Điểm đánh giá dự đoán", value=predicted_label)

                        st.markdown("---")
                        st.subheader("Điểm tin cậy cho mỗi điểm:")
                        confidence_data = {label_map[i]: f"{prob * 100:.2f}%" for i, prob in enumerate(probs)}
                        st.json(confidence_data)

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
                st.info("Vui lòng kiểm tra lại đầu vào hoặc cấu hình mô hình/tokenizer/scalers.")

st.markdown("---")
st.caption("Ứng dụng demo bởi nhóm AI Research. Sử dụng các mô hình BERT cho phân tích cảm xúc.")