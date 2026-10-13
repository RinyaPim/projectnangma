from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


st.set_page_config(page_title="Customer Marketing Analytics", page_icon="📊", layout="wide")

REQUIRED_COLUMNS = {
    "customer_id",
    "age",
    "monthly_income",
    "months_customer",
    "avg_monthly_spend",
    "orders_6m",
    "website_visits_month",
    "discount_usage_percent",
    "membership_tier",
    "contact_channel",
    "campaign_response",
}
CATEGORY_MAPPINGS = {
    "membership_tier": {"basic": "Basic", "silver": "Silver", "gold": "Gold"},
    "contact_channel": {"sms": "SMS", "app": "App", "email": "Email"},
}


@st.cache_data
def load_workbook(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    with pd.ExcelFile(BytesIO(file_bytes)) as workbook:
        return {sheet: workbook.parse(sheet) for sheet in workbook.sheet_names}


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"ไฟล์ไม่มีคอลัมน์ที่จำเป็น: {', '.join(sorted(missing_columns))}")

    cleaned = data.drop_duplicates().copy()
    cleaned["campaign_response"] = pd.to_numeric(
        cleaned["campaign_response"], errors="coerce"
    )
    cleaned = cleaned[cleaned["campaign_response"].isin([0, 1])].copy()
    cleaned["campaign_response"] = cleaned["campaign_response"].astype(int)

    for column, mapping in CATEGORY_MAPPINGS.items():
        normalized = cleaned[column].astype("string").str.strip().str.casefold()
        cleaned[column] = normalized.map(mapping).fillna("ไม่ระบุ")

    return cleaned


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


st.title("Customer Marketing Analytics")
st.caption("สำรวจข้อมูล ทำนายการตอบรับแคมเปญ และแบ่งกลุ่มลูกค้า")

default_file = Path(__file__).with_name("marketing.xlsx")
uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ Excel", type=["xlsx"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
elif default_file.exists():
    file_bytes = default_file.read_bytes()
    st.sidebar.success("กำลังใช้ marketing.xlsx จากโฟลเดอร์โปรเจกต์")
else:
    st.info("อัปโหลดไฟล์ Excel ที่มีข้อมูลลูกค้าเพื่อเริ่มใช้งาน")
    st.stop()

try:
    sheets = load_workbook(file_bytes)
except (ValueError, OSError, ImportError) as error:
    st.error(f"อ่านไฟล์ Excel ไม่สำเร็จ: {error}")
    st.stop()

sheet_options = list(sheets)
default_sheet_index = sheet_options.index("Raw_Data") if "Raw_Data" in sheet_options else 0
selected_sheet = st.sidebar.selectbox(
    "เลือกชีตข้อมูล", sheet_options, index=default_sheet_index
)

try:
    data = clean_data(sheets[selected_sheet])
except ValueError as error:
    st.error(str(error))
    st.stop()

if data.empty:
    st.error("ไม่พบแถวข้อมูลที่มีค่า campaign_response เป็น 0 หรือ 1")
    st.stop()

features = data.drop(columns=["campaign_response", "customer_id"])
target = data["campaign_response"]
if target.nunique() < 2:
    st.error("ต้องมีทั้ง campaign_response 0 และ 1 เพื่อเปรียบเทียบโมเดล")
    st.stop()

overview_tab, prediction_tab, clustering_tab = st.tabs(
    ["ภาพรวมข้อมูล", "เปรียบเทียบโมเดลทำนาย", "แบ่งกลุ่มลูกค้า"]
)

with overview_tab:
    response_rate = target.mean() * 100
    metric_columns = st.columns(4)
    metric_columns[0].metric("จำนวนลูกค้า", f"{len(data):,}")
    metric_columns[1].metric("จำนวนคุณลักษณะ", len(features.columns))
    metric_columns[2].metric("อัตราตอบรับ", f"{response_rate:.1f}%")
    metric_columns[3].metric("ข้อมูลซ้ำที่ลบ", f"{sheets[selected_sheet].duplicated().sum():,}")

    st.subheader("สัดส่วนการตอบรับ")
    response_counts = target.value_counts().reindex([0, 1], fill_value=0)
    response_counts.index = ["ไม่ตอบรับ (0)", "ตอบรับ (1)"]
    st.bar_chart(response_counts)

    st.subheader("ตัวอย่างข้อมูล")
    st.dataframe(data.head(100), use_container_width=True)

    missing_values = data.isna().sum().rename("ค่าว่าง").to_frame()
    st.subheader("จำนวนค่าว่าง")
    st.dataframe(missing_values, use_container_width=True)

with prediction_tab:
    st.subheader("ประเมิน Classification")
    test_size = st.slider("สัดส่วนข้อมูลทดสอบ (%)", 10, 40, 20, 5) / 100
    if st.button("ฝึกและเปรียบเทียบโมเดล", key="run_classification"):
        class_counts = target.value_counts()
        if class_counts.min() < 2:
            st.error("แต่ละคลาสต้องมีอย่างน้อย 2 ตัวอย่างเพื่อแบ่ง train/test แบบ stratify")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=42,
                stratify=target,
            )
            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Random Forest": RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                ),
                "SVM (RBF)": SVC(class_weight="balanced"),
            }
            results = []
            matrices = {}
            preprocessor = build_preprocessor(features)

            for model_name, classifier in models.items():
                pipeline = Pipeline(
                    [
                        ("preprocessing", clone(preprocessor)),
                        ("classifier", classifier),
                    ]
                )
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                results.append(
                    {
                        "โมเดล": model_name,
                        "Accuracy": accuracy_score(y_test, predictions),
                        "Precision (ตอบรับ)": precision_score(
                            y_test, predictions, zero_division=0
                        ),
                        "Recall (ตอบรับ)": recall_score(
                            y_test, predictions, zero_division=0
                        ),
                        "F1 (ตอบรับ)": f1_score(
                            y_test, predictions, zero_division=0
                        ),
                    }
                )
                matrices[model_name] = confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                )

            results_frame = pd.DataFrame(results).set_index("โมเดล")
            st.dataframe(
                results_frame.sort_values("Accuracy", ascending=False).style.format(
                    "{:.3f}"
                ),
                use_container_width=True,
            )
            st.subheader("Confusion Matrix")
            for model_name, matrix in matrices.items():
                st.markdown(f"**{model_name}**")
                st.dataframe(
                    pd.DataFrame(
                        matrix,
                        index=["จริง: ไม่ตอบรับ", "จริง: ตอบรับ"],
                        columns=["ทำนาย: ไม่ตอบรับ", "ทำนาย: ตอบรับ"],
                    ),
                    use_container_width=True,
                )
            st.caption("ผลขึ้นกับการแบ่งข้อมูล train/test และไม่ได้รับประกันว่าจะสะท้อนผลกับลูกค้าในอนาคต")

with clustering_tab:
    st.subheader("เปรียบเทียบ Clustering")
    n_clusters = st.slider("จำนวนกลุ่ม", 2, 6, 2, key="cluster_count")
    if st.button("จัดกลุ่มและเปรียบเทียบ", key="run_clustering"):
        preprocessor = build_preprocessor(features)
        transformed = preprocessor.fit_transform(features)
        if sparse.issparse(transformed):
            transformed = transformed.toarray()

        labels_by_model = {
            "K-Means": KMeans(
                n_clusters=n_clusters, n_init=20, random_state=42
            ).fit_predict(transformed),
            "Agglomerative": AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward"
            ).fit_predict(transformed),
        }
        scores = {
            name: silhouette_score(
                transformed,
                labels,
                sample_size=min(1000, len(transformed)),
                random_state=42,
            )
            for name, labels in labels_by_model.items()
        }
        score_frame = pd.DataFrame(
            [{"โมเดล": name, "Silhouette Score": score} for name, score in scores.items()]
        ).sort_values("Silhouette Score", ascending=False)
        st.dataframe(
            score_frame.set_index("โมเดล").style.format("{:.3f}"),
            use_container_width=True,
        )
        for model_name, cluster_labels in labels_by_model.items():
            profile = pd.DataFrame(
                {
                    "กลุ่ม": cluster_labels,
                    "campaign_response": target.to_numpy(),
                }
            ).groupby("กลุ่ม").agg(
                ลูกค้า=("campaign_response", "size"),
                อัตราตอบรับ=("campaign_response", "mean"),
            )
            profile["อัตราตอบรับ"] *= 100
            st.markdown(f"**โปรไฟล์กลุ่ม: {model_name}**")
            st.dataframe(
                profile.style.format({"อัตราตอบรับ": "{:.1f}%"}),
                use_container_width=True,
            )
        if score_frame["Silhouette Score"].max() < 0.25:
            st.warning("Silhouette Score ค่อนข้างต่ำ กลุ่มอาจทับซ้อนกัน ควรใช้เป็นผลสำรวจเบื้องต้น")
        st.caption("campaign_response ใช้แสดงอัตราตอบรับหลังจัดกลุ่มเท่านั้น ไม่ได้ใช้สร้างกลุ่ม")
