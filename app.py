import io
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DEFAULT_DATA = BASE_DIR / "suppliers.csv"
REQUIRED_COLUMNS = [
    "supplier", "category", "regions", "products", "certifications",
    "unit_price", "quality_score", "on_time_rate", "compliance_score", "lead_days",
]

st.set_page_config(page_title="供方智选 | 供应商评估", page_icon="📦", layout="wide")
st.title("供方智选：供应商匹配与评估")
st.caption("基于需求文本检索候选供应商，并按采购指标生成可解释的综合排序。演示数据均为虚构数据。")


def read_suppliers(uploaded_file):
    if uploaded_file is None:
        return pd.read_csv(DEFAULT_DATA)
    return pd.read_csv(uploaded_file)


def validate_data(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error("数据缺少必需字段：" + "、".join(missing))
        st.stop()
    df = df.copy()
    numeric = ["unit_price", "quality_score", "on_time_rate", "compliance_score", "lead_days"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=numeric)


def rank_suppliers(df, query, weights):
    corpus = (df["category"].fillna("") + " " + df["products"].fillna("") + " " + df["certifications"].fillna("")).tolist()
    if query.strip():
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        matrix = vectorizer.fit_transform(corpus + [query])
        relevance = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    else:
        relevance = [0.0] * len(df)

    out = df.copy()
    out["需求相关度"] = relevance * 100
    low, high = out["unit_price"].min(), out["unit_price"].max()
    out["价格得分"] = 100.0 if high == low else (high - out["unit_price"]) / (high - low) * 100
    out["交付得分"] = out["on_time_rate"].clip(0, 100)
    out["质量得分"] = out["quality_score"].clip(0, 100)
    out["合规得分"] = out["compliance_score"].clip(0, 100)
    max_lead = max(float(out["lead_days"].max()), 1.0)
    out["交期得分"] = (1 - (out["lead_days"] - out["lead_days"].min()) / max(max_lead - out["lead_days"].min(), 1)) * 100
    score_cols = {
        "需求相关度": weights["需求匹配"], "价格得分": weights["价格"],
        "质量得分": weights["质量"], "交付得分": weights["交付"],
        "合规得分": weights["合规"], "交期得分": weights["交期"],
    }
    total = sum(score_cols.values())
    out["综合得分"] = sum(out[col] * weight for col, weight in score_cols.items()) / total
    out["风险提示"] = out.apply(lambda r: "；".join(
        (["合规分偏低"] if r["compliance_score"] < 70 else []) +
        (["准时交付率偏低"] if r["on_time_rate"] < 85 else []) +
        (["质量分偏低"] if r["quality_score"] < 75 else [])
    ) or "暂无明显风险", axis=1)
    return out.sort_values("综合得分", ascending=False).reset_index(drop=True)


with st.sidebar:
    st.header("评估设置")
    uploaded = st.file_uploader("上传供应商 CSV", type=["csv"])
    st.caption("字段模板见项目目录 suppliers.csv。请勿上传未经授权的真实供应商或个人信息。")
    st.subheader("评分权重")
    weights = {
        "需求匹配": st.slider("需求匹配", 0, 100, 30),
        "价格": st.slider("价格", 0, 100, 20),
        "质量": st.slider("质量", 0, 100, 20),
        "交付": st.slider("交付", 0, 100, 15),
        "合规": st.slider("合规", 0, 100, 10),
        "交期": st.slider("交期", 0, 100, 5),
    }

try:
    suppliers = validate_data(read_suppliers(uploaded))
except Exception as exc:
    st.error(f"无法读取供应商数据：{exc}")
    st.stop()

left, right = st.columns([2, 1])
with left:
    query = st.text_input("采购需求", placeholder="例如：深圳办公家具，需环保认证、按时交付")
with right:
    categories = ["全部品类"] + sorted(suppliers["category"].dropna().unique().tolist())
    selected_category = st.selectbox("品类筛选", categories)

if selected_category != "全部品类":
    suppliers = suppliers[suppliers["category"] == selected_category].copy()
if suppliers.empty:
    st.warning("当前筛选条件下没有供应商。")
    st.stop()

ranked = rank_suppliers(suppliers, query, weights)
col1, col2, col3 = st.columns(3)
col1.metric("候选供应商", len(ranked))
col2.metric("最高综合得分", f"{ranked.iloc[0]['综合得分']:.1f} / 100")
col3.metric("权重总和", f"{sum(weights.values())}")

st.subheader("供应商排序")
display = ranked[["supplier", "category", "regions", "综合得分", "需求相关度", "unit_price", "quality_score", "on_time_rate", "compliance_score", "lead_days", "风险提示"]].rename(columns={
    "supplier": "供应商", "category": "品类", "regions": "服务区域", "unit_price": "参考单价",
    "quality_score": "质量分", "on_time_rate": "准时交付率", "compliance_score": "合规分", "lead_days": "交期（天）",
})
st.dataframe(display.style.format({"综合得分": "{:.1f}", "需求相关度": "{:.1f}", "参考单价": "{:.2f}", "准时交付率": "{:.1f}%"}), use_container_width=True, hide_index=True)

best = ranked.iloc[0]
st.subheader(f"推荐候选：{best['supplier']}")
st.write(f"**综合得分 {best['综合得分']:.1f} 分。** 该排序按当前需求文本、品类筛选和权重计算，推荐结果需由采购人员结合实际报价与资质复核。")
st.write(f"**主要指标：** 参考单价 {best['unit_price']:.2f}；质量分 {best['quality_score']:.0f}；准时交付率 {best['on_time_rate']:.0f}%；合规分 {best['compliance_score']:.0f}；交期 {best['lead_days']:.0f} 天。")
st.write(f"**风险提示：** {best['风险提示']}。")

with st.expander("评分方法与使用说明"):
    st.markdown("""
    - 需求匹配：对品类、产品描述和资质文本做字符 n-gram TF-IDF 相似度检索，适合中文短文本的基础匹配演示。
    - 价格：在当前候选集合内按相对价格归一化，价格越低得分越高。
    - 质量、交付、合规：使用表格中的 0–100 分或百分比。
    - 交期：在当前候选集合内按相对交期归一化，天数越少得分越高。
    - 综合分：各指标按左侧权重加权平均。它是辅助决策的规则评分，不代表模型训练出的真实中标概率。
    """)

csv_bytes = ranked.to_csv(index=False).encode("utf-8-sig")
st.download_button("下载评估结果 CSV", data=csv_bytes, file_name="supplier_ranking.csv", mime="text/csv")
