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
TEXT_COLUMNS = ["supplier", "category", "regions", "products", "certifications"]
SCORE_COLUMNS = ["quality_score", "on_time_rate", "compliance_score"]

st.set_page_config(page_title="供方智选 | 供应商评估", page_icon="📦", layout="wide")
st.title("供方智选：供应商匹配与评估")
st.caption(
    "本地演示原型：结合文本相关度和采购指标生成候选排序。"
    "示例数据均为虚构数据，结果仅供人工复核。"
)


def read_suppliers(uploaded_file, sheet_name=None):
    """Load the bundled CSV, an uploaded CSV, or one worksheet from an XLSX file."""
    if uploaded_file is None:
        return pd.read_csv(DEFAULT_DATA, encoding="utf-8-sig")

    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except UnicodeDecodeError:
            # Excel on some Chinese Windows setups exports CSV with a GBK-family encoding.
            return pd.read_csv(io.BytesIO(raw), encoding="gb18030")
    if suffix == ".xlsx":
        return pd.read_excel(
            io.BytesIO(raw),
            sheet_name=sheet_name,
            engine="openpyxl",
        )
    raise ValueError("仅支持 CSV 和 .xlsx 文件。")


def get_excel_sheet_names(uploaded_file):
    """Return worksheet names so the user can choose which sheet contains suppliers."""
    raw = uploaded_file.getvalue()
    workbook = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
    return workbook.sheet_names


def validate_data(df):
    """Normalize headers, validate required values, and skip invalid rows visibly."""
    if df is None or df.empty:
        st.error("数据表为空，请上传包含供应商记录的文件。")
        st.stop()

    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    if len(set(df.columns)) != len(df.columns):
        st.error("表格存在重复列名，请检查表头。")
        st.stop()

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        st.error("数据缺少必需字段：" + "、".join(missing))
        st.info("请参考项目目录中的 suppliers.csv 模板，保持列名一致。")
        st.stop()

    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    numeric_columns = ["unit_price", "quality_score", "on_time_rate", "compliance_score", "lead_days"]
    invalid = pd.Series(False, index=df.index)
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        invalid |= df[column].isna()

    invalid |= df["supplier"].eq("") | df["category"].eq("")
    invalid |= df["unit_price"] < 0
    invalid |= df["lead_days"] < 0
    for column in SCORE_COLUMNS:
        invalid |= ~df[column].between(0, 100, inclusive="both")

    invalid_count = int(invalid.sum())
    if invalid_count:
        st.warning(
            f"已跳过 {invalid_count} 条无效记录。请检查必填名称、数字字段、"
            "非负价格/交期，以及 0–100 范围内的质量、交付和合规分。"
        )
        df = df.loc[~invalid].copy()

    if df.empty:
        st.error("没有可用于评估的有效记录，请修正数据后重试。")
        st.stop()

    return df.reset_index(drop=True)


def minmax_score(values, lower_is_better):
    """Scale a candidate set to 0–100; ties receive the same neutral-equivalent score."""
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return pd.Series(100.0, index=values.index)
    if lower_is_better:
        return (high - values) / (high - low) * 100
    return (values - low) / (high - low) * 100


def rank_suppliers(df, query, weights, thresholds):
    """Calculate text relevance, weighted score, and transparent rule-based risk flags."""
    corpus = (
        df["category"].fillna("") + " " +
        df["regions"].fillna("") + " " +
        df["products"].fillna("") + " " +
        df["certifications"].fillna("")
    ).tolist()

    if query.strip():
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
        )
        try:
            matrix = vectorizer.fit_transform(corpus + [query.strip()])
            relevance = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        except ValueError as exc:
            # Very short text may contain no 2–4 character features to compare.
            if "empty vocabulary" not in str(exc).lower():
                raise
            relevance = [0.0] * len(df)
    else:
        relevance = [0.0] * len(df)

    out = df.copy()
    out["需求相关度"] = relevance * 100
    out["价格得分"] = minmax_score(out["unit_price"], lower_is_better=True)
    out["质量得分"] = out["quality_score"]
    out["交付得分"] = out["on_time_rate"]
    out["合规得分"] = out["compliance_score"]
    out["交期得分"] = minmax_score(out["lead_days"], lower_is_better=True)

    score_columns = {
        "需求相关度": weights["需求匹配"],
        "价格得分": weights["价格"],
        "质量得分": weights["质量"],
        "交付得分": weights["交付"],
        "合规得分": weights["合规"],
        "交期得分": weights["交期"],
    }
    total_weight = sum(score_columns.values())
    if total_weight <= 0:
        raise ValueError("评分权重总和必须大于 0。")

    out["综合得分"] = sum(
        out[column] * weight for column, weight in score_columns.items()
    ) / total_weight

    def risk_flags(row):
        flags = []
        if row["compliance_score"] < thresholds["合规"]:
            flags.append(f"合规分低于 {thresholds['合规']}")
        if row["on_time_rate"] < thresholds["准时交付率"]:
            flags.append(f"准时交付率低于 {thresholds['准时交付率']}%")
        if row["quality_score"] < thresholds["质量"]:
            flags.append(f"质量分低于 {thresholds['质量']}")
        return "；".join(flags) if flags else "未触发当前风险阈值"

    out["风险提示"] = out.apply(risk_flags, axis=1)
    return out.sort_values(
        ["综合得分", "需求相关度", "supplier"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


with st.sidebar:
    st.header("数据与评估设置")
    uploaded = st.file_uploader(
        "上传供应商数据",
        type=["csv", "xlsx"],
        help="支持 CSV 或 .xlsx；Excel 文件可选择工作表。",
    )
    st.caption("字段模板见项目目录 suppliers.csv。请勿上传未经授权的真实供应商或个人信息。")

    selected_sheet = None
    if uploaded is not None and Path(uploaded.name).suffix.lower() == ".xlsx":
        try:
            sheet_names = get_excel_sheet_names(uploaded)
            if not sheet_names:
                st.error("该 Excel 文件没有可读取的工作表。")
                st.stop()
            selected_sheet = st.selectbox("选择供应商数据工作表", sheet_names)
        except Exception as exc:
            st.error(f"无法读取 Excel 工作簿：{exc}")
            st.stop()

    st.subheader("评分权重")
    weights = {
        "需求匹配": st.slider("需求匹配", 0, 100, 30),
        "价格": st.slider("价格", 0, 100, 20),
        "质量": st.slider("质量", 0, 100, 20),
        "交付": st.slider("准时交付率", 0, 100, 15),
        "合规": st.slider("合规", 0, 100, 10),
        "交期": st.slider("交期", 0, 100, 5),
    }

    st.subheader("风险提示阈值")
    thresholds = {
        "合规": st.slider("合规分下限", 0, 100, 70),
        "准时交付率": st.slider("准时交付率下限", 0, 100, 85),
        "质量": st.slider("质量分下限", 0, 100, 75),
    }

if sum(weights.values()) == 0:
    st.warning("评分权重不能全部为 0，请至少提高一项权重。")
    st.stop()

try:
    suppliers = validate_data(read_suppliers(uploaded, selected_sheet))
except Exception as exc:
    st.error(f"无法读取供应商数据：{exc}")
    st.stop()

left, middle, right = st.columns([2, 1, 1])
with left:
    query = st.text_input(
        "采购需求",
        placeholder="例如：深圳办公家具，需环保认证、按时交付",
    )
with middle:
    categories = ["全部品类"] + sorted(suppliers["category"].unique().tolist())
    selected_category = st.selectbox("品类筛选", categories)
with right:
    all_regions = sorted({
        region.strip()
        for value in suppliers["regions"]
        for region in value.replace("；", ";").split(";")
        if region.strip()
    })
    selected_region = st.selectbox("服务区域", ["全部区域"] + all_regions)

candidates = suppliers.copy()
if selected_category != "全部品类":
    candidates = candidates[candidates["category"] == selected_category].copy()
if selected_region != "全部区域":
    candidates = candidates[
        candidates["regions"].apply(
            lambda value: selected_region in [
                region.strip()
                for region in value.replace("；", ";").split(";")
            ]
        )
    ].copy()

if candidates.empty:
    st.warning("当前品类和区域筛选条件下没有供应商，请调整筛选条件。")
    st.stop()

try:
    ranked = rank_suppliers(candidates, query, weights, thresholds)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"无法计算匹配结果：{exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("候选供应商", len(ranked))
col2.metric("最高综合得分", f"{ranked.iloc[0]['综合得分']:.1f} / 100")
col3.metric("权重总和", sum(weights.values()))

st.subheader("供应商排序")
display = ranked[
    [
        "supplier", "category", "regions", "综合得分", "需求相关度",
        "unit_price", "quality_score", "on_time_rate", "compliance_score",
        "lead_days", "风险提示",
    ]
].rename(columns={
    "supplier": "供应商",
    "category": "品类",
    "regions": "服务区域",
    "unit_price": "参考单价",
    "quality_score": "质量分",
    "on_time_rate": "准时交付率",
    "compliance_score": "合规分",
    "lead_days": "交期（天）",
})
st.dataframe(
    display.style.format({
        "综合得分": "{:.1f}",
        "需求相关度": "{:.1f}",
        "参考单价": "{:.2f}",
        "准时交付率": "{:.1f}%",
    }),
    use_container_width=True,
    hide_index=True,
)

best = ranked.iloc[0]
st.subheader(f"优先候选：{best['supplier']}")
st.write(
    f"综合得分 {best['综合得分']:.1f} 分。该排序仅基于当前筛选结果、需求文本和权重，"
    "建议结合实际报价、资质原件、服务范围和履约记录人工复核。"
)
st.write(
    f"**主要指标：**参考单价 {best['unit_price']:.2f}；质量分 "
    f"{best['quality_score']:.0f}；准时交付率 {best['on_time_rate']:.0f}%；"
    f"合规分 {best['compliance_score']:.0f}；交期 {best['lead_days']:.0f} 天。"
)
st.write(f"**规则风险提示：**{best['风险提示']}。")

with st.expander("评分方法、适用范围与限制"):
    st.markdown(
        """
        - **需求相关度：**对品类、服务区域、产品描述和资质描述使用字符级
          n-gram TF-IDF 与余弦相似度。它比较字面片段，不理解同义词，也不是中标概率。
        - **价格与交期：**在当前筛选候选集合内分别做 0–100 相对归一化。
          候选集合全部同价或同交期时，该项统一记为 100 分。
        - **质量、交付、合规：**直接使用表格中的 0–100 分或准时交付百分数。
        - **综合分：**各项分数按左侧权重做加权平均；权重总和必须大于 0。
        - **风险提示：**使用左侧可调整阈值，是规则提示，不是模型预测或证照核验。
        - **用途边界：**本项目是本地演示原型，样例数据为虚构数据；正式采购仍需人工核验并按制度审批。
        """
    )

csv_bytes = ranked.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下载排序结果 CSV",
    data=csv_bytes,
    file_name="supplier_ranking.csv",
    mime="text/csv",
)
