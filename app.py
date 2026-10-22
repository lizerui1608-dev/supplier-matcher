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
    "qualification_status", "scope_match", "blacklisted",
    "unit_price", "quality_score", "on_time_rate", "compliance_score", "lead_days",
]
TEXT_COLUMNS = ["supplier", "category", "regions", "products", "certifications"]
SCORE_COLUMNS = ["quality_score", "on_time_rate", "compliance_score"]


st.set_page_config(page_title="供方智选 | 供应商评估", page_icon="📦", layout="wide")
st.title("供方智选：供应商匹配与评估")
st.caption(
    "个人项目演示原型：按品类和服务区域筛选资源，结合资格初筛、文本相关度和履约指标排序。"
    "所有样例数据均为虚构数据，最终结果需由采购人员复核。"
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
    workbook = pd.ExcelFile(
        io.BytesIO(uploaded_file.getvalue()),
        engine="openpyxl",
    )
    return workbook.sheet_names


def normalize_flag(value, positive_values, negative_values, positive_label, negative_label):
    if pd.isna(value):
        return "待核验"
    text = str(value).strip().lower()
    if text in positive_values:
        return positive_label
    if text in negative_values:
        return negative_label
    return "待核验"


def validate_data(df):
    """Validate schema and values. Unknown qualification flags remain pending, never pass."""
    if df is None or df.empty:
        st.error("数据表为空，请上传包含供应商记录的文件。")
        st.stop()

    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
    if len(set(df.columns)) != len(df.columns):
        st.error("表格存在清理列名后重复的字段，请检查表头。")
        st.stop()

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        st.error("数据缺少必需字段：" + "、".join(missing))
        st.info("请参考项目目录中的 suppliers.csv 模板，保持列名一致。")
        st.stop()

    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    pass_values = {"通过", "合格", "pass", "passed", "yes", "true", "1"}
    fail_values = {"不通过", "未通过", "fail", "failed", "no", "false", "0"}
    df["qualification_status"] = df["qualification_status"].apply(
        lambda value: normalize_flag(value, pass_values, fail_values, "通过", "不通过")
    )
    df["scope_match"] = df["scope_match"].apply(
        lambda value: normalize_flag(
            value,
            {"是", "匹配", "yes", "true", "1"},
            {"否", "不匹配", "no", "false", "0"},
            "匹配",
            "不匹配",
        )
    )
    df["blacklisted"] = df["blacklisted"].apply(
        lambda value: normalize_flag(
            value,
            {"是", "yes", "true", "1", "黑名单"},
            {"否", "no", "false", "0", "正常"},
            "是",
            "否",
        )
    )

    numeric_columns = [
        "unit_price", "quality_score", "on_time_rate", "compliance_score", "lead_days"
    ]
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
            f"已跳过 {invalid_count} 条数据格式无效的记录。请检查供应商名称、品类、"
            "数值字段，以及 0–100 范围内的质量、交付和合规分。"
        )
        df = df.loc[~invalid].copy()

    if df.empty:
        st.error("没有可用于评估的有效记录，请修正数据后重试。")
        st.stop()

    return df.reset_index(drop=True)


def qualification_review(row):
    """Apply transparent hard gates to manually supplied review statuses."""
    reasons = []
    if row["qualification_status"] != "通过":
        reasons.append(f"资格状态：{row['qualification_status']}")
    if row["scope_match"] != "匹配":
        reasons.append(f"经营范围：{row['scope_match']}")
    if row["blacklisted"] == "是":
        reasons.append("表格标记为黑名单")
    elif row["blacklisted"] == "待核验":
        reasons.append("黑名单状态待核验")

    if reasons:
        return pd.Series(["暂不进入能力评分", "；".join(reasons)])
    return pd.Series(["通过", "资格字段通过，仍需人工核验原始材料"])


def minmax_score(values, lower_is_better):
    """Normalize one metric within the current candidate set to a 0–100 scale."""
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return pd.Series(100.0, index=values.index)
    if lower_is_better:
        return (high - values) / (high - low) * 100
    return (values - low) / (high - low) * 100


def rank_suppliers(df, query, weights, thresholds):
    """Score eligible candidates using text relevance and weighted purchasing indicators."""
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
        "交付得分": weights["准时交付率"],
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

    out["规则风险提示"] = out.apply(risk_flags, axis=1)
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
        "准时交付率": st.slider("准时交付率", 0, 100, 15),
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

review = suppliers.apply(qualification_review, axis=1)
suppliers["资格初审结论"] = review[0]
suppliers["资格初审说明"] = review[1]
eligible = suppliers[suppliers["资格初审结论"] == "通过"].copy()
excluded = suppliers[suppliers["资格初审结论"] != "通过"].copy()

metric1, metric2, metric3 = st.columns(3)
metric1.metric("供应商记录", len(suppliers))
metric2.metric("资格字段初审通过", len(eligible))
metric3.metric("暂不进入能力评分", len(excluded))

if not excluded.empty:
    with st.expander("查看未进入能力评分的记录及原因"):
        st.dataframe(
            excluded[
                [
                    "supplier", "category", "regions", "qualification_status",
                    "scope_match", "blacklisted", "资格初审说明",
                ]
            ].rename(columns={
                "supplier": "供应商",
                "category": "品类",
                "regions": "服务区域",
                "qualification_status": "资格状态",
                "scope_match": "经营范围匹配",
                "blacklisted": "黑名单标记",
            }),
            use_container_width=True,
            hide_index=True,
        )

if eligible.empty:
    st.warning("没有同时满足当前资格字段的供应商，暂不进行能力评分。")
    st.stop()

left, middle, right = st.columns([2, 1, 1])
with left:
    query = st.text_input(
        "采购需求",
        placeholder="例如：深圳办公家具，需环保认证、按时交付",
    )
with middle:
    categories = ["全部品类"] + sorted(eligible["category"].unique().tolist())
    selected_category = st.selectbox("品类筛选", categories)
with right:
    all_regions = sorted({
        region.strip()
        for value in eligible["regions"]
        for region in value.replace("；", ";").split(";")
        if region.strip()
    })
    selected_region = st.selectbox("服务区域", ["全部区域"] + all_regions)

candidates = eligible.copy()
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
except Exception as exc:
    st.error(f"无法计算匹配结果：{exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("进入能力评分", len(ranked))
col2.metric("最高综合得分", f"{ranked.iloc[0]['综合得分']:.1f} / 100")
col3.metric("权重总和", sum(weights.values()))

st.subheader("能力评估候选排序")
display = ranked[
    [
        "supplier", "category", "regions", "资格初审结论",
        "综合得分", "需求相关度", "unit_price", "quality_score",
        "on_time_rate", "compliance_score", "lead_days", "规则风险提示",
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
    f"综合得分 {best['综合得分']:.1f} 分。资格初审只依据表格中的人工录入状态，"
    "不代表程序核验了营业执照、信用、黑名单或证书原件。请结合真实材料和业务意见复核。"
)
st.write(
    f"**主要指标：**参考单价 {best['unit_price']:.2f}；质量分 "
    f"{best['quality_score']:.0f}；准时交付率 {best['on_time_rate']:.0f}%；"
    f"合规分 {best['compliance_score']:.0f}；交期 {best['lead_days']:.0f} 天。"
)
st.write(f"**规则风险提示：**{best['规则风险提示']}。")

with st.expander("项目方法与使用边界"):
    st.markdown(
        """
        - **资格初筛：**根据表格中的资格状态、经营范围匹配和黑名单标记执行硬性过滤。
          这些状态由使用者录入；程序不读取或鉴真原始证照，也不查询企业信用数据。
        - **能力评分：**对通过资格字段筛选的候选，计算需求文本相关度，并结合价格、
          质量、准时交付率、合规分和交期形成可调权重排序。
        - **文本相关度：**使用字符级 2–4 gram TF-IDF 和余弦相似度，比较品类、
          服务区域、产品和资质描述的字面片段；不理解同义词，不代表中标概率。
        - **相对指标：**价格和交期只在当前品类/区域/资格筛选后的候选集合内归一化。
          候选全部同价或同交期时，该项统一记为 100 分。
        - **规则风险提示：**按可配置阈值标记质量、准时交付率或合规分，不是模型预测。
        - **范围：**本项目为本地个人演示原型，样例数据均为虚构。正式采购应按组织制度
          完成资料审核、能力评估、审批和人工决策。
        """
    )

csv_bytes = ranked.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下载能力评分结果 CSV",
    data=csv_bytes,
    file_name="supplier_capability_ranking.csv",
    mime="text/csv",
)
