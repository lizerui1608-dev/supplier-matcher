# 供方智选：供应商匹配与评估

一个面向采购寻源场景的本地演示项目。输入采购需求后，系统根据供应商品类、产品描述和资质文本计算文本相关度，并结合价格、质量、交付、合规和交期生成可解释的候选排序。项目样例数据为虚构数据。

## 功能

- 支持 CSV 供应商数据导入和品类筛选。
- 使用字符 n-gram TF-IDF 与余弦相似度完成中文需求文本匹配。
- 支持调整多指标权重并重新排序。
- 展示综合评分、分项指标和风险提示。
- 导出排序结果 CSV。

## 本地运行

需要 Python 3.10 或更新版本。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 Streamlit 显示的本地地址即可。macOS/Linux 激活命令为 `source .venv/bin/activate`。

## CSV 字段

`supplier`, `category`, `regions`, `products`, `certifications`, `unit_price`, `quality_score`, `on_time_rate`, `compliance_score`, `lead_days`。

评分字段应为 0–100，准时交付率使用百分数。`unit_price` 需在同一品类内使用可比较的计价单位。当前样例为了演示，价格仅用于同品类相对排序。

## 评分逻辑

需求匹配采用字符 n-gram TF-IDF 相似度；价格和交期在当前候选集合内归一化；质量、交付和合规使用输入分值。各项按用户设置的权重加权平均。调整权重可以表达采购侧的不同优先级，例如紧急采购可提高交期权重。

这是一个辅助决策原型。TF-IDF 不是大语言模型或深度语义模型，评分也不代表真实中标概率。正式采购应核验证照、报价有效期、税费、服务范围和履约记录，并由采购人员作出最终判断。

## 可继续完善

- 增加硬性准入条件与一票否决项。
- 加入历史履约数据和缺失值处理。
- 用人工标注的需求—供应商匹配样本评估 Precision@K / NDCG@K。
- 增加报价单字段抽取和多家报价对比。
- 增加评分版本、人工复核意见和审计记录。

## 数据说明

`supplier` 及所有供应商记录均为虚构内容，不对应真实企业或实际采购交易。
