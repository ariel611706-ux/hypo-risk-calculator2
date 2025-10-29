
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# ---------- Page ----------
st.set_page_config(page_title="消化内镜住院治疗患者低血糖风险计算器", layout="wide")
st.title("消化内镜住院治疗患者发生低血糖的风险计算器")
st.caption("基于Logistic回归的网络版护理辅助决策工具（含个体级解释、可下载图表）")

# ---------- Coefficients ----------
COEF = {
  "intercept": -6.18,
  "dx_colon": -2.342,
  "dx_gastric": -3.267,
  "dx_esophageal": -1.344,
  "dx_other": -1.534,
  "lax_yes": 2.826,
  "dbp_cat2": 1.928,
  "dbp_cat3": 0.287,
  "dbp_cat4": -0.168,
  "bun": 0.101,
  "glucose": 0.074,
  "hb": 0.151,
  "nrs_yes": 5.46
}

# ---------- Sidebar: Inputs ----------
with st.sidebar:
    st.header("输入变量")
    diagnosis = st.selectbox("原发诊断", [
        "Gallbladder/pancreatic disease (reference)",
        "Colonic lesion","Gastric neoplasm","Esophageal lesion","Other diseases"])
    dbp_cat = st.selectbox("舒张压分组", ["< 90 mmHg (reference)","90–100 mmHg","101–110 mmHg","> 110 mmHg"])
    glucose = st.number_input("血糖 (mmol/L)", 2.0, 30.0, 5.6, 0.1)
    bun = st.number_input("尿素氮 (mmol/L)", 0.0, 40.0, 6.0, 0.1)
    hb = st.number_input("血红蛋白 (g/L)", 60, 200, 130, 1)
    lax = st.selectbox("导泻剂使用", ["No / 否","Yes / 是"])
    nrs = st.selectbox("营养风险 NRS-2002 ≥ 3", ["No / 否","Yes / 是"])

    with st.expander("解释设置（基线与配色）"):
        base_glu = st.number_input("基线：血糖 mmol/L", 2.0, 30.0, 5.6, 0.1)
        base_bun = st.number_input("基线：尿素氮 mmol/L", 0.0, 40.0, 6.0, 0.1)
        base_hb  = st.number_input("基线：血红蛋白 g/L", 60, 200, 130, 1)
        base_dbp = st.selectbox("基线：舒张压分组", ["< 90 mmHg (reference)","90–100 mmHg","101–110 mmHg","> 110 mmHg"], index=0)
        base_lax = st.selectbox("基线：导泻剂使用", ["No / 否","Yes / 是"], index=0)
        base_nrs = st.selectbox("基线：NRS-2002 ≥3", ["No / 否","Yes / 是"], index=0)
        pos_color = st.color_picker("正向贡献颜色（提高风险）", "#c0392b")
        neg_color = st.color_picker("负向贡献颜色（降低风险）", "#2980b9")

    go = st.button("计算风险")

# ---------- Helper ----------
def logistic(x): return 1/(1+np.exp(-x))

def logit_from_inputs(diagnosis, dbp_cat, glucose, bun, hb, lax, nrs):
    z = COEF["intercept"] + COEF["glucose"]*glucose + COEF["bun"]*bun + COEF["hb"]*hb
    if dbp_cat == "90–100 mmHg": z += COEF["dbp_cat2"]
    elif dbp_cat == "101–110 mmHg": z += COEF["dbp_cat3"]
    elif dbp_cat == "> 110 mmHg": z += COEF["dbp_cat4"]
    if "Yes" in lax: z += COEF["lax_yes"]
    if "Yes" in nrs: z += COEF["nrs_yes"]
    if diagnosis == "Colonic lesion": z += COEF["dx_colon"]
    elif diagnosis == "Gastric neoplasm": z += COEF["dx_gastric"]
    elif diagnosis == "Esophageal lesion": z += COEF["dx_esophageal"]
    elif diagnosis == "Other diseases": z += COEF["dx_other"]
    return z

def contributions(diagnosis, dbp_cat, glucose, bun, hb, lax, nrs, base):
    contrib = {}
    contrib["Serum_glucose (mmol/L)"] = COEF["glucose"] * (glucose - base["glucose"])
    contrib["Blood_urea_nitrogen (mmol/L)"] = COEF["bun"] * (bun - base["bun"])
    contrib["Hemoglobin (g/L)"] = COEF["hb"] * (hb - base["hb"])
    for k,label in [("dbp_cat2","90–100 mmHg"),("dbp_cat3","101–110 mmHg"),("dbp_cat4","> 110 mmHg")]:
        contrib[f"DBP: {label}"] = COEF[k] * (1.0 if dbp_cat==label else 0.0)
    contrib["Laxative use (Yes)"] = COEF["lax_yes"] * (1.0 if "Yes" in lax else 0.0)
    contrib["NRS-2002 ≥3 (Yes)"] = COEF["nrs_yes"] * (1.0 if "Yes" in nrs else 0.0)
    for key,label in [("dx_colon","Colonic lesion"),("dx_gastric","Gastric neoplasm"),("dx_esophageal","Esophageal lesion"),("dx_other","Other diseases")]:
        contrib[f"Diagnosis: {label}"] = COEF[key] * (1.0 if diagnosis==label else 0.0)
    return contrib

# ---------- Layout ----------
col1, col2 = st.columns([1,1])
if go:
    z = logit_from_inputs(diagnosis, dbp_cat, glucose, bun, hb, lax, nrs)
    p = logistic(z)
    if p < 0.35:
        tier, advice = "低风险", "例行监测；常规术前宣教与观察。"
        col1.success(f"预测概率：{p:.1%}｜风险等级：{tier}")
    elif p < 0.65:
        tier, advice = "中等风险", "加强术前/术后血糖监测；确保补液与碳水化合物支持；评估导泻耐受。"
        col1.warning(f"预测概率：{p:.1%}｜风险等级：{tier}")
    else:
        tier, advice = "高风险", "术前优化血糖并个体化肠道准备；考虑缩短禁食；密切动态监测与沟通。"
        col1.error(f"预测概率：{p:.1%}｜风险等级：{tier}")
    col1.write(advice)

    base = {"glucose": base_glu, "bun": base_bun, "hb": base_hb, "dbp": base_dbp, "lax": base_lax, "nrs": base_nrs}
    contrib = contributions(diagnosis, dbp_cat, glucose, bun, hb, lax, nrs, base)
    df = pd.DataFrame({"Feature": list(contrib.keys()), "Contribution": list(contrib.values())}).set_index("Feature").sort_values("Contribution")

    col2.subheader("个体级解释（特征贡献，SHAP-like）")
    col2.caption("基于线性logit逐项分解；正值↑风险，负值↓风险；相对你设定的“基线”计算。")

    fig, ax = plt.subplots(figsize=(7,4))
    colors = [pos_color if v>0 else neg_color for v in df["Contribution"].values]
    ax.barh(df.index, df["Contribution"].values, color=colors)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Contribution to log-odds"); ax.set_ylabel("Feature")
    col2.pyplot(fig)

    # Downloads
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    col2.download_button("下载解释图（PNG）", data=buf.getvalue(), file_name="shap_like_contrib.png", mime="image/png")

    # Export inputs & prediction
    out = pd.DataFrame({
        "diagnosis":[diagnosis], "dbp_category":[dbp_cat], "glucose":[glucose], "bun":[bun], "hb":[hb],
        "laxative":[lax], "nrs2002":[nrs], "pred_prob":[p], "risk_tier":[tier]
    })
    csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
    col1.download_button("下载结果（CSV）", data=csv_bytes, file_name="prediction_result.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("全局变量重要性（|系数|近似）")
    imp = {
        "Serum_glucose (mmol/L)": abs(COEF["glucose"]),
        "Blood_urea_nitrogen (mmol/L)": abs(COEF["bun"]),
        "Hemoglobin (g/L)": abs(COEF["hb"]),
        "DBP: 90–100 mmHg": abs(COEF["dbp_cat2"]),
        "DBP: 101–110 mmHg": abs(COEF["dbp_cat3"]),
        "DBP: > 110 mmHg": abs(COEF["dbp_cat4"]),
        "Laxative use (Yes)": abs(COEF["lax_yes"]),
        "NRS-2002 ≥3 (Yes)": abs(COEF["nrs_yes"]),
        "Diagnosis: Colonic lesion": abs(COEF["dx_colon"]),
        "Diagnosis: Gastric neoplasm": abs(COEF["dx_gastric"]),
        "Diagnosis: Esophageal lesion": abs(COEF["dx_esophageal"]),
        "Diagnosis: Other diseases": abs(COEF["dx_other"]),
    }
    imp_df = pd.DataFrame(sorted(imp.items(), key=lambda x: x[1], reverse=True), columns=["Feature","|Coefficient| (proxy)"])
    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.barh(imp_df["Feature"][::-1], imp_df["|Coefficient| (proxy)"][::-1])
    ax2.set_xlabel("|Coefficient| (proxy for importance)"); ax2.set_ylabel("Feature")
    st.pyplot(fig2)
else:
    st.info("请在左侧输入变量后点击 **计算风险**。")
