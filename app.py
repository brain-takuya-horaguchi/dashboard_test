# app.py  --------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import logging, sys
# ---------- 追加 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # コンソール / Streamlit Terminal
    ]
)
log = logging.getLogger(__name__)
# ----------------------------------------------------------
# 列名マッピング（CSV 列名に合わせて適宜修正）
# ----------------------------------------------------------
COLUMN_MAP = {
    "id":           "ID",
    "age":          "年齢",
    "gender":       "性別",
    "base":         "拠点",
    "answer_flag":  "回答有無",
    "tenure_days":  "入社からの日数",
    "condition":    "コンディション",
    "text":         "自由記述",
    "created_at":   "登録日時"
}
CONDITION_ORDER = {"好調": 4, "普通": 3, "やや不調": 2, "不調": 1}

# ----------------------------------------------------------
# Streamlit 基本セットアップ
# ----------------------------------------------------------
st.set_page_config("従業員コンディション・ダッシュボード", layout="wide")
st.title("従業員コンディション・ダッシュボード")

uploaded = st.sidebar.file_uploader("CSV ファイルを選択してください", type=["csv"])
if uploaded is None:
    st.info("左のサイドバーから CSV をアップロードしてください。")
    st.stop()

# ----------------------------------------------------------
# データ読込 & 前処理
# ----------------------------------------------------------
df = pd.read_csv(uploaded, parse_dates=[COLUMN_MAP["created_at"]])
today = pd.Timestamp.today().normalize()

df["回答済み"]        = df[COLUMN_MAP["answer_flag"]].eq("あり")
df["コンディション数値"] = df[COLUMN_MAP["condition"]].map(CONDITION_ORDER)

df["年齢"]  = pd.to_numeric(df[COLUMN_MAP["age"]], errors="coerce")
df["年代"]  = df["年齢"].apply(lambda x: f"{int(x)//10*10}代" if pd.notna(x) else "不明")

df["勤続日数"]  = pd.to_numeric(df[COLUMN_MAP["tenure_days"]], errors="coerce")
df["勤続年数"]  = (df["勤続日数"] // 365).astype("Int64")
df["勤続カテゴリ"] = pd.cut(
    df["勤続年数"].fillna(-1),
    bins=[-1, 0, 1, 3, 5, 10, np.inf],
    labels=["1年未満", "1年", "1-3年", "3-5年", "5-10年", "10年以上"],
    right=False
)

df["週"] = df[COLUMN_MAP["created_at"]].dt.to_period("W").dt.start_time

# ----------------------------------------------------------
# サイドバー・フィルタ
# ----------------------------------------------------------
with st.sidebar.expander("フィルタ", expanded=False):
    base_opt   = st.multiselect("拠点",  df[COLUMN_MAP["base"]].dropna().unique())
    gender_opt = st.multiselect("性別",  df[COLUMN_MAP["gender"]].dropna().unique())
    cond_opt   = st.multiselect("コンディション", list(CONDITION_ORDER.keys()))

mask = pd.Series(True, index=df.index)
if base_opt:
    mask &= df[COLUMN_MAP["base"]].isin(base_opt)
if gender_opt:
    mask &= df[COLUMN_MAP["gender"]].isin(gender_opt)
if cond_opt:
    mask &= df[COLUMN_MAP["condition"]].isin(cond_opt)

data = df.loc[mask]

# ----------------------------------------------------------
# 共通ユーティリティ
# ----------------------------------------------------------
def stacked_bar(source: pd.DataFrame, category: str, title: str) -> None:
    """
    <category> × コンディション の“構成比”を積み上げ棒で描画
    """
    # 件数 → 割合（tidy）-----------------------------------
    counts = (
        source
        .groupby([category, COLUMN_MAP["condition"]])
        .size()
        .reset_index(name="件数")
    )
    counts["割合"] = counts["件数"] / counts.groupby(category)["件数"].transform("sum")

    # 描画 -----------------------------------------------
    fig = px.bar(
        counts,
        x=category,
        y="割合",
        color=COLUMN_MAP["condition"],
        category_orders={COLUMN_MAP["condition"]: list(CONDITION_ORDER.keys())},
        barmode="stack",
        text_auto=".0%",
        title=title
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# タブ構成
# ----------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["A. KPI/全体概要", "B. 属性 × コンディション", "C. 自由記述分析",
     "D. 深掘り", "E. 時系列", "F. 従業員リスト"]
)

# ===============================
# A. KPI / 全体概要
# ===============================
with tab1:
    st.subheader("全体のコンディション分布")
    cond_count = data[COLUMN_MAP["condition"]].value_counts().reindex(CONDITION_ORDER.keys())
    st.plotly_chart(
        px.pie(cond_count, values=cond_count.values, names=cond_count.index,
               hole=0.4, title="コンディション構成比"),
        use_container_width=True
    )

    # 回答率
    answered_rate = data["回答済み"].mean() if len(data) else 0
    col1, col2 = st.columns(2)
    col1.metric("回答率", f"{answered_rate*100:.1f} %")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=answered_rate*100,
        gauge={'axis': {'range': [0, 100]}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    col2.plotly_chart(gauge, use_container_width=True)

    # ワードクラウド
    st.subheader("自由記述ワードクラウド")
    text_join = " ".join(data[COLUMN_MAP["text"]].dropna().astype(str))
    if text_join:
        jp_fonts = [  # 利用可能な日本語フォントの候補
            Path(__file__).parent / "assets/fonts/NotoSansJP-VariableFont.ttf",
        ]
        font_path = next((p for p in jp_fonts if pathlib.Path(p).exists()), None)
        wc = WordCloud(width=800, height=400, background_color="white",
                       collocations=False, font_path=font_path).generate(text_join)
        fig_wc, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("自由記述のテキストがありません。")

    # 登録日別回答数
    st.subheader("登録日別 回答件数")
    daily = data.groupby(data[COLUMN_MAP["created_at"]].dt.date).size()
    st.bar_chart(daily)

# ===============================
# B. 属性 × コンディション
# ===============================
with tab2:
    st.subheader("属性別 コンディション割合")
    stacked_bar(data, COLUMN_MAP["base"],   "拠点別コンディション割合")
    stacked_bar(data, COLUMN_MAP["gender"], "性別コンディション割合")
    stacked_bar(data, "年代",              "年代別コンディション割合")
    stacked_bar(data, "勤続カテゴリ",       "勤続年数別コンディション割合")

# ===============================
# C. 自由記述分析
# ===============================
with tab3:
    st.subheader("自由記述 ネガ/ポジ分類（簡易キーワード抽出）")
    NEG_KEYS = ["評価", "不満", "人間関係", "忙しい", "残業"]
    POS_KEYS = ["やりがい", "雰囲気", "キャリア", "楽しい", "感謝"]

    def key_count(keys):
        return pd.Series({k: data[COLUMN_MAP["text"]].fillna("").str.contains(k).sum()
                          for k in keys}).sort_values(ascending=False)

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.bar(key_count(NEG_KEYS), title="主要ネガティブコメント分類"),
                      use_container_width=True)
    col2.plotly_chart(px.bar(key_count(POS_KEYS), title="主要ポジティブコメント分類"),
                      use_container_width=True)

    st.subheader("コンディション別 主要テーマ（件数）")
    theme_df = pd.DataFrame({
        "ネガ": data.groupby(COLUMN_MAP["condition"])[COLUMN_MAP["text"]]
                   .apply(lambda s: s.fillna("").str.contains("|".join(NEG_KEYS)).sum()),
        "ポジ": data.groupby(COLUMN_MAP["condition"])[COLUMN_MAP["text"]]
                   .apply(lambda s: s.fillna("").str.contains("|".join(POS_KEYS)).sum())
    })
    st.dataframe(theme_df)

# ===============================
# D. 深掘り
# ===============================
with tab4:
    st.subheader("不調・やや不調の属性分布")
    focus = data[data[COLUMN_MAP["condition"]].isin(["不調", "やや不調"])]
    if focus.empty:
        st.info("該当者がいません。")
    else:
        stacked_bar(focus, COLUMN_MAP["base"],   "拠点別（不調＋やや不調）")
        stacked_bar(focus, "年代",              "年代別（不調＋やや不調）")
        stacked_bar(focus, "勤続カテゴリ",       "勤続年数別（不調＋やや不調）")

    st.subheader("未回答者の属性分析")
    unanswered = df[~df["回答済み"]]
    if unanswered.empty:
        st.info("未回答者がいません。")
    else:
        stacked_bar(unanswered, COLUMN_MAP["base"], "未回答者の拠点別割合")
        stacked_bar(unanswered, "年代",              "未回答者の年代別割合")

    st.subheader("勤続日数 vs コンディションスコア")
    st.plotly_chart(
        px.scatter(data, x="勤続日数", y="コンディション数値",
                   color=COLUMN_MAP["condition"],
                   hover_data=[COLUMN_MAP["base"], "年代", COLUMN_MAP["gender"]],
                   title="勤続日数とコンディションの関係"),
        use_container_width=True
    )

    st.subheader("コンディション × 自由記述有無")
    tmp = (data.assign(自由記述あり=data[COLUMN_MAP["text"]].notna())
                 .groupby([COLUMN_MAP["condition"], "自由記述あり"]).size().unstack().fillna(0))
    st.plotly_chart(px.bar(tmp, barmode="group", title="自由記述有無"),
                    use_container_width=True)

# ===============================
# E. 時系列
# ===============================
with tab5:
    st.subheader("週次推移")

    weekly = (data.groupby(["週", COLUMN_MAP["condition"]])
                    .size()
                    .unstack()
                    .reindex(columns=CONDITION_ORDER.keys())
                    .fillna(0))

    try:
        # --- 変更: stackgroup キーワードを削除 -----------------
        fig_area = px.area(
            weekly,
            title="コンディションの週次推移",  # デフォルトで stacked
        )
        st.plotly_chart(fig_area, use_container_width=True)
        log.info("週次推移グラフを正常に描画しました。（行=%d, 列=%d）",
                 *weekly.shape)
    except Exception as e:
        log.exception("週次推移グラフの描画で例外発生")
        st.error(f"Plotly area グラフの描画でエラー: {e}")
        # デバッグ用に DF 情報をサイドバーへ
        with st.sidebar.expander("デバッグ: weekly DataFrame", expanded=False):
            st.write(weekly.head())
            st.code(repr(weekly.dtypes))

    st.subheader("拠点別 平均年齢 & 平均勤続年数")
    base_stats = (data.groupby(COLUMN_MAP["base"])
                       .agg(平均年齢=("年齢", "mean"),
                            平均勤続年数=("勤続年数", "mean"))
                       .round(1).sort_values("平均年齢"))
    st.dataframe(base_stats)

    st.subheader("コンディション改善 / 悪化トレンド (前週比)")
    last_week = (today - pd.Timedelta(weeks=1)).to_period("W").start_time
    if len(weekly) >= 2 and last_week in weekly.index:
        current_ratio = weekly.iloc[-1] / weekly.iloc[-1].sum()
        prev_ratio    = weekly.loc[last_week] / weekly.loc[last_week].sum()
        change = (current_ratio - prev_ratio) * 100
    else:
        change = pd.Series(0, index=weekly.columns)
    col_pos, col_neg = st.columns(2)
    col_pos.metric("好調 % 変化", f"{change['好調']:+.1f} %")
    col_neg.metric("不調 % 変化", f"{change['不調']:+.1f} %")

# ===============================
# F. 従業員リスト
# ===============================
with tab6:
    st.subheader("従業員リスト（フィルタ可能）")
    show_cols = [
        COLUMN_MAP["id"], COLUMN_MAP["base"], COLUMN_MAP["gender"],
        "年代", "勤続カテゴリ", COLUMN_MAP["condition"], "コンディション数値",
        COLUMN_MAP["text"]
    ]
    st.dataframe(data[show_cols].sort_values(COLUMN_MAP["base"]))
