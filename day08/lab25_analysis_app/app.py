import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dotenv import load_dotenv
import google.generativeai as genai

plt.rcParams["axes.unicode_minus"] = False

_KOREAN_FONT_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic-Regular.ttf"),
    r"C:\Windows\Fonts\malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _font_path in _KOREAN_FONT_CANDIDATES:
    if os.path.exists(_font_path):
        font_manager.fontManager.addfont(_font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_font_path).get_name()
        break

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def get_api_key():
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.getenv("GOOGLE_API_KEY")
    return key


api_key = get_api_key()

st.set_page_config(page_title="설비 측정값으로 고장을 미리 알아채기")

st.title("설비 측정값으로 고장을 미리 알아채기")
st.caption("공정 조건 다섯 가지로 고장 여부를 판별합니다")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["데이터 훑기", "전처리", "학습", "결과", "리포트"])

with tab1:
    uploaded_file = st.file_uploader("CSV 파일을 올려주세요", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df

        st.write(f"행 수: {df.shape[0]}, 열 수: {df.shape[1]}")

        st.dataframe(df.head())

        missing_counts = df.isnull().sum()
        missing_total = missing_counts.sum()
        st.write(f"빈칸 개수: {missing_total}")

        if missing_total > 0:
            missing_cols = missing_counts[missing_counts > 0]
            missing_table = pd.DataFrame({
                "열 이름": missing_cols.index,
                "빈칸 개수": missing_cols.values,
                "빈칸 비율": (missing_cols.values / len(df) * 100).round(2),
            })
            st.dataframe(missing_table)
        else:
            st.write("빈칸 없음")

        st.write("맞는 열인지 확인하세요")
        result_col = st.selectbox("결과 열을 선택하세요", df.columns, index=len(df.columns) - 1, key="result_col")

        value_counts = df[result_col].value_counts()
        value_ratio = df[result_col].value_counts(normalize=True) * 100
        result_table = pd.DataFrame({
            "값": value_counts.index,
            "개수": value_counts.values,
            "비율": value_ratio.values.round(2),
        })
        st.dataframe(result_table)
    else:
        st.write("파일을 올려주세요")

with tab2:
    if st.session_state.get("df") is None:
        st.write("1번 탭에서 파일을 올려주세요")
    else:
        df = st.session_state["df"]
        result_col = st.session_state.get("result_col", df.columns[-1])

        missing_before = int(df.isnull().sum().sum())
        st.write(f"빈칸 개수: {missing_before}")

        if missing_before == 0:
            st.write("빈칸이 없습니다. 채울 것이 없어요")
            fill_method = None
        else:
            fill_method = st.selectbox("빈칸을 무엇으로 채울까요", ["중앙값", "평균", "0"])

        text_cols = [c for c in df.select_dtypes(include="object").columns if c != result_col]
        if text_cols:
            st.write("글자로 된 열:", ", ".join(text_cols))
            text_col_action = st.radio("글자 열을 어떻게 할까요", ["학습에서 빼기", "숫자로 바꾸기"])
        else:
            text_col_action = None

        available_values = df[result_col].dropna().unique()

        if len(available_values) == 0:
            st.write(f"'{result_col}' 열에 값이 하나도 없어 진행할 수 없습니다")
        else:
            positive_value = st.selectbox(f"'{result_col}' 열에서 어떤 값을 1로 볼까요", available_values)

            test_ratio = st.slider("시험용 비율", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

            if st.button("적용"):
                df_processed = df.copy()
                numeric_cols = df_processed.select_dtypes(include="number").columns

                if missing_before > 0:
                    if fill_method == "중앙값":
                        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                            df_processed[numeric_cols].median()
                        )
                    elif fill_method == "평균":
                        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
                            df_processed[numeric_cols].mean()
                        )
                    else:
                        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)

                missing_after = int(df_processed.isnull().sum().sum())

                if text_cols:
                    if text_col_action == "학습에서 빼기":
                        df_processed = df_processed.drop(columns=text_cols)
                        text_summary = f"글자 열 {len(text_cols)}개를 학습에서 뺐습니다"
                    else:
                        for col in text_cols:
                            df_processed[col] = df_processed[col].astype("category").cat.codes
                        text_summary = f"글자 열 {len(text_cols)}개를 숫자로 바꿨습니다"
                else:
                    text_summary = "글자 열이 없습니다"

                target = (df[result_col] == positive_value).astype(int)
                st.write(f"결과 열을 0과 1로 바꿨습니다. 1: {int(target.sum())}건")
                X = df_processed.drop(columns=[result_col])
                y = target

                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_ratio, stratify=y, random_state=42
                    )
                except ValueError:
                    st.write("학습용·시험용으로 나눌 수 없습니다: 고른 값의 건수가 너무 적습니다 (최소 2건 필요)")
                else:
                    st.session_state["X_train"] = X_train
                    st.session_state["X_test"] = X_test
                    st.session_state["y_train"] = y_train
                    st.session_state["y_test"] = y_test

                    st.write(f"빈칸: {missing_before}개 → {missing_after}개")
                    st.write(text_summary)
                    st.write(f"학습용 행 수: {len(X_train)}, 시험용 행 수: {len(X_test)}")

                    train_counts = y_train.value_counts()
                    test_counts = y_test.value_counts()
                    train_ratio_1 = y_train.value_counts(normalize=True) * 100
                    test_ratio_1 = y_test.value_counts(normalize=True) * 100

                    split_table = pd.DataFrame({
                        "구분": ["학습용", "시험용"],
                        "1 개수": [int(train_counts.get(1, 0)), int(test_counts.get(1, 0))],
                        "1 비율": [
                            round(float(train_ratio_1.get(1, 0)), 2),
                            round(float(test_ratio_1.get(1, 0)), 2),
                        ],
                    })
                    st.dataframe(split_table)

with tab3:
    if "X_train" not in st.session_state:
        st.write("전처리를 먼저 해주세요")
    else:
        model_choice = st.selectbox(
            "어떤 모델을 쓸까요", ["로지스틱 회귀", "의사결정나무", "랜덤 포레스트"], key="model_choice"
        )
        use_weight = st.toggle("적은 쪽에 가중치 주기", value=False, key="use_weight")

        if st.button("학습"):
            X_train = st.session_state["X_train"]
            X_test = st.session_state["X_test"]
            y_train = st.session_state["y_train"]
            y_test = st.session_state["y_test"]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            class_weight = "balanced" if use_weight else None

            if model_choice == "로지스틱 회귀":
                model = LogisticRegression(max_iter=1000, class_weight=class_weight)
                learning_point = (
                    "각 변수에 가중치(계수)를 곱해서 더한 값을 0~1 사이 확률로 바꿔 판단합니다. "
                    "계수의 절댓값이 클수록 그 변수가 판단에 크게 작용합니다."
                )
            elif model_choice == "의사결정나무":
                model = DecisionTreeClassifier(random_state=42, class_weight=class_weight)
                learning_point = (
                    "한 번에 변수 하나를 골라 기준값보다 큰지 작은지로 데이터를 둘로 나누는 것을 반복합니다. "
                    "어떤 변수로 얼마나 자주 나눴는지가 변수 중요도로 남습니다."
                )
            else:
                model = RandomForestClassifier(random_state=42, class_weight=class_weight)
                learning_point = (
                    "여러 개의 의사결정나무를 조금씩 다르게 만들어서 다수결로 판단합니다. "
                    "나무 하나만 쓸 때보다 결과가 안정적인 경향이 있습니다."
                )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            baseline_value = y_train.mode()[0]
            baseline_pred = pd.Series(baseline_value, index=y_test.index)

            compare_table = pd.DataFrame({
                "구분": ["기준 모델(전부 정상)", f"내 모델({model_choice})"],
                "정확도": [
                    round(accuracy_score(y_test, baseline_pred) * 100, 2),
                    round(accuracy_score(y_test, y_pred) * 100, 2),
                ],
                "정밀도": [
                    round(precision_score(y_test, baseline_pred, zero_division=0), 3),
                    round(precision_score(y_test, y_pred, zero_division=0), 3),
                ],
                "재현율": [
                    round(recall_score(y_test, baseline_pred, zero_division=0), 3),
                    round(recall_score(y_test, y_pred, zero_division=0), 3),
                ],
                "F1": [
                    round(f1_score(y_test, baseline_pred, zero_division=0), 3),
                    round(f1_score(y_test, y_pred, zero_division=0), 3),
                ],
            })

            st.session_state["scaler"] = scaler
            st.session_state["model"] = model
            st.session_state["compare_table"] = compare_table
            st.write("학습이 끝났습니다")
            st.write(f"학습 포인트 ({model_choice}): {learning_point}")
            st.dataframe(compare_table)

with tab4:
    if "model" not in st.session_state:
        st.write("학습을 먼저 해주세요")
    else:
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        scaler = st.session_state["scaler"]
        model = st.session_state["model"]

        y_pred = model.predict(scaler.transform(X_test))
        compare_table = st.session_state["compare_table"]

        st.write("기준 모델과 내 모델을 견줍니다 (3번 탭 학습 결과)")
        st.dataframe(compare_table)

        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(figures_dir, exist_ok=True)

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.4)
        min_val = min(float(y_test.min()), float(y_pred.min()))
        max_val = max(float(y_test.max()), float(y_pred.max()))
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        ax.set_xlabel("실제값")
        ax.set_ylabel("예측값")
        ax.set_title("실제값과 예측값")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "actual_vs_predicted.png"))
        st.pyplot(fig)

        X_train = st.session_state["X_train"]

        if hasattr(model, "coef_"):
            importance = pd.Series(model.coef_.ravel(), index=X_train.columns).abs().sort_values(ascending=False)
            importance_label = "영향 크기(계수 절댓값)"
        else:
            importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            importance_label = "영향 크기(변수 중요도)"

        fig1, ax1 = plt.subplots()
        sns.barplot(x=importance.values, y=importance.index, ax=ax1)
        ax1.set_xlabel(importance_label)
        ax1.set_ylabel("변수")
        ax1.set_title("중요 변수")
        fig1.tight_layout()
        fig1.savefig(os.path.join(figures_dir, "importance.png"))
        st.pyplot(fig1)
        st.write("어느 항목이 판단에 많이 쓰였는지 보여줍니다")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_xlabel("예측")
        ax2.set_ylabel("실제")
        ax2.set_title("혼동행렬")
        fig2.tight_layout()
        fig2.savefig(os.path.join(figures_dir, "confusion_matrix.png"))
        st.pyplot(fig2)
        st.write("맞춘 것과 틀린 것이 각각 몇 건인지 보여줍니다")

        compare_ratio_table = compare_table.copy()
        compare_ratio_table["정확도"] = compare_ratio_table["정확도"] / 100
        compare_long = compare_ratio_table.melt(id_vars="구분", var_name="지표", value_name="값")
        fig3, ax3 = plt.subplots()
        sns.barplot(data=compare_long, x="지표", y="값", hue="구분", ax=ax3)
        ax3.set_ylabel("값 (0~1로 맞춤)")
        ax3.set_title("기준 모델과 내 모델 점수 비교")
        fig3.tight_layout()
        fig3.savefig(os.path.join(figures_dir, "compare_bar.png"))
        st.pyplot(fig3)
        st.write("기준 모델과 내 모델의 점수를 나란히 놓고 비교합니다")

        st.write("문턱을 옮겨서 다시 잘라봅니다 (다시 학습하지 않습니다)")
        y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
        threshold = st.slider("문턱", min_value=0.05, max_value=0.95, value=0.5, step=0.05)
        st.write(f"지금 문턱: {threshold}")

        y_pred_th = (y_proba >= threshold).astype(int)
        cm_th = confusion_matrix(y_test, y_pred_th, labels=[0, 1])
        tn_th, fp_th, fn_th, tp_th = cm_th.ravel()

        st.write(f"지목한 건수: {int(y_pred_th.sum())}")
        st.write(f"그중 진짜 건수: {int(tp_th)}")
        st.write(f"놓친 건수: {int(fn_th)}")

        threshold_score_table = pd.DataFrame({
            "지표": ["정밀도", "재현율", "F1"],
            "값": [
                round(precision_score(y_test, y_pred_th, zero_division=0), 3),
                round(recall_score(y_test, y_pred_th, zero_division=0), 3),
                round(f1_score(y_test, y_pred_th, zero_division=0), 3),
            ],
        })
        st.dataframe(threshold_score_table)

        threshold_confusion_table = pd.DataFrame({
            "구분": ["잡은 것", "놓친 것", "헛경보", "정상을 정상이라 한 것"],
            "건수": [int(tp_th), int(fn_th), int(fp_th), int(tn_th)],
        })
        st.dataframe(threshold_confusion_table)

with tab5:
    if not api_key:
        st.write("열쇠가 없습니다")
    elif "compare_table" not in st.session_state:
        st.write("4번 탭에서 결과를 먼저 확인해주세요")
    else:
        if st.button("리포트 만들기"):
            compare_table = st.session_state["compare_table"]
            prompt = (
                "다음은 설비 고장 예측 모델의 결과 표입니다. 이 표만 보고 다섯 줄로 리포트를 써줘. "
                "각 줄은 두 문장을 넘지 않게. 숫자를 새로 만들지 말고 표에 있는 값만 써. "
                "1) 무엇을 판단하려 했나 2) 데이터를 어떻게 손봤나 "
                "3) 어떤 모델을 왜 골랐나 4) 결과가 어땠나(잘된 것과 놓친 것을 같이) "
                "5) 표에 있는 지표를 근거로 아쉬운 점이나 다음에 해볼 것\n\n"
                f"{compare_table.to_string(index=False)}"
            )
            try:
                genai.configure(api_key=api_key)
                gmodel = genai.GenerativeModel("gemini-3.6-flash")
                response = gmodel.generate_content(prompt)
                st.write(response.text)
            except Exception as e:
                st.write(f"리포트를 만들지 못했습니다: {e}")

st.write("현재 시각:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
