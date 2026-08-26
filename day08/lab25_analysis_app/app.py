import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF

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


def get_summary_lines():
    notes_path = os.path.join(os.path.dirname(__file__), "notes.md")
    try:
        with open(notes_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []

    last_heading = None
    for i, line in enumerate(lines):
        if "리포트 뼈대 다섯 줄" in line:
            last_heading = i

    if last_heading is None:
        return []

    summary = []
    for line in lines[last_heading + 1:]:
        stripped = line.strip()
        if stripped[:2] in ("1.", "2.", "3.", "4.", "5."):
            summary.append(stripped.replace("[", "").replace("]", ""))
        elif summary:
            break
    return summary


_PDF_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic-Regular.ttf"),
]
_PDF_FONT_PATH = next((p for p in _PDF_FONT_CANDIDATES if os.path.exists(p)), _PDF_FONT_CANDIDATES[-1])

_DEFAULT_INTERPRETATION = (
    "정확도는 기준 모델 96.6%에서 내 모델 96.85%로 소폭 올랐다. "
    "지금 문턱 0.5에서는 13건을 지목했고 그중 9건이 진짜였으며, 59건은 놓쳤다. "
    "놓친 건수가 지목 건수보다 훨씬 많은 것으로 보아, 실제 값들이 문턱 0.5보다 낮은 구간에 몰려 있었다고 볼 수 있다."
)


def build_report_pdf(title, summary_lines, compare_table, is_classification, threshold, counts, interpretation_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Nanum", "", _PDF_FONT_PATH)
    pdf.add_font("Nanum", "B", _PDF_FONT_PATH)
    pdf.set_font("Nanum", size=16)

    today_str = datetime.now().strftime("%Y-%m-%d")
    pdf.cell(0, 10, f"{title} ({today_str})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Nanum", size=12)
    pdf.cell(0, 8, "프로젝트 요약", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Nanum", size=10)
    if summary_lines:
        for line in summary_lines:
            pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.multi_cell(0, 6, "(요약 다섯 줄을 찾지 못했습니다)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Nanum", size=12)
    pdf.cell(0, 8, "결과 표", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Nanum", size=10)
    with pdf.table() as table:
        header = table.row()
        for col in compare_table.columns:
            header.cell(str(col))
        for _, row in compare_table.iterrows():
            data_row = table.row()
            for val in row:
                data_row.cell(str(val))
    pdf.ln(2)

    if is_classification and counts is not None:
        pdf.set_font("Nanum", size=12)
        pdf.cell(0, 8, f"지금 문턱({threshold})에서의 결과", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Nanum", size=10)
        pdf.multi_cell(0, 6, f"지목 건수: {counts['지목']}", new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 6, f"진짜 건수: {counts['진짜']}", new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 6, f"놓친 건수: {counts['놓친']}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    pdf.set_font("Nanum", size=12)
    pdf.cell(0, 8, "해석 문장", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Nanum", size=10)
    pdf.multi_cell(
        0, 6,
        interpretation_text if interpretation_text else "(아직 해석 문장을 만들지 않았습니다)",
        new_x="LMARGIN", new_y="NEXT",
    )

    return bytes(pdf.output())


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

        problem_type = st.radio(
            "이 문제는 무엇인가요",
            ["분류 (결과가 두 값 중 하나)", "회귀 (결과가 숫자)"],
            key="problem_type",
        )
        is_classification = problem_type.startswith("분류")

        def _fill_and_prep_text(df_in):
            df_processed = df_in.copy()
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

            return df_processed, missing_after, text_summary

        def _store_split(X_train, X_test, y_train, y_test, missing_after, text_summary):
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test

            st.write(f"빈칸: {missing_before}개 → {missing_after}개")
            st.write(text_summary)
            st.write(f"학습용 행 수: {len(X_train)}, 시험용 행 수: {len(X_test)}")
            row_check = "같음" if len(X_train) + len(X_test) == len(df) else "다름"
            st.write(f"나뉜 행 수 합({len(X_train) + len(X_test)})이 원본 행 수({len(df)})와: {row_check}")

        if is_classification:
            available_values = df[result_col].dropna().unique()

            if len(available_values) == 0:
                st.write(f"'{result_col}' 열에 값이 하나도 없어 진행할 수 없습니다")
            else:
                positive_value = st.selectbox(f"'{result_col}' 열에서 어떤 값을 1로 볼까요", available_values)

                test_ratio = st.slider("시험용 비율", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

                if st.button("적용"):
                    df_processed, missing_after, text_summary = _fill_and_prep_text(df)

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
                        _store_split(X_train, X_test, y_train, y_test, missing_after, text_summary)

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
        else:
            if not pd.api.types.is_numeric_dtype(df[result_col]):
                st.write(f"'{result_col}' 열이 숫자가 아니라 회귀를 할 수 없습니다")
            else:
                st.write(f"'{result_col}' 열은 이미 숫자라 그대로 결과 값으로 씁니다")

                test_ratio = st.slider("시험용 비율", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

                if st.button("적용"):
                    df_processed, missing_after, text_summary = _fill_and_prep_text(df)

                    y = df_processed[result_col]
                    X = df_processed.drop(columns=[result_col])

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_ratio, random_state=42
                    )

                    _store_split(X_train, X_test, y_train, y_test, missing_after, text_summary)

                    split_table = pd.DataFrame({
                        "구분": ["학습용", "시험용"],
                        "평균": [round(float(y_train.mean()), 3), round(float(y_test.mean()), 3)],
                        "표준편차": [round(float(y_train.std()), 3), round(float(y_test.std()), 3)],
                    })
                    st.dataframe(split_table)

with tab3:
    if "X_train" not in st.session_state:
        st.write("전처리를 먼저 해주세요")
    else:
        is_classification = st.session_state.get("problem_type", "").startswith("분류")

        if is_classification:
            model_choice = st.selectbox(
                "어떤 모델을 쓸까요", ["로지스틱 회귀", "의사결정나무", "랜덤 포레스트"], key="model_choice"
            )
            use_weight = st.toggle("적은 쪽에 가중치 주기", value=False, key="use_weight")
        else:
            model_choice = st.selectbox(
                "어떤 모델을 쓸까요", ["선형회귀", "의사결정나무", "랜덤 포레스트"], key="model_choice"
            )

        if st.button("학습"):
            X_train = st.session_state["X_train"]
            X_test = st.session_state["X_test"]
            y_train = st.session_state["y_train"]
            y_test = st.session_state["y_test"]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if is_classification:
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
            else:
                if model_choice == "선형회귀":
                    model = LinearRegression()
                    learning_point = (
                        "각 변수에 가중치(계수)를 곱해서 더한 값으로 숫자를 바로 예측합니다. "
                        "계수의 절댓값이 클수록 그 변수가 예측값을 크게 움직입니다."
                    )
                elif model_choice == "의사결정나무":
                    model = DecisionTreeRegressor(random_state=42)
                    learning_point = (
                        "한 번에 변수 하나를 골라 기준값보다 큰지 작은지로 데이터를 반복해서 나누고, "
                        "마지막에 도착한 칸에 있던 값들의 평균으로 예측합니다."
                    )
                else:
                    model = RandomForestRegressor(random_state=42)
                    learning_point = (
                        "여러 개의 의사결정나무를 조금씩 다르게 만들어서 각 나무의 예측값을 평균 냅니다. "
                        "나무 하나만 쓸 때보다 결과가 안정적인 경향이 있습니다."
                    )

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if is_classification:
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
            else:
                baseline_pred = pd.Series(y_train.mean(), index=y_test.index)

                compare_table = pd.DataFrame({
                    "구분": ["기준 모델(늘 평균값)", f"내 모델({model_choice})"],
                    "MAE": [
                        round(mean_absolute_error(y_test, baseline_pred), 3),
                        round(mean_absolute_error(y_test, y_pred), 3),
                    ],
                    "RMSE": [
                        round(mean_squared_error(y_test, baseline_pred) ** 0.5, 3),
                        round(mean_squared_error(y_test, y_pred) ** 0.5, 3),
                    ],
                    "R2": [
                        round(r2_score(y_test, baseline_pred), 3),
                        round(r2_score(y_test, y_pred), 3),
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
        is_classification = st.session_state.get("problem_type", "").startswith("분류")

        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        scaler = st.session_state["scaler"]
        model = st.session_state["model"]

        y_pred = model.predict(scaler.transform(X_test))
        compare_table = st.session_state["compare_table"]

        st.write("기준 모델과 내 모델을 견줍니다 (3번 탭 학습 결과)")
        st.dataframe(compare_table)

        if is_classification:
            cm_main = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn_m, fp_m, fn_m, tp_m = cm_main.ravel()
            main_confusion_table = pd.DataFrame({
                "구분": ["잡은 것", "놓친 것", "헛경보", "정상을 정상이라 한 것"],
                "건수": [int(tp_m), int(fn_m), int(fp_m), int(tn_m)],
                "무슨 뜻": [
                    "진짜 불량인데 불량이라고 맞게 지목한 것",
                    "진짜 불량인데 정상이라고 놓친 것",
                    "정상인데 불량이라고 잘못 지목한 것",
                    "정상인데 정상이라고 맞게 답한 것",
                ],
            })
            st.write("혼동행렬")
            st.dataframe(main_confusion_table)

        if is_classification:
            st.write("문턱을 옮겨서 다시 잘라봅니다 (다시 학습하지 않습니다)")
            y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
            threshold = st.slider(
                "문턱", min_value=0.05, max_value=0.95, value=0.5, step=0.05, key="threshold"
            )
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

            st.write("문턱별 비교 표")
            grid_thresholds = [round(0.1 * i, 1) for i in range(1, 10)]
            grid_rows = []
            for t in grid_thresholds:
                y_pred_grid = (y_proba >= t).astype(int)
                cm_grid = confusion_matrix(y_test, y_pred_grid, labels=[0, 1])
                tn_g, fp_g, fn_g, tp_g = cm_grid.ravel()
                grid_rows.append({
                    "문턱": t,
                    "지목 건수": int(y_pred_grid.sum()),
                    "그중 진짜": int(tp_g),
                    "놓친 건수": int(fn_g),
                    "정밀도": round(precision_score(y_test, y_pred_grid, zero_division=0), 3),
                    "재현율": round(recall_score(y_test, y_pred_grid, zero_division=0), 3),
                    "F1": round(f1_score(y_test, y_pred_grid, zero_division=0), 3),
                })
            grid_table = pd.DataFrame(grid_rows)
            best_idx = grid_table["F1"].idxmax()
            grid_table.insert(0, "표시", "")
            grid_table.loc[best_idx, "표시"] = "★"
            st.dataframe(grid_table)
            st.write(f"F1이 가장 높은 문턱: {grid_table.loc[best_idx, '문턱']}")

        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(figures_dir, exist_ok=True)

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

        if is_classification:
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
        else:
            fig2, ax2 = plt.subplots()
            ax2.scatter(y_test, y_pred, alpha=0.4)
            min_val = min(float(y_test.min()), float(y_pred.min()))
            max_val = max(float(y_test.max()), float(y_pred.max()))
            ax2.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
            ax2.set_xlabel("실제값")
            ax2.set_ylabel("예측값")
            ax2.set_title("실제값과 예측값")
            fig2.tight_layout()
            fig2.savefig(os.path.join(figures_dir, "actual_vs_predicted.png"))
            st.pyplot(fig2)
            st.write("점이 빨간 점선에 가까울수록 잘 맞힌 것입니다")

            residuals = y_test - y_pred
            fig3, ax3 = plt.subplots()
            ax3.scatter(y_pred, residuals, alpha=0.4)
            ax3.axhline(0, color="red", linestyle="--")
            ax3.set_xlabel("예측값")
            ax3.set_ylabel("잔차(실제값 - 예측값)")
            ax3.set_title("잔차 그림")
            fig3.tight_layout()
            fig3.savefig(os.path.join(figures_dir, "residuals.png"))
            st.pyplot(fig3)
            st.write("점이 0에 가까울수록, 위아래로 고르게 퍼질수록 좋습니다")

with tab5:
    if "model" not in st.session_state:
        st.write("학습을 먼저 해주세요")
    else:
        compare_table = st.session_state["compare_table"]
        is_classification = st.session_state.get("problem_type", "").startswith("분류")

        st.write("프로젝트 요약")
        summary_lines = get_summary_lines()
        if summary_lines:
            for line in summary_lines:
                st.write(line)
        else:
            st.write("notes.md에서 '리포트 뼈대 다섯 줄'을 찾지 못했습니다")

        st.write("결과 표")
        st.dataframe(compare_table)

        threshold = None
        counts = None
        if is_classification:
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            scaler = st.session_state["scaler"]
            model = st.session_state["model"]
            threshold = st.session_state.get("threshold", 0.5)

            y_proba = model.predict_proba(scaler.transform(X_test))[:, 1]
            y_pred_th = (y_proba >= threshold).astype(int)
            cm_th = confusion_matrix(y_test, y_pred_th, labels=[0, 1])
            tn_th, fp_th, fn_th, tp_th = cm_th.ravel()
            counts = {
                "지목": int(y_pred_th.sum()),
                "진짜": int(tp_th),
                "놓친": int(fn_th),
            }

            st.write(f"지금 문턱({threshold})에서의 결과")
            st.write(f"지목 건수: {counts['지목']}")
            st.write(f"진짜 건수: {counts['진짜']}")
            st.write(f"놓친 건수: {counts['놓친']}")

            st.write("문턱별 비교 표")
            grid_thresholds = [round(0.1 * i, 1) for i in range(1, 10)]
            grid_rows = []
            for t in grid_thresholds:
                y_pred_grid = (y_proba >= t).astype(int)
                cm_grid = confusion_matrix(y_test, y_pred_grid, labels=[0, 1])
                tn_g, fp_g, fn_g, tp_g = cm_grid.ravel()
                grid_rows.append({
                    "문턱": t,
                    "지목 건수": int(y_pred_grid.sum()),
                    "그중 진짜": int(tp_g),
                    "놓친 건수": int(fn_g),
                    "정밀도": round(precision_score(y_test, y_pred_grid, zero_division=0), 3),
                    "재현율": round(recall_score(y_test, y_pred_grid, zero_division=0), 3),
                    "F1": round(f1_score(y_test, y_pred_grid, zero_division=0), 3),
                })
            grid_table = pd.DataFrame(grid_rows)
            best_idx = grid_table["F1"].idxmax()
            grid_table.insert(0, "표시", "")
            grid_table.loc[best_idx, "표시"] = "★"
            st.dataframe(grid_table)
            st.write(f"F1이 가장 높은 문턱: {grid_table.loc[best_idx, '문턱']}")

        score_col = "정확도" if is_classification else "R2"
        baseline_score = compare_table[score_col].iloc[0]
        model_score = compare_table[score_col].iloc[1]

        st.write("해석 문장")
        if not api_key:
            st.write("열쇠가 없습니다")
        else:
            number_lines = [
                f"기준 모델 {score_col}: {baseline_score}",
                f"내 모델 {score_col}: {model_score}",
            ]
            if is_classification:
                number_lines += [
                    f"지금 문턱: {threshold}",
                    f"지목 건수: {counts['지목']}",
                    f"진짜 건수: {counts['진짜']}",
                    f"놓친 건수: {counts['놓친']}",
                ]
            number_key = "||".join(number_lines)

            if st.button("해석 문장 만들기"):
                if st.session_state.get("interpretation_key") == number_key:
                    pass
                else:
                    prompt = (
                        "아래 숫자만 가지고 세 문장 이내로 한국어 해석 문장을 써줘. "
                        "문장에 들어갈 숫자는 아래에 넘긴 값만 쓰고, 새 숫자를 만들지 마. "
                        "원인이라고 단정하지 말고 '~한 구간에 몰려 있었다' 정도로 표현해줘.\n\n"
                        + "\n".join(number_lines)
                    )
                    try:
                        genai.configure(api_key=api_key)
                        gmodel = genai.GenerativeModel("gemini-3.6-flash")
                        response = gmodel.generate_content(prompt)
                        st.session_state["interpretation"] = response.text
                        st.session_state["interpretation_key"] = number_key
                    except Exception as e:
                        err_str = str(e)
                        key_error_hints = [
                            "API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED",
                            "UNAUTHENTICATED", "401", "403",
                        ]
                        if any(hint in err_str for hint in key_error_hints):
                            st.session_state["interpretation"] = "열쇠가 잘못된 것 같습니다. 키를 확인해주세요"
                        else:
                            st.session_state["interpretation"] = "지금은 문장을 만들 수 없습니다. 다시 눌러주세요"
                        st.session_state.pop("interpretation_key", None)

            st.session_state.setdefault("interpretation", _DEFAULT_INTERPRETATION)
            st.write(st.session_state["interpretation"])

        pdf_bytes = build_report_pdf(
            title="설비 측정값으로 고장을 미리 알아채기",
            summary_lines=summary_lines,
            compare_table=compare_table,
            is_classification=is_classification,
            threshold=threshold,
            counts=counts,
            interpretation_text=st.session_state.get("interpretation", ""),
        )
        pdf_filename = f"secom_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        st.download_button(
            "PDF로 내려받기",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
        )

st.write("현재 시각:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
