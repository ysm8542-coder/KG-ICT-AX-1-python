import glob
import os
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
 
st.set_page_config(page_title="배터리 코팅 결함 검사 대시보드", layout="wide")
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.7"))

def find_csv_files(folder: str):
    files = glob.glob(os.path.join(folder, "results*.csv"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files
 
@st.cache_data
def load_data(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S_%f", errors="coerce")
    df["status"] = df["label"].apply(lambda x: "정상" if str(x).startswith("정상") else "불량")
    return df
 
def slim_bar_chart(series: pd.Series, x_label: str = "항목", y_label: str = "건수", bar_size: int = 28, height: int = 320):
    data = series.rename_axis(x_label).reset_index(name=y_label)
    data[x_label] = data[x_label].astype(str)
    chart = (
        alt.Chart(data)
        .mark_bar(size=bar_size)
        .encode(
            x=alt.X(f"{x_label}:N", sort=None, title=x_label),
            y=alt.Y(f"{y_label}:Q", title=y_label),
            tooltip=[x_label, y_label],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)
st.title("배터리 코팅 결함 검사 대시보드")
 
default_folder = os.environ.get("DATA_DIR")
if not default_folder:
    default_folder = "results" if os.path.isdir("results") else os.getcwd()
data_folder = st.sidebar.text_input("데이터 폴더 경로", value=default_folder)
st.sidebar.caption("results*.csv 파일이 들어있는 폴더 경로를 입력하세요")
 
if not os.path.isdir(data_folder):
    st.error(f"폴더를 찾을 수 없습니다: {data_folder}")
    st.stop()
 
csv_files = find_csv_files(data_folder)
if not csv_files:
    st.warning(f"'{data_folder}' 폴더에서 results*.csv 파일을 찾을 수 없습니다. Server.py를 실행해 검사를 먼저 진행해주세요.")
    st.stop()
 
default_csv = os.environ.get("LATEST_SESSION_CSV")
default_csv_path = os.path.join(data_folder, default_csv) if default_csv else None
default_index = csv_files.index(default_csv_path) if default_csv_path in csv_files else 0
 
selected_csv = st.sidebar.selectbox(
    "분석할 csv 파일",
    csv_files,
    index=default_index,
    format_func=os.path.basename,
)
st.sidebar.caption(
    "**results.csv** = 전체 누적 로그\n\n"
    "**results_YYYYMMDD_HHMMSS.csv** = 종료 시점의 세션 스냅샷 (한 번 생성되면 바뀌지 않음)"
)
 
df = load_data(selected_csv, os.path.getmtime(selected_csv))
 
if df.empty:
    st.info("선택한 파일에 데이터가 없습니다.")
    st.stop()
 
total = len(df)
normal_count = int((df["status"] == "정상").sum())
defect_count = total - normal_count
defect_df = df[df["status"] == "불량"]
env_df = df.dropna(subset=["temperature", "humidity"])
 
# 총 검사수 및 양/불량 비율
st.header("1. 총 검사수 및 양/불량 비율")
c1, c2, c3 = st.columns(3)
c1.metric("총 검사수", f"{total} 개")
c2.metric("양품", f"{normal_count} 개", f"{normal_count/total*100:.1f}%")
c3.metric("불량", f"{defect_count} 개", f"{defect_count/total*100:.1f}%", delta_color="inverse")
slim_bar_chart(df["status"].value_counts(), x_label="판정", y_label="건수")
 
# 전체 공정 판정 결과(라벨별)
st.header("2. 전체 공정 판정 결과")
label_counts = df["label"].value_counts()
col1, col2 = st.columns([2, 1])
with col1:
    slim_bar_chart(label_counts, x_label="라벨", y_label="건수")
with col2:
    st.dataframe(label_counts.rename("건 수"), use_container_width=True)
 
# 결함 유형별 발생 빈도
st.header("3. 결함 유형별 발생 빈도")
if defect_df.empty:
    st.success("이 데이터에는 불량이 하나도 없습니다")
else:
    slim_bar_chart(defect_df["label"].value_counts(), x_label="결함 유형", y_label="건수")
 
# 환경(온습도) 통계
st.header("4. 환경(온습도) 통계")
missing_env = total - len(env_df)
c1, c2, c3, c4 = st.columns(4)
if not env_df.empty:
    temp_mean = env_df["temperature"].mean()
    hum_mean = env_df["humidity"].mean()
    temp_min = env_df["temperature"].min()
    temp_max = env_df["temperature"].max()
    c1.metric("평균 온도", f"{temp_mean:.2f}°C")
    c2.metric("평균 습도", f"{hum_mean:.2f}%")
    c3.metric("온도 범위", f"{temp_min:.2f}°C ~ {temp_max:.2f}°C")
else:
    c1.metric("평균 온도", "-")
    c2.metric("평균 습도", "-")
    c3.metric("온도 범위", "-")
c4.metric("센서 결측", f"{missing_env}건", f"{missing_env/total*100:.2f}%", delta_color="inverse")
 
if not env_df.empty:
    st.line_chart(env_df.set_index("datetime")[["temperature", "humidity"]])
else:
    st.info("온습도 데이터가 없습니다")
 
# 결함 발생 시각 및 환경
st.header("5. 결함 발생 시각 및 환경")
if defect_df.empty:
    st.info("표시할 불량 데이터가 없습니다")
else:
    st.dataframe(
        defect_df[["datetime", "label", "confidence", "temperature", "humidity"]]
        .sort_values("datetime")
        .rename(columns={
            "datetime": "발생 시각",
            "label": "결함 유형",
            "confidence": "신뢰도",
            "temperature": "온도(°C)",
            "humidity": "습도(%)",
        }),
        use_container_width=True,
        hide_index=True,
    )
 
#온습도 구간별 불량률 - 임계값 탐색
st.header("6. 온도, 습도 구간별 불량률 추이")
st.caption("특정 온습도 구간에서 불량률이 튀어오르면 공정 관리 임계값 후보로 볼 수 있습니다")
if env_df.empty or env_df["status"].nunique() < 2:
    st.info("온습도 데이터가 부족해 상관관계를 찾을 수 없습니다")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**온도 구간별 불량률**")
        bins = min(5, env_df["temperature"].nunique())
        if bins >= 2:
            temp_bin = pd.cut(env_df["temperature"], bins=bins)
            rate = (env_df.assign(is_defect=(env_df["status"] == "불량").astype(int))
                    .groupby(temp_bin, observed=True)["is_defect"].mean() * 100)
            slim_bar_chart(rate, x_label="온도 구간", y_label="불량률(%)")
        else:
            st.info("온도 값의 다양성이 부족합니다.")
    with col2:
        st.write("**습도 구간별 불량률**")
        bins = min(5, env_df["humidity"].nunique())
        if bins >= 2:
            hum_bin = pd.cut(env_df["humidity"], bins=bins)
            rate = (env_df.assign(is_defect=(env_df["status"] == "불량").astype(int))
                    .groupby(hum_bin, observed=True)["is_defect"].mean() * 100)
            slim_bar_chart(rate, x_label="습도 구간", y_label="불량률(%)")
        else:
            st.info("습도 값의 다양성이 부족합니다")
 
#검사 처리량
st.header("7. 검사 처리량")
if total > 1:
    duration = df["datetime"].max() - df["datetime"].min()
    intervals = df["datetime"].sort_values().diff().dt.total_seconds().dropna()
    c1, c2 = st.columns(2)
    c1.metric("전체 소요 시간", str(duration).split(".")[0])
    c2.metric("평균 촬영 간격", f"{intervals.mean():.1f}초")
 
    per_min = (df.set_index("datetime").assign(count=1)["count"]
               .resample("1min").sum().rename("분당 검사 수"))
    st.line_chart(per_min)
else:
    st.info("처리량을 계산하기엔 데이터가 부족합니다")
 
#판정 신뢰도 분포
st.header("8. 판정 신뢰도 분포")
conf = df["confidence"].dropna()
if conf.nunique() > 1:
    counts, edges = np.histogram(conf, bins  =10, range = (0,1))
    hist_df = pd.DataFrame(
    {
        "신뢰도 구간": [
            f"{edges[i]:.2f} ~ {edges[i+1]:.2f}" for i in range(len(counts))],"빈도": counts,
    }).set_index("신뢰도 구간")
    st.bar_chart(hist_df)
else:
    st.info("신뢰도 값이 모두 동일 합니다. 실제 모델을 다시 확인하세요")
 
# 평균 신뢰도 및 신뢰도 낮은 판정 목록
st.header("9. 평균 신뢰도 및 저신뢰도 판정 목록")
if conf.empty:
    st.info("신뢰도 데이터가 없습니다")
else:
    low_conf_df = df[df["confidence"] < LOW_CONFIDENCE_THRESHOLD].dropna(subset=["confidence"])
 
    c1, c2 = st.columns(2)
    c1.metric("전체 평균 신뢰도", f"{conf.mean()*100:.1f}%")
    c2.metric(
        f"신뢰도 {int(LOW_CONFIDENCE_THRESHOLD*100)}% 이하 건수",
        f"{len(low_conf_df)} 개",
        f"{len(low_conf_df)/total*100:.1f}%",
        delta_color="inverse",
    )
 
    st.write(f"**신뢰도 {int(LOW_CONFIDENCE_THRESHOLD*100)}% 이하 판정 목록** (신뢰도 낮은 순)")
    if low_conf_df.empty:
        st.success(f"신뢰도 {int(LOW_CONFIDENCE_THRESHOLD*100)}% 이하인 판정이 없습니다")
    else:
        st.dataframe(
            low_conf_df[["datetime", "filename", "label", "confidence", "temperature", "humidity"]]
            .sort_values("confidence")
            .rename(columns={
                "datetime": "판정 시각",
                "filename": "파일명",
                "label": "판정 라벨",
                "confidence": "신뢰도",
                "temperature": "온도(°C)",
                "humidity": "습도(%)",
            }),
            use_container_width=True,
            hide_index=True,
        )
 
# 평균 신뢰도 및 신뢰도 낮은 판정 목록
st.header("9. 평균 신뢰도 및 저신뢰도 판정 목록")
if conf.empty:
    st.info("신뢰도 데이터가 없습니다")
else:
    low_conf_df = df[df["confidence"] < LOW_CONFIDENCE_THRESHOLD].dropna(subset=["confidence"])
 
    c1, c2 = st.columns(2)
    c1.metric("전체 평균 신뢰도", f"{conf.mean()*100:.1f}%")
    c2.metric(
        f"신뢰도 {int(LOW_CONFIDENCE_THRESHOLD*100)}% 이하 건수",
        f"{len(low_conf_df)} 개",
        f"{len(low_conf_df)/total*100:.1f}%",
        delta_color="inverse",
    )
 
    st.write(f"**신뢰도 {int(LOW_CONFIDENCE_THRESHOLD*100)}% 이하 판정 목록** (신뢰도 낮은 순)")
    if low_conf_df.empty:
        st.success(f"신뢰도 {int(LOW_CONFIDENCE_THRESHOLD*100)}% 이하인 판정이 없습니다")
    else:
        st.dataframe(
            low_conf_df[["datetime", "filename", "label", "confidence", "temperature", "humidity"]]
            .sort_values("confidence")
            .rename(columns={
                "datetime": "판정 시각",
                "filename": "파일명",
                "label": "판정 라벨",
                "confidence": "신뢰도",
                "temperature": "온도(°C)",
                "humidity": "습도(%)",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
st.caption(f"데이터 출처: {selected_csv} -마지막 갱신: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")