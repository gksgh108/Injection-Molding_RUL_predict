import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
#######################
# Page configuration
st.set_page_config(
    page_title="사출 성형 예지보전",
    page_icon="🏭")

alt.themes.enable("dark")

#######################
# Custom CSS for dark mode
st.markdown(
    """
    <style>
    /* Set background color for the entire page */
    body {
        background-color: #121212;
        color: #e0e0e0;
    }

    /* Set background color for the sidebar */
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }

    /* Set text color for various elements */
    .st-bx, .st-ae, .st-af, .st-ag, .st-bz, .st-c0, .st-c1, .st-co {
        color: #e0e0e0;
    }

    /* Set background and text color for input elements */
    .st-ao, .st-bd {
        background-color: #333333;
        color: #e0e0e0;
    }

    /* Set background and text color for buttons */
    .st-cl, .st-cm {
        background-color: #444444;
        color: #e0e0e0;
    }

    /* Set background and text color for headers */
    .st-dg, .st-dh, .st-di, .st-dj, .st-dk {
        color: #e0e0e0;
    }

    /* Set border color for inputs and buttons */
    .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as, .st-at, .st-au {
        border-color: #555555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#######################
# Sidebar
with st.sidebar:
    st.title('🏭 사출 성형 예지보전')
    page = st.selectbox("원하는 페이지를 선택하세요", ["Main", "EDA", "Labeling", "Anomaly Detection", "Model"])

# 데이터를 한 번만 로드하고 세션 상태에 저장
if 'data_n_lot' not in st.session_state:
    @st.cache_data
    def load_data():
        data = pd.read_csv("../data/InjectionMolding_Raw_Data.csv")
        return data

    data = load_data()
    
    # 데이터 전처리
    data_drop = data.copy().drop(['Minimum_Cushion', 'Weighing_Start_Position', '_ID'], axis=1)

    def to_lot(data):
        lot_list = [0, data.shape[0]]
        for i in range(data.shape[0]):
            if data['No_Shot'][i] == 0:
                lot_list.append(i)
        lot_list.sort()
        data_lot = [data.iloc[lot_list[i]:lot_list[i+1]] for i in range(len(lot_list)-1)]
        return data_lot

    data_lot = to_lot(data_drop)

    def error_drop(data_lot):
        return [lot for lot in data_lot if lot.shape[0] >= 100]

    data_n_lot = error_drop(data_lot)
      
    st.session_state.data_n_lot = data_n_lot

if page == "Main":
    with open("main.py", encoding="utf-8") as file:
        exec(file.read())
elif page == "EDA":
    with open("eda.py", encoding="utf-8") as file:
        exec(file.read())
elif page == "Labeling":
    with open("labeling.py", encoding="utf-8") as file:
        exec(file.read())
elif page == "Anomaly Detection":
    with open("anomaly_detection.py", encoding="utf-8") as file:
        exec(file.read())
elif page == "Model":
    with open("model.py", encoding="utf-8") as file:
        exec(file.read())
