import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load data from session state
if 'data_n_lot' in st.session_state:
    data_n_lot = st.session_state.data_n_lot
    st.write("### 데이터를 정상적으로 불러왔습니다!")
else:
    st.error("No data found. Please make sure to load and preprocess the data on the main page.")
    st.stop()

# Ensure combined_error_list exists in session state
if 'combined_error_list' not in st.session_state:
    st.error("abnormal Lot가 없습니다. 순서대로 앞 페이지부터 실행해주세요")
    st.stop()

# Access combined_error_list
combined_error_list = st.session_state['combined_error_list']

# Load or create labeled data
def load_or_label_data(data_n_lot):
    # Label and concatenate data if not already stored
    n_Lot_list = []
    for i in range(len(data_n_lot)):
        a = i
        Lot_label = data_n_lot[i].copy()
        Lot_label.loc[:,['Lot']] = a
        n_Lot_list.append(Lot_label)

    data_lot_label = n_Lot_list[0]
    for i in range(1, len(n_Lot_list)):
        data_lot_label = pd.concat([data_lot_label, n_Lot_list[i]])

    data_lot_label = data_lot_label.reset_index(drop=True)

    # Save labeled data to pickle file
    with open("data_lot_label.pkl", "wb") as f:
        pickle.dump(data_lot_label, f)

    return data_lot_label

# Load labeled data from pickle file
def load_data_lot_label():
    with open("data_lot_label.pkl", "rb") as f:
        data_lot_label = pickle.load(f)
    return data_lot_label

# Check if pickle file exists, if not label and save it
if not os.path.exists("data_lot_label.pkl"):
    data_lot_label = load_or_label_data(data_n_lot)
else:
    data_lot_label = load_data_lot_label()

# Streamlit App
st.title("Anomaly Detection")

st.write(f'abnormal Lot : {combined_error_list}')
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

existing_lot_numbers = data_lot_label['Lot'].unique()
st.write(f'존재하는 Lot 번호 : {existing_lot_numbers}')

user_input_lots = st.text_input("원하는 Lot 번호를 쉼표로 구분하여 입력하세요", '').strip()
if user_input_lots == '':
    st.warning("Lot번호를 지정해주세요!")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

selected_lot_numbers = list(map(int, user_input_lots.split(',')))

# Filter data for selected Lot numbers
valid_lot_numbers = [lot for lot in selected_lot_numbers if lot in existing_lot_numbers]
filtered_data = data_lot_label[data_lot_label['Lot'].isin(valid_lot_numbers)]

# Function to create boxplot for selected columns
def create_boxplot(filtered_data, columns):
    fig, ax = plt.subplots(figsize=(15, 8))
    filtered_data.boxplot(column=columns, by='Lot', ax=ax)
    ax.set_xlabel('Lot Number')
    ax.set_ylabel('Values')
    ax.set_title(f'Boxplot of Columns: {", ".join(columns)}')
    st.pyplot(fig)

# Bootstrap 기반 신뢰 구간 계산 및 pickle 파일로 저장 및 불러오기
def calculate_and_save_bootstrap_bounds(lot_data, lot):
    # 파일 이름 설정
    lower_bound_file = f"lower_bound_{lot}.pkl"
    upper_bound_file = f"upper_bound_{lot}.pkl"

    # 만약 파일이 존재하는 경우 불러오기
    if os.path.exists(lower_bound_file) and os.path.exists(upper_bound_file):
        with open(lower_bound_file, "rb") as f:
            lower_bound = pickle.load(f)
        with open(upper_bound_file, "rb") as f:
            upper_bound = pickle.load(f)
        print(f"Loaded {lower_bound_file} and {upper_bound_file} for Lot {lot}")
    else:
        # Bootstrap 기반 신뢰 구간 계산
        n_iterations = 1000
        t_squared_bootstrap = np.zeros((n_iterations, len(lot_data)))

        for i in range(n_iterations):
            sample_indices = np.random.choice(lot_data.index, size=len(lot_data), replace=True)
            sample = lot_data.loc[sample_indices]
            sample_mean_vector = sample.mean(axis=0)
            sample_cov_matrix = np.cov(sample.T)
            sample_inv_cov_matrix = np.linalg.inv(sample_cov_matrix)

            # Hotelling's T-squared 계산 함수 (Bootstrap 내에서 재정의)
            def hotelling_t2(row, mean_vector=sample_mean_vector, inv_cov_matrix=sample_inv_cov_matrix):
                diff = row - mean_vector
                t_squared = np.dot(np.dot(diff, inv_cov_matrix), diff.T)
                return t_squared

            t_squared_sample = sample.apply(lambda row: hotelling_t2(row), axis=1)
            t_squared_bootstrap[i, :] = t_squared_sample

        # 신뢰 구간 계산
        confidence_level = 0.99
        lower_bound = np.percentile(t_squared_bootstrap, (1 - confidence_level) / 2 * 100, axis=0)
        upper_bound = np.percentile(t_squared_bootstrap, (1 + confidence_level) / 2 * 100, axis=0)

        # pickle 파일로 저장
        with open(lower_bound_file, "wb") as f:
            pickle.dump(lower_bound, f)

        with open(upper_bound_file, "wb") as f:
            pickle.dump(upper_bound, f)

        print(f"Saved {lower_bound_file} and {upper_bound_file} for Lot {lot}")

    return lower_bound, upper_bound

# Allow user to select columns for boxplot
columns = st.multiselect('Select columns for boxplot', filtered_data.columns.tolist(), default=['Machine_Cycle_Time'])
if columns:
    create_boxplot(filtered_data, columns)

# Button to execute anomaly detection
if st.button('Show Anomaly Detection Results'):
    # 결과를 저장할 데이터프레임
    residuals_df = pd.DataFrame(index=filtered_data.index)

    # 각 레이블에 대해 모델 학습 및 잔차 계산
    features = ['Machine_Cycle_Time', 'Cycle_Time', 'Barrel_Temp_Z1', 'Barrel_Temp_Z2', 'Barrel_Temp_Z3', 'Barrel_Temp_Z4', 'Hopper_Temp', 'Injection_Pressure_Real_Time', 'Screw_Position', 'Injection_Peak_Press', 'Max_Injection_Rate', 'Screw_Velocity', 'VP_Time', 'VP_Position', 'VP_Press', 'Plasticizing_Time', 'Plasticizing_Start_Position', 'Plasticizing_End_Position', 'Plasticizing_RPM', 'Cooling_Time', 'Back_Flow', 'Decompression_Time']

    # 전체 데이터를 사용하여 모델 학습
    for label in features:
        X = data_lot_label.drop(columns=[label, 'Lot'])
        y = data_lot_label[label]
        model = LinearRegression()
        model.fit(X, y)

        # 필터링된 데이터에 대해 예측 및 잔차 계산
        X_filtered = filtered_data.drop(columns=[label, 'Lot'])
        y_filtered = filtered_data[label]
        y_pred = model.predict(X_filtered)
        residuals = y_filtered - y_pred

        residuals_df[label] = residuals

    # 누적 잔차 계산
    cumulative_residuals = residuals_df.cumsum()

    # Hotelling's T-squared 통계량 계산
    mean_vector = cumulative_residuals.mean(axis=0)
    cov_matrix = np.cov(cumulative_residuals.T)

    # 공분산 행렬의 역행렬 계산
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Hotelling's T-squared 계산 함수
    def hotelling_t2(row, mean_vector, inv_cov_matrix):
        diff = row - mean_vector
        t_squared = np.dot(np.dot(diff, inv_cov_matrix), diff.T)
        return t_squared

    # 각 행에 대해 Hotelling's T-squared 계산
    cumulative_residuals['T_squared'] = cumulative_residuals.apply(hotelling_t2, axis=1, args=(mean_vector, inv_cov_matrix))

    # Hotelling's T-squared 값과 신뢰 구간 계산 결과를 pickle 파일로 저장
    t_squared_results = {
        'cumulative_residuals': cumulative_residuals,
        'mean_vector': mean_vector,
        'inv_cov_matrix': inv_cov_matrix,
    }
    with open("t_squared_results.pkl", "wb") as f:
        pickle.dump(t_squared_results, f)

    st.write("## Hotelling's T-squared Results")
    st.write(cumulative_residuals['T_squared'])

    # Lot 번호별로 T-squared 결과값을 구분하여 시각화
    unique_lots = filtered_data['Lot'].unique()
    
    # Lot 번호가 하나인 경우
    if len(unique_lots) == 1:
        lot = unique_lots[0]
        lot_indices = filtered_data[filtered_data['Lot'] == lot].index
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(cumulative_residuals.loc[lot_indices].index, cumulative_residuals.loc[lot_indices]['T_squared'], label=f'Lot {lot}')
        ax.set_xlabel('Index')
        ax.set_ylabel("Hotelling's T-squared")
        ax.set_title(f"Hotelling's T-squared over Time for Lot {lot}")
        ax.legend()
        st.pyplot(fig)

    # Lot 번호가 여러 개인 경우
    else:
        for lot in unique_lots:
            lot_indices = filtered_data[filtered_data['Lot'] == lot].index
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(cumulative_residuals.loc[lot_indices].index, cumulative_residuals.loc[lot_indices]['T_squared'], label=f'Lot {lot}')
            ax.set_xlabel('Index')
            ax.set_ylabel("Hotelling's T-squared")
            ax.set_title(f"Hotelling's T-squared for Lot {lot}")
            ax.legend()
            st.pyplot(fig)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

    # Bootstrap 기반 신뢰 구간 계산 및 RUL 커브 시각화
    for lot in unique_lots:
        lot_indices = filtered_data[filtered_data['Lot'] == lot].index
        lot_data = cumulative_residuals.loc[lot_indices]
    
        # 파일이 있으면 불러오고 없으면 계산 후 저장
        lower_bound, upper_bound = calculate_and_save_bootstrap_bounds(lot_data, lot)
    
        # RUL 커브 시각화
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(lot_data.index, lot_data['T_squared'], label=f'Lot {lot}')
        ax.fill_between(lot_data.index, lower_bound, upper_bound, color='b', alpha=0.2, label='Confidence Interval')
        ax.set_xlabel('Index')
        ax.set_ylabel("Hotelling's T-squared")
        ax.set_title(f"Hotelling's T-squared with Bootstrap confidence interval for {lot}")
    
        # 이상 감지 시각화
        anomalies = (lot_data['T_squared'] > upper_bound) | (lot_data['T_squared'] < lower_bound)
        ax.scatter(lot_data.index[anomalies], lot_data['T_squared'][anomalies], color='r', label='Anomalies')
    
        ax.legend()
        st.pyplot(fig)  # 여기서 plt.show() 대신 st.pyplot(fig)를 사용하면 Streamlit에서도 시각화 가능합니다.