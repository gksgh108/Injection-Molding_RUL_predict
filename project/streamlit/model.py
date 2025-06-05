import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression

# 데이터 로드 및 전처리
data_shot_label = pd.read_csv("../data/InjectionMolding_Labeled_Data.csv")

# 전처리 및 스케일링
scaler = MinMaxScaler()
df_sum_drop_scaler = scaler.fit_transform(data_shot_label.drop(columns=['PassOrFail']))  # PassOrFail 열을 제외한 데이터로 스케일링
df_sum_drop_scaler = pd.DataFrame(df_sum_drop_scaler, columns=data_shot_label.drop(columns=['PassOrFail']).columns)
df_sum_drop_scaler['PassOrFail'] = data_shot_label['PassOrFail']  # 다시 PassOrFail 열 추가

df_Y_scaler = df_sum_drop_scaler[df_sum_drop_scaler['PassOrFail'] == 0].drop(columns=['PassOrFail'])
df_N_scaler = df_sum_drop_scaler[df_sum_drop_scaler['PassOrFail'] == 1].drop(columns=['PassOrFail'])
Y_train, Y_test = train_test_split(df_Y_scaler, test_size=0.3, random_state=34)

# AutoEncoder 모델 정의 및 학습 함수
def train_autoencoder():
    encoder = Sequential([
        Dropout(0.3),
        Dense(15, activation="relu"),
        Dense(5, activation="relu")
    ])
    decoder = Sequential([
        Dense(15, activation="relu", input_shape=[5]),
        Dense(Y_train.shape[1], activation='relu'),
    ])
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    history = autoencoder.fit(Y_train, Y_train, batch_size=30, epochs=30, validation_split=0.2,
                              callbacks=[EarlyStopping(monitor="val_loss", patience=7, mode="min")])
    autoencoder.save('autoencoder_model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def load_autoencoder():
    autoencoder = load_model('autoencoder_model.h5', compile=False)
    autoencoder.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])  # 모델 컴파일 추가
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return autoencoder, scaler

def evaluate_autoencoder(autoencoder, scaler):
    Y_train_scaled = Y_train  # PassOrFail 열 제외한 데이터로 변환
    Y_test_scaled = Y_test  # PassOrFail 열 제외한 데이터로 변환
    df_N_scaler_scaled = df_N_scaler  # PassOrFail 열 제외한 데이터로 변환

    train_pred = autoencoder.predict(Y_train_scaled)
    train_loss = np.mean(np.square(train_pred - Y_train_scaled), axis=1)
    threshold = np.mean(train_loss) + 5 * np.std(train_loss)

    predict_Y = autoencoder.predict(Y_test_scaled)
    test_Y_mse = np.mean(np.square(predict_Y - Y_test_scaled), axis=1)
    test_Y_anomalies = test_Y_mse > threshold

    predict_N = autoencoder.predict(df_N_scaler_scaled)
    test_N_mse = np.mean(np.square(predict_N - df_N_scaler_scaled), axis=1)
    test_N_anomalies = test_N_mse > threshold

    true = np.concatenate([np.zeros(len(test_Y_anomalies)), np.ones(len(test_N_anomalies))])
    prediction = np.concatenate([test_Y_anomalies, test_N_anomalies])

    accuracy = np.round(accuracy_score(true, prediction), 4)
    recall = np.round(recall_score(true, prediction, pos_label=0), 4)
    precision = np.round(precision_score(true, prediction, pos_label=0), 4)
    f1_score = np.round((2 * recall * precision / (recall + precision)), 4)

    # Confusion Matrix 및 성능 지표 출력
    st.write("Confusion Matrix:")
    # Confusion Matrix 계산
    cm1 = confusion_matrix(true, prediction)
    
    # Confusion Matrix 히트맵 시각화
    fig, ax = plt.subplots()
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    st.write('정확도:', accuracy)
    st.write('재현율:', recall)
    st.write('정밀도:', precision)
    st.write('f1-score:', f1_score)

def train_logistic_regression():
    X = df_sum_drop_scaler.drop(columns=['PassOrFail'])
    y = df_sum_drop_scaler['PassOrFail']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    with open('logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_logistic_regression():
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_logistic_regression(model):
    X_test = df_sum_drop_scaler.drop(columns=['PassOrFail'])  # df_sum_drop_scaler 데이터 프레임 기준으로 수정
    y_test = df_sum_drop_scaler['PassOrFail']  # df_sum_drop_scaler 데이터 프레임 기준으로 수정

    y_pred = model.predict(X_test)
    
    # 성능 지표 계산
    accuracy = np.round(accuracy_score(y_test, y_pred), 4)
    recall = np.round(recall_score(y_test, y_pred), 4)
    precision = np.round(precision_score(y_test, y_pred), 4)
    f1_score = np.round((2 * recall * precision / (recall + precision)), 4)
    
    # Confusion Matrix 및 성능 지표 출력
    st.write("Confusion Matrix:")
    # Confusion Matrix 계산
    cm = confusion_matrix(y_test, y_pred)
    
    # Confusion Matrix 히트맵 시각화
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    st.write('정확도:', accuracy)
    st.write('재현율:', recall)
    st.write('정밀도:', precision)
    st.write('f1-score:', f1_score)
    
# Streamlit 앱 제목 및 설명
st.title("모델링")
st.markdown("""
    버튼을 눌러 AutoEncoder 또는 Logistic Regression 모델을 학습시키고 결과를 확인하세요.
""")

st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 구분선 추가
st.markdown("<br>", unsafe_allow_html=True)
if st.button('데이터 스케일링 및 학습 준비', key="scaling_button"):
    st.write("### 데이터 스케일링 및 학습 준비 진행 중")
    # 스케일링 결과 출력
    st.write("데이터 스케일링 및 학습, 테스트 데이터 준비 완료")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)
# Streamlit 버튼 추가 및 동작 설정
st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
if st.button('데이터셋 미리보기', key="preview_button"):
    st.write("### 데이터셋 미리보기")
    st.dataframe(data_shot_label.head())
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 굵은 구분선 추가

st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
if st.button('데이터 분포 시각화', key="distribution_button"):
    st.write("### 데이터 분포")
    fig, ax = plt.subplots()
    sns.countplot(x='PassOrFail', data=data_shot_label, ax=ax)
    st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 굵은 구분선 추가

st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
if st.button('AutoEncoder 평가', key="autoencoder_button"):
    st.write("### AutoEncoder 실행중")

    # AutoEncoder 모델 로드
    autoencoder, scaler = load_autoencoder()
    evaluate_autoencoder(autoencoder, scaler)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 굵은 구분선 추가

st.markdown('<div class="custom-button-container">', unsafe_allow_html=True)
if st.button('Logistic Regression 평가', key="logistic_button"):
    st.write("### Logistic Regression 실행중")

    # Logistic Regression 모델 로드
    model = load_logistic_regression()
    evaluate_logistic_regression(model)
st.markdown('</div>', unsafe_allow_html=True)
