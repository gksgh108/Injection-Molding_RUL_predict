import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan

# 세션 상태에서 데이터 가져오기
if 'data_n_lot' in st.session_state:
    data_n_lot = st.session_state.data_n_lot
    st.write("### 데이터를 정상적으로 불러왔습니다!")
else:
    st.error("No data found. Please make sure to load and preprocess the data on the main page.")
    st.stop()

# CSV 파일 경로
csv_file_path = "../data/InjectionMolding_Labeled_Data.csv"

# CSV 파일에서 데이터 불러오기
labeled_data = pd.read_csv(csv_file_path)

# 세션에 데이터 저장하기
st.session_state.labeled_data = labeled_data

# Function to aggregate lots
def aggregate_lots(data_lot, func):
    return pd.concat([lot.apply(func) for lot in data_lot], axis=1).T

# Calculate aggregated statistics for each lot
lot_mean = aggregate_lots(data_n_lot, lambda x: x.mean())
lot_median = aggregate_lots(data_n_lot, lambda x: x.median())
lot_75 = aggregate_lots(data_n_lot, lambda x: x.quantile(0.75))
lot_25 = aggregate_lots(data_n_lot, lambda x: x.quantile(0.25))

# IQR Outlier detection
def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((data > upper_bound) | (data < lower_bound))

def outlier_search(lot_rep):
    rep_index = [outliers_iqr(lot_rep[col])[0] for col in lot_rep.columns]
    outlier_index = [i for i in range(len(lot_rep)) if sum([i in idx for idx in rep_index]) > 3]
    return outlier_index

# Function for DBSCAN clustering
def perform_dbscan(x, eps, min_samples):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(principal_df)
    principal_df['DBSCAN_cluster'] = db.labels_
    return principal_df

# Function for HDBSCAN clustering
def perform_hdbscan(x):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10).fit(principal_df)
    principal_df['HDBSCAN_cluster'] = hdb.labels_
    return principal_df

# Function for OPTICS clustering
def perform_optics(x):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    optics = OPTICS(min_samples=10, cluster_method='xi', xi=0.05, min_cluster_size=0.05).fit(principal_df)
    principal_df['OPTICS_cluster'] = optics.labels_
    return principal_df

# Function to label data
def label_data(data_n_lot, error_list):
    # Shot 데이터에 Lot 번호 라벨링
    n_Lot_list = []
    for i in range(len(data_n_lot)):
        a = i
        Lot_label = data_n_lot[i].copy()
        Lot_label.loc[:, ['Lot']] = a
        n_Lot_list.append(Lot_label)

    # concat 함수를 통한 데이터 결합
    data_lot_label = n_Lot_list[0]
    for i in range(1, len(n_Lot_list)):
        data_lot_label = pd.concat([data_lot_label, n_Lot_list[i]])

    data_lot_label = data_lot_label.reset_index(drop=True)

    # abnormal shot 라벨링
    data_lot_label.loc[:, ['PassOrFail']] = 0
    for i in range(len(data_lot_label)):
        if data_lot_label['Lot'][i] in error_list:
            data_lot_label.loc[i, ['PassOrFail']] = 1

    data_shot_label = data_lot_label.copy().drop(['No_Shot'], axis=1)
    data_shot_label = data_shot_label.copy().drop(['Lot'], axis=1)

    return data_shot_label

# Streamlit App
def main():
    st.title('이진분류 라벨링')

    # Interactive selection of clustering methods
    selected_methods = st.multiselect("Select Clustering Methods", ["IQR", "DBSCAN", "HDBSCAN", "OPTICS"])

    error_lists = {}

    if "IQR" in selected_methods:
        st.write("## IQR 완료!")
        IQR_error = sorted(list(set(outlier_search(lot_mean) + outlier_search(lot_median) + outlier_search(lot_75) + outlier_search(lot_25))))
        error_lists["IQR"] = IQR_error
        st.write("\n")
    if "DBSCAN" in selected_methods:
        x = lot_mean.drop(['Machine_Cycle_Time'], axis=1).values
        x = MinMaxScaler().fit_transform(x)
        principal_df_db = perform_dbscan(x, 0.165, 10)

        st.write("## DBSCAN Clustering")
        fig_db, ax_db = plt.subplots(figsize=(8, 8))
        for i in range(-1, principal_df_db['DBSCAN_cluster'].max() + 1):
            ax_db.scatter(principal_df_db.loc[principal_df_db['DBSCAN_cluster'] == i, 'PC1'],
                          principal_df_db.loc[principal_df_db['DBSCAN_cluster'] == i, 'PC2'], label=f'DBSCAN cluster {i}')
        ax_db.legend()
        st.pyplot(fig_db)

        DB_error = principal_df_db[principal_df_db['DBSCAN_cluster'].isin([-1, 4])].index.tolist()
        error_lists["DBSCAN"] = DB_error

    if "HDBSCAN" in selected_methods:
        x = lot_mean.drop(['Machine_Cycle_Time'], axis=1).values
        x = MinMaxScaler().fit_transform(x)
        principal_df_hdb = perform_hdbscan(x)

        st.write("## HDBSCAN Clustering")
        fig_hdb, ax_hdb = plt.subplots(figsize=(8, 8))
        for cluster in set(principal_df_hdb['HDBSCAN_cluster']):
            ax_hdb.scatter(principal_df_hdb.loc[principal_df_hdb['HDBSCAN_cluster'] == cluster, 'PC1'],
                           principal_df_hdb.loc[principal_df_hdb['HDBSCAN_cluster'] == cluster, 'PC2'], label=f'HDBSCAN cluster {cluster}')
        ax_hdb.legend()
        st.pyplot(fig_hdb)

        HDB_error = principal_df_hdb[principal_df_hdb['HDBSCAN_cluster'] == -1].index.tolist()
        error_lists["HDBSCAN"] = HDB_error

    if "OPTICS" in selected_methods:
        x = lot_mean.drop(['Machine_Cycle_Time'], axis=1).values
        x = MinMaxScaler().fit_transform(x)
        principal_df_optics = perform_optics(x)

        st.write("## OPTICS Clustering")
        fig_optics, ax_optics = plt.subplots(figsize=(8, 8))
        for klass in set(principal_df_optics['OPTICS_cluster']):
            ax_optics.plot(principal_df_optics.loc[principal_df_optics['OPTICS_cluster'] == klass, 'PC1'],
                           principal_df_optics.loc[principal_df_optics['OPTICS_cluster'] == klass, 'PC2'], 'o', label=f'OPTICS cluster {klass}')
        ax_optics.legend()
        st.pyplot(fig_optics)

        OPTICS_error = principal_df_optics[principal_df_optics['OPTICS_cluster'] == -1].index.tolist()
        error_lists["OPTICS"] = OPTICS_error

    # Store combined_error_list in session_state
    combined_error_list = sorted(set().union(*[set(errors) for errors in error_lists.values()]))
    st.session_state['combined_error_list'] = combined_error_list

    # Display the list of abnormal lots for each selected method
    for method, error_list in error_lists.items():
        st.write(f"## {method}이 판정한 비정상 Lot")
        st.write(f'비정상 Lot 번호: {error_list}')
        st.write(f'비정상 Lot 개수: {len(error_list)}')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Retrieve combined_error_list from session_state
    if 'combined_error_list' in st.session_state:
        combined_error_list = st.session_state['combined_error_list']
    
        st.write(f"## 비정상으로 판별한 Lot 번호")
        st.write(f'abnormal Lot : {combined_error_list}')
    else:
        st.warning("비정상으로 판별한 Lot 번호를 찾을 수 없습니다.")

    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 굵은 구분선 추가

    # Button to label data
    if st.button('라벨링 데이터 생성하기'):
        st.write("라벨링 데이터 생성하는 중 ...")
        labeled_data = label_data(data_n_lot, combined_error_list)
        st.session_state.labeled_data = labeled_data
        st.write("라벨링 데이터 생성완료")

    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 굵은 구분선 추가

    # Button to show first few rows of labeled data
    if st.button('라벨링 데이터 보기'):
        if 'labeled_data' in st.session_state:
            st.write("라벨링된 데이터:")
            st.write(st.session_state.labeled_data.head())
        else:
            st.write("라벨링된 데이터가 없습니다. 먼저 데이터를 라벨링하세요.")

    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 굵은 구분선 추가

    # Button to show PassOrFail value counts
    if st.button('이진분류 값 개수'):
        if 'labeled_data' in st.session_state:
            st.write("이진분류 값의 개수:")
            st.write(st.session_state.labeled_data['PassOrFail'].value_counts())
        else:
            st.write("라벨링된 데이터가 없습니다. 먼저 데이터를 라벨링하세요.")

    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 굵은 구분선 추가

    # Button to label data and save to CSV
    if st.button('라벨링 데이터 저장하기'):
        csv_file_path = "../data/InjectionMolding_Labeled_Data.csv"
        labeled_data = st.session_state.labeled_data  # 세션 상태에서 labeled_data 가져오기
    
        if labeled_data is not None:  # labeled_data가 None이 아닌 경우에만 처리
            labeled_data.to_csv(csv_file_path, index=False)
            st.write("라벨링 데이터 저장완료")
        else:
            st.write("저장할 데이터가 없습니다. 먼저 데이터를 라벨링하세요.")

    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)  # 굵은 구분선 추가

if __name__ == "__main__":
    main()

