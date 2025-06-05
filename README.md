# 사출 성형 예지보전 (Injection Molding Predictive Maintenance)

## 프로젝트 개요

본 프로젝트는 사출 성형 공정에서 발생하는 데이터를 분석하고 시각화하여, **예지 보전(Predictive Maintenance)** 시스템을 구축하는 것을 목표로 합니다. 특히 **RUL (Remaining Useful Life) 커브**를 활용한 이상 감지 기법을 적용하고, 그 결과를 **웹 대시보드** 형태로 구현합니다.

* **주요 기술**: Python, Jupyter Notebook, Linux, Jupyter Lab
* **주요 내용**:
    * 사출 성형 공정 데이터 분석 및 시각화
    * RUL 커브를 이용한 이상 감지 (Anomaly Detection)
    * 웹 기반 대시보드 (Dashboard) 구현을 통한 직관적인 결과 제공

---

## 필요한 패키지 (Dependencies)

프로젝트를 실행하기 위해 다음 Python 패키지들이 필요합니다. (기본적으로 설치되어 있는 `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras` 등은 제외)

* `streamlit`
* `hdbscan`
* `pickle`
* `altair` (Streamlit 대시보드 시각화에 필요)

설치 방법:
```bash
pip install streamlit hdbscan pickle altair
```
---
## **다운로드 해야할 파일**
github에서는 25mb이상은 업로드가 불가능하기에 mega로 필요한 파일을 업로드 

[**파일 다운로드**](https://mega.nz/folder/KD4lVaID#H02o0kC66XBJBlgN0jBOWw)

InjectionMolding_Raw_Data.csv -> data폴더

InjectionMolding_Labeled_Data.csv -> data폴더

data_lot_label.pkl -> streamlit폴더

## 실행 방법
1. Jupyter Notebook으로 실행하기
RUL_predict.ipynb 파일을 Jupyter Notebook 또는 Jupyter Lab에서 열어 순서대로 셀을 실행하며 분석 과정을 따라갈 수 있습니다. 이 방법을 통해 데이터 전처리, 모델 학습, 이상 감지 과정 등을 단계별로 상세하게 확인하고 실행할 수 있습니다.

2. 웹 대시보드로 실행하기 (Streamlit)
구현된 웹 대시보드를 통해 직관적으로 데이터 분석 및 이상 감지 결과를 확인할 수 있습니다.

- 터미널 (또는 명령 프롬프트) 열기

- 프로젝트의 streamlit 폴더가 있는 상위 디렉토리로 이동합니다. (예: cd /path/to/your/project)

- 다음 명령어를 입력하여 Streamlit 앱을 실행합니다:

```bash
streamlit run streamlit/app.py
```
명령어 실행 후, 자동으로 웹 브라우저가 열리면서 대시보드가 나타납니다.

대시보드의 사이드바를 이용하여 각 분석 단계 (EDA, Labeling, Anomaly Detection, Model)로 이동하면서 실행하면 됩니다.
