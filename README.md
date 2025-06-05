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
