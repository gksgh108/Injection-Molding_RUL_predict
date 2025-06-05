import streamlit as st



def main():
    st.title("생산 시스템 구축 실무 프로젝트")
    st.image("https://ko.deepmould-de.com/Content/upload/2019379222/201910281735047998143.gif", caption="사출성형 공정 이미지", width=500)
    
    st.header("프로젝트 설명")
    st.write(""" 
        주요 내용
        - 데이터 수집 및 전처리
        - 정상 / 불량 라벨링
        - 머신러닝을 이용한 예측 모델 구축
        - 결과 시각화 및 분석
    """)

    st.subheader("팀원 소개")
    st.write("""
        - 김한호: 프로젝트 팀장
        - 송영재: 분석 팀장
        - 김정현: 개발 팀장
        - 이현준: 팀원
    """)

    st.subheader("프로젝트 목표")
    st.write("""
        - 데이터들을 활용하여 해당 shot이 정상인지 불량인지 구분하고자 한다.
        - 생산 효율성 극대화
        - 데이터 기반의 의사 결정 지원
    """)

if __name__ == "__main__":
    main()
    