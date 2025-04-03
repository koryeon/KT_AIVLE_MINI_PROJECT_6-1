# KT_AIVLE_MINI_PROJECT_6
# 시계열 데이터 기반 상품 판매량 예측
  
## 프로젝트 개요

### 주제
시계열 데이터 기반 상품 판매량 예측

### 데이터 출처
자체 제작 데이터 (Tabular CSV 파일)

### 문제 유형
Regression

### 중점사항
1. 데이터 탐색과 전처리
2. 딥러닝 모델 구축
3. 기술 및 비즈니스 측면에서의 모델 성능 평가

---

## 프로젝트 배경 및 비즈니스 상황

### 배경
- 미국 전역에 매장을 보유한 유통회사 **K-Mart**는 상품별 재고 문제를 겪고 있음
- 주요 문제:
  - 수요가 많은 상품의 재고 부족
  - 수요가 적은 상품의 과재고
  - 현금 흐름 문제 발생
- 기존 판매량 기반 발주 시스템이 효과적이지 않음

### 새로운 해결책: AI 기반 수요량 예측 시스템
- 가장 매출이 높은 **44번 매장**을 대상으로 파일럿 프로젝트 진행
- 핵심 상품 3개 선정 후 수요 예측을 통한 최적 발주 시스템 검토

### 기본 비즈니스 절차
- 매장 업무 마감 후(매일 저녁 9시) 발주 담당자가 당일 및 최근 판매량을 기준으로 발주
- 상품별 리드타임에 따라 적절한 시점에 입고됨

---

## 데이터셋 소개

### 데이터 파일
- **sales_train.csv** / **sales_test.csv**: 판매 정보 데이터
- **orders_train.csv** / **orders_test.csv**: 매장 방문 고객 수 데이터
- **oil_price_train.csv** / **oil_price_test.csv**: 유가 데이터 (WTI 기준)
- **products.csv**: 상품 정보 데이터
- **stores.csv**: 매장 정보 데이터

### 주요 컬럼 설명
#### 판매 정보 (sales)
- `Date`: 날짜
- `Store_ID`: 매장 ID
- `Qty`: 판매 수량
- `Product_ID`: 상품 ID

#### 상품 정보 (products)
- `Product_ID`: 상품 ID
- `Product_Code`: 상품 코드
- `SubCategory`: 중분류
- `Category`: 대분류
- `LeadTime`: 발주 후 입고까지 소요 기간 (일 단위)
- `Price`: 상품 판매 가격 (달러)

#### 매장 정보 (stores)
- `Store_ID`: 매장 ID
- `City`: 매장 위치 도시
- `State`: 매장 위치 주
- `Store_Type`: 매장 크기 및 형태

#### 고객 방문 정보 (orders)
- `Date`: 날짜
- `Store_ID`: 매장 ID
- `CustomerCount`: 고객 방문 수

#### 유가 정보 (oil_price)
- `Date`: 날짜
- `WTI_Price`: 유가 (단위: 배럴당 달러)

---

## 프로젝트 목표 및 요구사항

### 목표
- **44번 매장**의 **3개 핵심 상품**에 대한 일별 판매량을 예측하여 최적의 재고 관리 수행

### 주요 요구사항
1. 매일 저녁 9시, 리드타임에 맞춰 판매량 예측
2. 3개 핵심 상품 대상
   - **DB001 (Beverage)** - Drink
   - **GA001 (Milk)** - Food
   - **FM001 (Agricultural products)** - Grocery
3. **평가 기준**: 일 평균 재고금액
   - `일 평균 재고액 = (일 기초재고 + 일 기말재고) / 2 * (판매가 50%)`
   - 기회손실 수량 0을 목표로 설정
   - 평가 기간: **2017-03-01 ~ 2017-03-31**

---

## 프로젝트 진행 절차

### 1️⃣ 데이터 탐색 및 가설 도출
- 시계열 패턴 분석 (라인 차트 활용)
- 판매량과 관련된 다양한 요인 비교 분석
- 고객 행동 및 외부 요인 고려 (예: 유가 변동, 지역 내 고객 방문 수 변화 등)

### 2️⃣ 데이터 전처리 및 베이스라인 모델 생성
- 데이터프레임 구성 및 전처리
  - 결측값 처리
  - 가변수화 (범주형 데이터 변환)
  - 스케일링 (정규화 및 표준화)
  - 데이터 분할 (마지막 60일을 검증셋으로 활용)
- **Baseline Model** (Dense Layer 1개)
  - 성능 지표: RMSE, MAE, MAPE, R2 Score
- **LSTM 초기 모델**
  - LSTM Layer 1개 + Dense Layer 1개

### 3️⃣ 모델링 및 비즈니스 평가
- 다양한 모델링 시도 및 성능 평가
  - **LSTM** (Layer 및 노드 수 조정)
  - **CNN 1D 기반 시계열 모델링**
- 성능 검증 후 최적 모델 2~3개 선정
- **비즈니스 평가** (재고 금액 및 안전 재고 최적화)
  - `inv_simulator()` 함수 활용하여 최적 안전재고 계산
  - 기회손실을 0으로 만드는 최소 안전재고 값 확인

---

## 주요 코드 예제

### 재고량 평가 함수 (비즈니스 평가)
```python
import pandas as pd

def inv_simulator(y, pred, safe_stock, price):
    temp = pd.DataFrame({'y': y.reshape(-1,), 'pred': pred.reshape(-1,).round()})
    # (이후 시뮬레이션 로직 구현)
    print(f'* 일평균 재고량: {AvgDailyStock:.2f}')
    print(f'* 일평균 재고금액: {AvgDailyStockAmt:.2f}')
    print(f'* 기회손실 수량: {lost_sum}')
    return inventory

# 비즈니스 평가 실행
result = inv_simulator(y_test.values, y_pred, 13525, 8)
```

---

## 사용 기술 스택
- **Python**
- **Pandas, NumPy** (데이터 처리 및 분석)
- **Matplotlib, Seaborn** (데이터 시각화)
- **Scikit-learn** (머신러닝 모델 개발 및 평가)
- **TensorFlow, PyTorch** (딥러닝 모델 구축)

---

## 기대 효과
- AI 기반 수요 예측으로 **재고 문제 해결** 및 **운영 최적화**
- 최적의 발주 시스템 구축을 통한 **현금 흐름 개선**
- 기회손실 없는 **안전재고 관리**로 **비용 절감**


