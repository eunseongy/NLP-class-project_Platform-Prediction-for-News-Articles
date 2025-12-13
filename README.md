# 뉴스 기사 유입 플랫폼 예측 모델
(News Referrer Prediction with Text & Demographics)

## 프로젝트 개요

본 프로젝트는 뉴스 기사 콘텐츠의 텍스트 정보와 독자 인구통계 정보를 기반으로, 해당 기사가 어떤 플랫폼(referrer)을 통해 유입될 가능성이 높은지를 예측하는 모델을 개발하는 것을 목표로 한다.
기사의 제목·요약 텍스트, 연령·성별 기반 독자 분포, 그리고 조회수 기반 soft label을 활용하여 플랫폼별 유입 확률을 출력하는 다중 클래스 분류 모델을 설계·학습하였다.

## 데이터셋 설명

본 연구에서는 데이콘(Dacon)에서 주최한 '2025 신문과 방송 독자 데이터 분석 아이디어 경진대회'에서 제공된 공식 데이터셋을 사용하였다.

### 데이터 구성

원본 데이터는 다음과 같은 복합 구조로 이루어져 있다.

- 뉴스 기사 본문 텍스트

- 월별 조회수 정보

- 유입 경로(referrer) 정보

- 연령·성별 기반 인구통계(demographics) 데이터

본 프로젝트에서는 모델 학습에 필요한 항목을 선별·정제하여 하나의 통합 분석용 데이터셋으로 재구성하였다.

### 데이터 사용 및 라이선스 

데이콘(Dacon) 대회 규정에 따라 본 데이터는 아래 조건하에서 사용 가능하다.

허용:
- 대회 참여
- 학술 연구
- 교육 목적
- 비영리 프로젝트

제한:
- 상업적 이용
- 원본 데이터 재배포

본 GitHub의 모든 실험은 대회 규정 및 비영리 목적을 준수하여 수행되었다.

## Method
본 연구에서는 긴 뉴스 기사 본문을 처리하기 위해 두 가지 접근 방식을 비교·검토하였다. 하나는 긴 문서를 직접 인코딩하는 방식(BigBird 기반)이며, 다른 하나는 요약을 통해 입력 길이를 축소한 후 예측 모델에 입력하는 방식이다.
전체 모델 구조는 그림 1과 그림 2에 각각 제시되어 있다.

### Long-Sequence Encoding with Chunk Aggregation (Baseline)

그림 1은 긴 뉴스 본문을 직접 인코딩하기 위한 기존 접근 방식을 나타낸다.

뉴스 기사 본문은 일반적으로 Transformer 모델의 최대 입력 길이를 초과하므로, 본문을 고정 길이 토큰 단위(예: 0–3999, 4000–7999) 로 분할하여 여러 개의 chunk로 구성한다.
각 chunk는 동일한 인코더(BigBird)를 통해 개별적으로 임베딩되며, 이후 모든 chunk representation을 aggregation layer에서 결합하여 최종 문서 표현을 생성한다.

이 방식은 원문 정보를 최대한 보존할 수 있다는 장점이 있으나, 모델 파라미터 수가 매우 크고 연산 비용이 높으며 학습 및 추론 시간이 길고 한국어 특화가 부족한 경우 성능이 제한됨이라는 한계를 가진다. 실험 결과에서도 긴 인코더 구조가 항상 성능 향상으로 이어지지는 않음을 확인하였다.

<img width="561" height="307" alt="bigbird_input 설명" src="https://github.com/user-attachments/assets/7e9ead35-2a3d-45d3-87f8-7dd0492c4eb0" />

### Two-Stage Summarization for Long News Articles

그림 2는 본 연구에서 제안한 요약 기반 입력 축소 전략을 보여준다.

KoBART 요약 모델은 최대 입력 길이가 1024 tokens로 제한되어 있기 때문에, 긴 뉴스 본문을 단일 입력으로 요약하는 것은 불가능하다. 이를 해결하기 위해 2단계 요약(X-Summarization) 구조를 적용하였다.

(1) Local Summarization
- 뉴스 본문을 1024 token 단위로 chunking
- 각 chunk를 KoBART 요약 모델에 입력하여 부분 요약(Local summary) 생성
- 정보 손실을 방지하기 위해 최소 요약 길이(min_length ≥ 200 tokens)를 강제

(2) Global Summarization
- 생성된 모든 local summary를 하나의 텍스트로 병합
- 다시 KoBART 요약 모델을 적용하여 최종 요약(Global summary) 생성
- 최종 요약 단계에서는 min_length ≥ 300 tokens로 설정하여 문맥 보존 강화

이 구조를 통해 긴 뉴스 문서의 핵심 정보를 유지하면서도 후속 예측 모델이 처리 가능한 길이의 텍스트를 생성할 수 있다.

<img width="539" height="657" alt="요약모델 설명 drawio" src="https://github.com/user-attachments/assets/e0f54b57-1b1c-4547-99c4-a7182edce03f" />

### Final Prediction Model with Demographics-Aware Input

최종 요약된 텍스트는 KoELECTRA 기반 분류 모델의 입력으로 사용된다.
KoELECTRA는 최대 입력 길이가 512 tokens로 제한되어 있으나,
요약 기반 전처리를 통해 이 제약을 효과적으로 우회할 수 있다.

모델 입력은 다음과 같이 구성된다.

```
[DEMO] 연령_성별 [TITLE] 기사 제목 [TEXT] 요약 텍스트
```

이를 통해 모델은 단순 텍스트 정보뿐 아니라 독자 인구통계(demographics) 정보를 함께 고려하여 플랫폼(referrer) 예측을 수행할 수 있다.
최종적으로 모델은 7개 플랫폼 클래스에 대해 확률 분포 형태의 출력을 생성한다.

### 예시

#### input
```
age_group = "10대"
gender = "여"
title = "10대 여성 소비 트렌드: 편의점과 패션의 변화"
summary = "10대 여성의 소비 패턴을 분석하고 주요 플랫폼별 반응을 정리했다."
```
#### output
```
1. Google: 0.3958
2. 네이버: 0.1487
3. 네이버 블로그: 0.1391
4. Daum: 0.1054
5. AI 검색엔진: 0.0972
6. 기타: 0.0588
7. Bing: 0.0550
```

모델은 입력 텍스트에 대해 플랫폼별 유입 확률 분포를 출력하며, 가장 높은 확률의 플랫폼을 최종 예측값으로 사용한다

### 향후 개선 방향

- likes, comments 등 사용자 반응 feature 추가

- multi-input 모델 구조 확장

- 요약 품질 개선 및 요약 길이 실험

- API 형태의 서비스 적용 가능성 검토

## 참고 자료
- Dacon: [https://dacon.io](https://dacon.io/competitions/official/236606/overview/description)
- HuggingFace Transformers(KoELECTRA, KoBART, Ko-bigbird)

## Email
dmstjd1542@gmail.com

