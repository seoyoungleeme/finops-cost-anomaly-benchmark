# FOCUS 실제 데이터 기반 FinOps 비용 이상 탐지 실험 보고서

작성일: 2026-05-21  
프로젝트: `finops-cost-anomaly-benchmark`  
연구 주제: 클라우드 비용 시계열 이상 탐지를 위한 비용 가중 평가 프레임워크

## 1. 요약

본 연구는 클라우드 비용 이상 탐지 모델을 단순한 분류 성능이 아니라 FinOps 운영 의사결정 관점에서 비교하기 위한 benchmark를 구축한다. 기존 순수 합성 데이터 실험은 정답 라벨을 명확히 제공한다는 장점이 있지만, 실제 클라우드 청구 데이터와의 연결성이 약하다는 한계가 있다. 이를 보완하기 위해 본 실험에서는 FinOps Foundation의 FOCUS Sample Data를 사용하여 실제 billing 데이터의 비용 규모, 추세, 변동성, 요일 패턴을 추출하고, 그 통계량으로 synthetic baseline을 보정한 뒤 통제된 anomaly를 주입했다.

핵심 결론은 다음과 같다.

- 실제 FOCUS full sample을 로드하여 5,488,359 rows의 비용 데이터를 확보했다.
- FOCUS 원자료에는 anomaly ground truth label이 없으므로, F1, recall, detection delay, MCTD 같은 정량 성능 평가는 raw FOCUS 자체가 아니라 FOCUS-calibrated synthetic benchmark에서 수행했다.
- raw FOCUS 데이터는 rolling z-score 기반 sanity check로 사용하여 실제 비용 시계열에서도 의심 날짜가 탐지되는지 확인했다.
- full FOCUS strict benchmark를 5개 모델(EWMA, SeasonalNaiveMAD, IsolationForest, LSTM_AE, Prophet)로 재실행한 결과, Prophet이 F1과 cost-weighted recall에서 가장 우수했다.
- EWMA와 SeasonalNaiveMAD는 alert cost efficiency가 높게 보일 수 있지만, dollar recall과 MCTD/CTDR이 낮아 비용 손실을 조기에 잡는 목적에는 취약했다.
- 평가 지표가 F1, dollar recall, MCTD, alert cost efficiency 중 무엇인지에 따라 모델 순위가 달라졌고, 이는 본 연구의 핵심 주장인 "FinOps에서는 표준 분류 지표만으로 모델을 선택하면 부족하다"를 뒷받침한다.
- 최종 full strict run은 anomaly type × intensity 이벤트 수를 균형화하고, calibration clipping saturation과 normalized cost-to-detect 지표를 산출물에 기록했다.

본 연구에서 가장 방어 가능한 표현은 다음과 같다.

> 본 연구는 FOCUS 실제 청구 데이터를 직접 labeled benchmark로 사용하지 않는다. FOCUS 데이터에는 anomaly 정답 라벨이 없기 때문이다. 대신 FOCUS에서 실제 비용 패턴 통계량을 추출하여 benchmark baseline을 보정하고, 통제된 anomaly injection으로 평가 가능한 ground truth를 확보했다. 따라서 본 실험은 "FOCUS-calibrated labeled benchmark"이며, raw FOCUS 결과는 정량 성능 평가가 아니라 sanity check로 해석한다.

## 2. 연구 배경 및 필요성

### 2.1 클라우드 비용 이상 탐지가 중요한 이유

클라우드 비용은 전통적인 고정 IT 비용과 달리 사용량 기반으로 빠르게 변한다. 인스턴스 수, GPU 사용, 데이터 전송량, managed database I/O, storage request, cross-region traffic, batch job, autoscaling 정책 같은 운영 요소가 곧바로 비용으로 연결된다. 따라서 작은 설정 오류나 사용량 변화도 며칠 안에 큰 비용 손실로 이어질 수 있다.

예를 들어 다음과 같은 상황은 모두 비용 anomaly로 나타날 수 있다.

- 실험 후 GPU instance를 종료하지 않은 경우
- autoscaling upper bound가 잘못 설정되어 필요 이상으로 instance가 증가한 경우
- 데이터 전송 job이 반복 실행되어 egress 비용이 증가한 경우
- log ingestion, monitoring, object storage request가 갑자기 늘어난 경우
- 할인, credit, refund, commitment 적용 방식 때문에 비용 series가 음수 또는 급변 형태로 나타나는 경우
- 신규 서비스 출시, 마케팅 이벤트, migration 작업처럼 정상적인 business event가 비용 급등으로 보이는 경우

이 문제는 단순히 "비용이 높다"의 문제가 아니라 탐지 지연의 문제다. 클라우드 비용은 시간이 지날수록 누적되기 때문에, 동일한 anomaly라도 하루 만에 발견하는 것과 2주 뒤에 발견하는 것은 실제 손실 규모가 다르다. 따라서 FinOps 비용 이상 탐지에서는 다음 질문이 중요하다.

- anomaly를 탐지했는가?
- 큰 비용 impact를 가진 anomaly를 우선적으로 잡았는가?
- anomaly 시작 후 얼마나 빨리 잡았는가?
- 탐지 전까지 얼마의 비용이 이미 손실되었는가?
- alert가 너무 많아 운영자가 무시하게 되지는 않는가?
- alert 하나당 실제로 어느 정도 비용 impact를 포착했는가?

FinOps Framework의 Anomaly Management capability도 unexpected 또는 unforecasted cloud cost event를 timely하게 detect, identify, clarify, alert, manage하는 것을 강조한다. 또한 성공 지표로 anomaly count, alert와 관련된 spend, time to detect, anomaly duration, actioned anomaly, avoided spend 등을 제시한다. 이는 본 연구가 precision/recall/F1뿐 아니라 cost-weighted recall, MCTD, alert cost efficiency를 함께 보는 이유와 직접 연결된다. 출처: [FinOps Framework - Managing Anomalies](https://www.finops.org/framework/previous-capabilities/manage-anomalies/), [Managing Cloud Cost Anomalies](https://www.finops.org/wg/managing-cloud-cost-anomalies/).

### 2.2 기존 이상 탐지 평가가 FinOps에 충분하지 않은 이유

일반적인 anomaly detection 연구에서는 모델을 precision, recall, F1, AUPRC 같은 표준 분류 지표로 평가하는 경우가 많다. 이 지표들은 모델이 label을 얼마나 잘 맞혔는지 보여준다는 점에서 유용하다. 그러나 FinOps 비용 anomaly에서는 label 개수 기준 성능과 실제 운영 가치가 항상 일치하지 않는다.

예를 들어 두 모델을 비교해보자.

- 모델 A: 작은 anomaly 10개를 모두 잡았지만, 큰 비용 사고 1개를 놓침
- 모델 B: 작은 anomaly 몇 개는 놓쳤지만, 큰 비용 사고를 시작 직후 탐지함

point-wise F1만 보면 모델 A가 더 좋아 보일 수 있다. 하지만 실제 FinOps 운영에서는 모델 B가 더 유용할 가능성이 크다. 비용 anomaly detection의 목적은 anomaly label을 많이 맞히는 것만이 아니라, 비용 손실을 줄이고, 대응 우선순위를 정하고, alert fatigue를 관리하는 것이기 때문이다.

따라서 본 연구는 다음과 같은 비용 중심 지표를 포함한다.

- cost-weighted recall: 전체 anomaly cost 중 탐지한 비용의 비율
- MCTD: anomaly 시작부터 최초 탐지 전까지 누적된 비용 손실
- alert cost efficiency: alert 1개당 포착한 anomaly cost

이 지표들은 "모델이 얼마나 정확한가"뿐 아니라 "운영자가 이 모델을 썼을 때 비용 손실을 얼마나 줄일 수 있는가"를 평가한다.

### 2.3 공개 labeled cloud cost benchmark가 부족한 이유

클라우드 비용 이상 탐지 연구에서 가장 큰 어려움은 공개 labeled billing dataset이 부족하다는 점이다. 실제 cloud billing data는 기업의 resource 사용량, 서비스 구조, 고객 트래픽, 지역, account 구조, 할인 계약, commitment 사용량 등을 포함할 수 있어 민감하다. 또한 비용 급등이 실제 anomaly인지 정상 business event인지는 조직 내부 맥락 없이는 판단하기 어렵다.

FOCUS처럼 공개된 billing sample data가 있어도, 그 안에는 anomaly ground truth label이 없다. 즉 특정 날짜의 비용 변화가 실제 장애, misconfiguration, 테스트, migration, 신규 서비스 론칭, discount correction 중 무엇인지 알 수 없다. 따라서 raw billing data만으로 precision, recall, F1을 계산하면 연구 주장이 약해진다.

이 한계 때문에 본 연구는 다음 전략을 선택한다.

```text
실제 billing data의 통계량은 사용한다.
하지만 성능 평가 label은 통제된 anomaly injection으로 만든다.
```

이 방식은 순수 합성 데이터보다 현실 비용 패턴과 더 연결되어 있고, raw real data만 사용할 때보다 평가 가능성이 높다.

## 3. 선행연구 및 관련 시스템

### 3.1 FinOps anomaly management

FinOps Foundation은 cloud cost anomaly management를 FinOps 운영 역량의 일부로 다룬다. FinOps 관점에서 anomaly는 정상 또는 기대 spend와 다른 비용 event이며, 이를 timely하게 detect하고 alert하며 triage하는 것이 중요하다. 특히 anomaly의 비용 impact, time to detect, resolution duration, avoided spend 같은 운영 지표를 강조한다.

본 연구와의 연결점은 명확하다. FinOps anomaly management가 단순히 "비정상 점을 찾는 것"이 아니라 "비용 impact와 대응 우선순위를 관리하는 것"이라면, anomaly detection model 평가도 F1 하나에 머물 수 없다. 따라서 본 연구는 cost-weighted recall, MCTD, alert cost efficiency를 핵심 지표로 포함한다.

관련 출처:

- [FinOps Framework - Managing Anomalies](https://www.finops.org/framework/previous-capabilities/manage-anomalies/)
- [FinOps Working Group - Managing Cloud Cost Anomalies](https://www.finops.org/wg/managing-cloud-cost-anomalies/)

### 3.2 Cloud provider의 비용 anomaly detection 기능

주요 cloud provider들은 이미 비용 anomaly detection 기능을 제공한다. 이는 cloud cost anomaly detection이 실제 운영에서 중요한 문제임을 보여준다.

AWS Cost Anomaly Detection은 machine learning model을 사용해 deployed AWS services의 anomalous spend pattern을 detect하고 alert하는 기능이다. 사용자는 cost monitor와 alert subscription을 설정해 특정 account, service, cost category, tag 등 scope에서 이상 비용을 받을 수 있다. 출처: [AWS Cost Anomaly Detection](https://docs.aws.amazon.com/cost-management/latest/userguide/manage-ad.html), [Getting started with AWS Cost Anomaly Detection](https://docs.aws.amazon.com/cost-management/latest/userguide/getting-started-ad.html).

Google Cloud Billing의 Anomalies 기능은 billing account의 project 비용에서 expected spend와 다른 spike 또는 deviation을 보여주며, cost impact와 deviation threshold를 기준으로 anomaly view와 notification을 관리할 수 있다. 또한 root cause analysis panel에서 service, region, SKU 등 원인 후보를 제공한다. 출처: [Google Cloud - View and manage cost anomalies](https://docs.cloud.google.com/billing/docs/how-to/manage-anomalies).

Microsoft Cost Management도 FinOps 도구의 일부로 anomaly detection을 제공하며, Cost Analysis smart views에서 normalized usage 기반 anomaly를 확인할 수 있다고 설명한다. 출처: [Microsoft Learn - Overview of Cost Management](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/overview-cost-management).

그러나 provider-native 도구들은 다음 한계를 가진다.

- 내부 탐지 알고리즘이 충분히 공개되지 않는다.
- provider별 scope, cost definition, threshold policy가 달라 공정 비교가 어렵다.
- anomaly type별 성능, cost-to-detect, alert efficiency를 연구자가 통제해서 비교하기 어렵다.
- 공개 재현 가능한 labeled benchmark를 제공하지 않는다.

따라서 provider tool의 존재는 본 연구의 필요성을 약화시키기보다 강화한다. 실제 현업 수요는 존재하지만, 모델 선택과 지표 비교를 위한 독립적이고 재현 가능한 benchmark가 부족하기 때문이다.

### 3.3 FOCUS와 billing data 표준화

FOCUS는 FinOps Open Cost and Usage Specification의 약자로, 기술 billing data를 일관된 column과 terminology로 표현하기 위한 open specification이다. FOCUS는 cloud, SaaS, data center 등 다양한 technology vendor의 billing dataset을 정규화해 FinOps 분석의 복잡성을 낮추는 것을 목표로 한다.

본 연구에서 FOCUS가 중요한 이유는 다음과 같다.

- 특정 provider의 독자 billing schema에 종속되지 않는다.
- `ProviderName`, `ServiceCategory`, `ServiceName`, `EffectiveCost`, `BilledCost`, `ChargePeriodStart` 같은 비용 분석 핵심 필드를 제공한다.
- AWS, Microsoft, Oracle 등 여러 provider의 sample data를 포함한다.
- 공개 저장소를 통해 재현 가능한 데이터 기반 실험이 가능하다.

관련 출처:

- [FOCUS official site](https://focus.finops.org/)
- [What is FOCUS?](https://focus.finops.org/what-is-focus/)
- [FOCUS Specification](https://focus.finops.org/focus-specification/)
- [FOCUS Sample Data GitHub repository](https://github.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data)

단, FOCUS는 billing data schema와 sample data를 제공하지만 anomaly label을 제공하지 않는다. 따라서 본 연구는 FOCUS를 "정답 label source"가 아니라 "realistic billing baseline source"로 사용한다.

버전 주의. 본 실험은 `FOCUS-Sample-Data/FOCUS-1.0` sample을 사용했다. 2026년 5월 현재 FOCUS specification은 FOCUS 1.3까지 공개되어 있으며, 최신 specification은 contract commitments, split cost allocation, completeness/recency 같은 항목을 확장한다. 따라서 본 결과는 FOCUS 1.0 sample 기반 calibration 결과로 한정해야 하며, FOCUS 1.3 sample 또는 실제 조직의 FOCUS-exported billing data로 확장 검증하는 것은 후속 과제다. 출처: [FOCUS Specification](https://focus.finops.org/focus-specification/), [FinOps Foundation - Introducing FOCUS 1.3](https://www.finops.org/insights/introducing-focus-1-3/).

### 3.4 일반 anomaly detection benchmark와의 차이

Numenta Anomaly Benchmark(NAB), Exathlon 같은 public time-series anomaly benchmark는 anomaly detection 연구에서 많이 사용된다. NAB는 real-time anomaly detector를 평가하기 위한 labeled time-series benchmark와 scoring mechanism을 제공한다. Exathlon은 Apache Spark cluster 실행 trace에 fault를 주입해 explainable anomaly detection benchmark를 구성한다.

관련 출처:

- [Numenta Anomaly Benchmark GitHub](https://github.com/numenta/NAB)
- Lavin and Ahmad, "Evaluating Real-time Anomaly Detection Algorithms - the Numenta Anomaly Benchmark" ([arXiv:1510.03336](https://arxiv.org/abs/1510.03336))
- Jacob et al., "Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series" ([arXiv:2010.05073](https://arxiv.org/abs/2010.05073))

이 benchmark들은 anomaly detection 방법론 평가에는 유용하지만, 본 연구 주제와는 차이가 있다.

- billing cost data가 아니다.
- cloud cost allocation, service category, provider, billing correction 같은 FinOps context가 없다.
- 비용 impact와 alert cost efficiency를 직접 평가하기 어렵다.
- FinOps 운영자가 중요하게 보는 spend impact, time-to-detect, avoided cost와 직접 연결되지 않는다.

따라서 본 연구는 일반 anomaly benchmark를 그대로 사용하는 대신 FOCUS billing sample을 활용해 FinOps domain에 맞는 benchmark를 구성한다.

### 3.5 본 연구의 차별성

선행연구와 기존 시스템 대비 본 연구의 차별성은 다음과 같다.

| 구분 | 기존 접근 | 본 연구 |
|---|---|---|
| 데이터 | 일반 time-series 또는 provider 내부 데이터 | 공개 FOCUS billing sample 기반 calibration |
| 라벨 | 공개 billing data에는 보통 없음 | 통제된 anomaly injection으로 확보 |
| 평가 지표 | precision, recall, F1 중심 | F1 + dollar recall + MCTD + alert efficiency |
| 재현성 | provider tool은 내부 알고리즘 비공개 | 코드, seed, metadata, output 저장 |
| FinOps 맥락 | 비용 impact 반영이 약함 | cost impact와 detection delay를 직접 반영 |

즉 본 연구는 새로운 anomaly detector를 제안하는 연구라기보다, FinOps 비용 이상 탐지에서 모델을 어떻게 평가하고 선택해야 하는지에 대한 benchmark 및 평가 프레임워크 연구다.

## 4. 사용한 실제 데이터

### 4.1 데이터 출처

사용한 데이터는 FinOps Open Cost and Usage Specification 조직의 공식 공개 저장소인 `FOCUS-Sample-Data`이다.

- GitHub organization: <https://github.com/orgs/FinOps-Open-Cost-and-Usage-Spec/repositories>
- FOCUS Sample Data repository: <https://github.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data>
- FOCUS 1.0 sample folder: <https://github.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/tree/main/FOCUS-1.0>
- FOCUS 소개: <https://focus.finops.org/what-is-focus/>

FOCUS는 여러 cloud provider의 비용 및 사용량 데이터를 공통 schema로 정규화하기 위한 open specification이다. 따라서 이 데이터는 특정 vendor 형식에 묶인 임의 CSV가 아니라, FinOps 커뮤니티에서 논의되는 표준 청구 데이터 포맷에 맞춘 sample data라는 점에서 연구 맥락과 잘 맞는다.

### 4.2 로컬에 확보한 데이터 파일

실험 과정에서 다음 FOCUS 파일을 로컬 cache에 저장했다.

| 파일 | 로컬 경로 | 용도 |
|---|---|---|
| 10k sample | `.focus_cache/focus_sample_10000.csv` | 초기 실험용. 현재 최종 주장에는 보조적 의미만 있음 |
| 100k sample | `.focus_cache/focus_sample_100000.csv.gz` | broader relaxed benchmark |
| full sample | `.focus_cache/focus_data_table.csv.gz` | full strict benchmark 및 raw sanity check |

특히 full sample은 GitHub raw URL로 받으면 Git LFS pointer만 내려오기 때문에, 실제 gzip 파일을 `media.githubusercontent.com` 경로에서 다운로드했다. 최종 full sample의 로컬 크기는 496,305,507 bytes이며, pandas로 로드한 결과 5,488,359 rows였다.

### 4.3 full sample의 기본 특성

full FOCUS sample 로드 결과는 다음과 같다.

| 항목 | 값 |
|---|---:|
| rows | 5,488,359 |
| columns | 44 |
| date range | 2024-03-20 to 2024-09-30 |
| unique billing days | 32 |
| 주요 provider rows | AWS 5,181,336 / Microsoft 270,492 / Oracle 36,416 / Google Cloud 115 |

이 인벤토리는 `scripts/build_focus_inventory.py`가 캐시된 FOCUS 파일에서 직접 산출한 결과이며, `outputs/focus_data_inventory.json`에 그대로 기록돼 있어 재현 검증이 가능하다. 동일한 데이터 인벤토리 블록이 각 실행의 `focus_run_metadata.json`의 `data_inventory` 필드에도 함께 저장된다.

Google Cloud row는 full sample 안에 존재하지만 115 rows로 매우 적어, 본 실험의 service group 필터를 통과하지 못했다. 따라서 최종 분석에 실제로 남은 provider 축은 AWS, Microsoft, Oracle이다.

**시간 해상도 주의 (sparsity disclosure).** `date_min=2024-03-20`, `date_max=2024-09-30`은 195일 캘린더 구간이지만 `unique_billing_days=32`이다. 즉 FOCUS sample은 매일 dense하게 누적되는 청구 데이터가 아니라 sparse한 청구 스냅샷 집합에 가깝다. 그 결과 service group별 `days_observed`도 최대 30일 수준이고, 한 요일이 관측 구간에 등장하는 횟수는 평균 4-5회 정도다. 이 sparsity는 calibration 단계의 `weekly_factor` 추정 정밀도를 떨어뜨릴 수 있어, 본 연구는 service-specific weekly factor를 global과 50/50 blending하고 `[0.25, 2.50]`로 clipping하여 보수적으로 처리했다 (Section 6.3). 그럼에도 매우 짧고 sparse한 관측 구간의 한계는 calibration 결과에 잔여 노이즈로 남는다는 점을 명시한다.

## 5. 데이터 사용의 정당성

### 5.1 왜 FOCUS를 사용했는가

본 연구의 주제는 FinOps 비용 이상 탐지이다. 따라서 일반적인 센서 데이터, 네트워크 침입 탐지 데이터, 웹 트래픽 데이터보다 클라우드 비용 및 사용량 billing 데이터가 연구 질문에 더 직접적으로 맞는다. FOCUS는 cloud cost and usage 데이터를 provider-independent schema로 정규화하려는 표준이며, 연구에서 다음 장점을 제공한다.

- 실제 cloud billing domain의 column 구조를 가진다.
- `ProviderName`, `ServiceCategory`, `ServiceName`, `EffectiveCost`, `BilledCost`, `ChargePeriodStart` 등 FinOps 분석에 필요한 필드를 포함한다.
- AWS, Microsoft, Oracle 등 여러 provider의 비용 분포를 포함한다.
- 공개 데이터이므로 연구 재현성이 높다.
- 특정 기업의 민감한 private billing data 없이도 실제 billing-like distribution을 반영할 수 있다.

### 5.2 왜 raw FOCUS만으로 benchmark하지 않았는가

raw FOCUS 데이터에는 "이 날짜가 실제 anomaly다"라는 ground truth label이 없다. 따라서 raw FOCUS만으로는 다음 지표를 정당하게 계산할 수 없다.

- precision
- recall
- F1
- AUPRC
- event recall
- detection delay
- cost-weighted recall
- MCTD
- alert cost efficiency

예를 들어 실제 비용이 튄 날짜가 있어도 그것이 정상적인 batch job인지, discount correction인지, credit/refund인지, 실제 misconfiguration인지 알 수 없다. 따라서 raw FOCUS에서 alert를 냈다는 사실만으로 "맞았다" 또는 "틀렸다"를 판정할 수 없다.

이 때문에 본 연구는 raw FOCUS를 다음처럼 제한적으로 사용한다.

- 정량 benchmark: FOCUS 통계량으로 보정한 synthetic baseline + injected labeled anomalies
- 정성 sanity check: raw FOCUS daily cost series에 rolling z-score detector 적용

이 설계는 실제 데이터의 realism과 실험 라벨의 통제 가능성을 동시에 확보하기 위한 절충안이다.

### 5.3 본 연구에서 주장 가능한 범위

주장 가능한 내용은 다음과 같다.

- 본 연구는 공식 FOCUS public sample data를 사용했다.
- full sample 기준 5,488,359 rows를 로드했고, service-level daily cost series로 집계했다.
- 실제 FOCUS 데이터에서 추출한 비용 수준, 추세, 변동성, 요일 패턴으로 benchmark baseline을 보정했다.
- 평가 라벨은 synthetic anomaly injection으로 확보했기 때문에 F1, recall, MCTD 등 정량 평가가 가능하다.
- raw FOCUS 시계열에서도 rolling z-score 기준 suspicious cost dates를 탐색했다.

주장하면 안 되는 내용은 다음과 같다.

- raw FOCUS 데이터 자체가 labeled anomaly benchmark라고 말하면 안 된다.
- raw FOCUS alert 결과를 precision/recall/F1 성능으로 해석하면 안 된다.
- FOCUS 결과가 모든 실제 cloud billing 환경으로 일반화된다고 단정하면 안 된다.
- 30일 내외의 관측 구간에서 장기 seasonality를 충분히 검증했다고 말하면 안 된다.

## 6. 데이터 처리 과정

### 6.1 FOCUS 로딩

FOCUS 데이터 로딩은 `finops_benchmark/focus_loader.py`에서 수행한다.

주요 처리 과정은 다음과 같다.

1. CSV 또는 CSV.GZ 파일을 pandas로 로드한다.
2. 날짜 column으로 `ChargePeriodStart`를 사용한다.
3. 비용 column은 `EffectiveCost`를 우선 사용하고, 없으면 `BilledCost`를 사용한다.
4. 내부 표준 비용 column으로 `_eff_cost`를 만든다.
5. `ChargePeriodStart`를 UTC datetime으로 파싱하고, 일 단위 `_date`로 정규화한다.
6. `_date`, provider/service grouping columns, `_eff_cost`를 기준으로 daily cost series를 만든다.

본 연구의 기본 group key는 다음과 같다.

```text
ProviderName, ServiceCategory
```

이 기준은 service-level 분석보다 약간 넓지만, 짧은 sample 기간에서 너무 많은 sparse group이 생기는 것을 막는다. `ProviderName,ServiceName`도 검토했지만, full strict 기준에서는 6개 group, relaxed 기준에서는 37개 group이 살아남아 더 세밀한 대신 sparsity와 해석 부담이 커진다.

### 6.2 필터링 기준

service group은 다음 조건을 통과해야 분석에 포함했다.

full strict calibrated benchmark:

```text
min_days = 21
min_nonzero_days = 14
min_mean_cost = 1.0
```

100k relaxed calibrated benchmark 및 raw sanity check:

```text
min_days = 14
min_nonzero_days = 7
min_mean_cost = 0.1
```

strict 기준은 더 안정적인 group만 남긴다. relaxed 기준은 실제 데이터 다양성을 더 많이 살리지만, 관측 기간이 짧거나 비용 규모가 작은 group이 포함될 수 있어 해석을 조심해야 한다.

### 6.3 FOCUS calibration

FOCUS daily series에서 추출한 calibration parameters는 다음과 같다.

| parameter | 의미 |
|---|---|
| `base_level` | 평균 일별 비용 수준 |
| `monthly_growth` | 선형 추세를 월간 증가율로 환산한 값 |
| `noise_pct` | de-trended residual의 coefficient of variation |
| `weekly_factor` | Monday to Sunday 요일별 비용 multiplier |

짧은 관측 기간 때문에 과적합을 줄이기 위해 다음 clipping을 적용했다.

| parameter | clipping range |
|---|---:|
| `monthly_growth` | -0.02 to 0.10 |
| `noise_pct` | 0.01 to 0.15 |
| `weekly_factor` | 0.25 to 2.50 |

또한 service-specific weekly factor는 global weekly factor와 50/50으로 blending했다. 이는 30일 내외의 짧은 FOCUS sample에서 특정 요일이 우연히 과도하게 반영되는 것을 줄이기 위한 보수적 처리이다.

최종 full strict run에서는 raw 추정값과 clipped 값을 함께 저장했다. `outputs/results_full_strict/focus_calibration_stats.csv`에는 `monthly_growth_raw`, `monthly_growth_saturated`, `noise_pct_raw`, `noise_pct_saturated`가 포함되며, cross-service 요약은 `focus_calibration_summary.csv`에 저장된다. 최종 결과에서는 4/4 service group의 `monthly_growth`와 `noise_pct`가 모두 clipping bound에 걸렸다. 즉 calibration은 FOCUS sample의 짧고 sparse한 관측치를 그대로 외삽하지 않고, 보수적 bound로 눌러 synthetic baseline을 생성한다. 이 때문에 본 연구의 realism은 "FOCUS에서 관측된 비용 수준과 패턴을 반영하되, trend/noise magnitude는 보수적으로 제한한 realism"으로 해석해야 한다.

### 6.4 calibrated synthetic benchmark 생성

`finops_benchmark/data.py`의 `build_focus_calibrated_dataset()`은 다음 순서로 데이터를 만든다.

1. FOCUS에서 추출한 `base_level`, `monthly_growth`, `noise_pct`, `weekly_factor`를 사용한다.
2. 730일 daily cost baseline을 생성한다.
3. Year 1은 정상 학습 구간으로 둔다.
4. Year 2에만 synthetic anomalies를 주입한다.
5. anomaly type, intensity, event id, excess cost, cost impact를 label로 저장한다.

이 방식의 핵심은 baseline의 형태는 실제 FOCUS 비용 통계에 맞추되, anomaly label은 완전히 통제한다는 점이다.

### 6.5 anomaly 유형

주입한 anomaly는 세 유형이다.

| anomaly type | 의미 |
|---|---|
| `spike` | 1-2일의 단기 비용 급등 |
| `contextual` | 전체 절대 비용은 크지 않지만 요일/맥락상 비정상적인 비용 |
| `gradual` | 7-14일 동안 점진적으로 비용이 증가하는 이상 |

각 anomaly는 `low`, `mid`, `high` intensity로 주입되며, event-level label과 cost impact가 저장된다. 최종 full strict run은 `--n-events-per-combo 3`을 사용해 service × seed × anomaly type × intensity 조합마다 3개 이벤트를 주입했다. 따라서 primary setting에서는 각 type × intensity cell이 36 events(4 services × 3 seeds × 3 events)로 균형을 이룬다. 이는 이전 run에서 gradual-mid/high 이벤트 수가 부족했던 문제를 줄이기 위한 보수적 수정이다. metadata의 `injection_balance_warnings`가 비어 있으면, 최종 산출물 기준으로 큰 event imbalance가 없다는 뜻이다.

## 7. 모델과 평가 프로토콜

### 7.1 비교 모델

본 연구는 다섯 가지 대표 탐지 패러다임을 비교했다.

| 모델 | 패러다임 | score 의미 |
|---|---|---|
| EWMA | 통계 기반 residual z-score | EWMA baseline 대비 이탈 정도 |
| SeasonalNaiveMAD | 실무형 robust seasonal baseline | 같은 요일 과거 median 대비 robust deviation |
| IsolationForest | tree 기반 비지도 학습 | calendar, lag, rolling feature 기반 이상도 |
| LSTM_AE | reconstruction 기반 deep learning | 정상 패턴 재구성 오차 |
| Prophet | forecasting residual 기반 | trend/seasonality forecast 대비 residual |

모든 모델은 `score`가 클수록 anomaly 가능성이 높도록 통일했다.

#### EWMA

EWMA는 Exponentially Weighted Moving Average의 약자이며, 최근 관측값에 더 큰 가중치를 두고 과거 관측값의 영향은 지수적으로 감소시키는 방식이다. 통계적 공정관리에서는 EWMA control chart가 작은 shift나 점진적 변화를 감지하는 데 사용되어 왔다. NIST Engineering Statistics Handbook도 EWMA statistic이 최근 측정값을 포함한 과거 data의 exponentially weighted average에 기반한다고 설명한다. 출처: [NIST - EWMA Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm).

본 연구에서 EWMA는 단순하고 해석 가능한 statistical baseline 역할을 한다. 비용 series의 local trend를 부드럽게 추정한 뒤, 실제 비용이 baseline에서 얼마나 벗어났는지를 residual z-score로 계산한다. 장점은 구현이 간단하고 계산 비용이 낮으며 운영자가 이해하기 쉽다는 점이다. 단점은 요일 context, service-specific pattern, complex seasonality를 명시적으로 모델링하지 못한다는 점이다. 따라서 spike에는 반응할 수 있지만 contextual anomaly나 gradual anomaly에는 취약할 수 있다.

#### SeasonalNaiveMAD

SeasonalNaiveMAD는 같은 요일의 과거 비용을 robust baseline으로 사용하는 실무형 baseline이다. 각 날짜의 비용을 이전 같은 요일 관측치들의 median과 비교하고, median absolute deviation(MAD)로 scale을 맞춘 robust z-score를 anomaly score로 사용한다. 이 방식은 복잡한 학습 없이도 weekly billing pattern을 반영할 수 있고, 운영자가 설명하기 쉽다. 다만 history가 짧거나 sparse하면 같은 요일 샘플 수가 부족해지고, 점진적 변화나 level shift에는 둔감할 수 있다. 본 연구에서는 provider-native black-box 도구 대신, 현업에서 구현 가능한 간단한 seasonal rule baseline으로 포함했다.

#### IsolationForest

Isolation Forest는 Liu, Ting, Zhou가 제안한 unsupervised anomaly detection 방법이다. 핵심 아이디어는 정상 point보다 anomaly point가 random partitioning 과정에서 더 적은 split만으로 고립되기 쉽다는 것이다. 즉 feature space에서 쉽게 isolate되는 관측치를 anomaly로 본다. 출처: Liu et al., "Isolation Forest" ([IEEE ICDM 2008 / Monash record](https://research.monash.edu/en/publications/isolation-forest/)), scikit-learn [IsolationForest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

본 연구에서 IsolationForest는 calendar feature, lag feature, rolling mean, rolling standard deviation, lag ratio 등을 사용한다. 이 모델은 선형 forecast model이 아니기 때문에 복잡한 feature interaction을 포착할 수 있고, 요일과 lag 기반 context를 활용해 contextual anomaly에 강할 수 있다. 다만 threshold calibration에 민감하고, feature engineering 방식에 따라 성능이 달라질 수 있다. 또한 anomaly score의 해석이 EWMA나 Prophet보다 직관적이지 않을 수 있다.

#### LSTM Autoencoder

LSTM Autoencoder는 정상 sequence를 압축하고 다시 복원하도록 학습한 뒤, reconstruction error가 큰 sequence를 anomaly로 보는 방식이다. Malhotra et al.은 LSTM encoder-decoder가 정상 time-series behavior를 reconstruct하도록 학습하고 reconstruction error를 anomaly detection에 사용할 수 있음을 보였다. 출처: Malhotra et al., "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection" ([arXiv:1607.00148](https://arxiv.org/abs/1607.00148)).

본 연구에서 LSTM_AE는 sequential pattern을 학습하는 deep learning 기반 baseline이다. 장점은 비선형 temporal dependency를 포착할 수 있고, 점진적 패턴 변화나 복잡한 sequence anomaly에서 강점을 보일 수 있다는 점이다. 단점은 데이터가 짧으면 학습 안정성이 낮아질 수 있고, hyperparameter와 random seed에 민감하며, 운영자가 score를 설명하기 어렵다는 점이다. 또한 alert 수가 많아지면 operational alert fatigue 문제가 생길 수 있다.

#### Prophet

Prophet은 Taylor와 Letham이 제안한 forecasting framework로, trend, seasonality, holiday effect 등을 additive model 형태로 결합한다. 원 논문은 business forecasting을 scale 있게 수행하기 위한 practical forecasting approach를 설명한다. 출처: Taylor and Letham, "Forecasting at Scale" ([The American Statistician, 2018](https://www.tandfonline.com/doi/abs/10.1080/00031305.2017.1380080)), [Prophet paper PDF](https://facebook.github.io/prophet/static/prophet_paper_20170113.pdf).

본 연구에서 Prophet은 비용 시계열의 trend와 weekly seasonality를 forecast하고, 실제 비용과 예측 비용의 residual을 anomaly score로 사용한다. 장점은 클라우드 비용처럼 trend와 weekly pattern이 있는 series에 잘 맞을 수 있고, forecast residual이 비교적 해석 가능하다는 점이다. 단점은 갑작스러운 regime change, 매우 짧은 history, sparse service group에서는 forecast가 불안정할 수 있다는 점이다. 본 실험에서는 FOCUS-calibrated baseline의 trend/seasonality와 anomaly injection 구조가 Prophet의 강점과 잘 맞아 F1과 cost-weighted recall에서 높은 성능을 보였다.

#### 다섯 모델을 함께 비교하는 이유

다섯 모델은 서로 다른 탐지 패러다임을 대표한다.

| 패러다임 | 대표 모델 | 비교 이유 |
|---|---|---|
| 단순 통계 baseline | EWMA | 운영 친화적이고 해석 가능하지만 복잡한 context에는 약할 수 있음 |
| 실무형 seasonal robust baseline | SeasonalNaiveMAD | 요일별 median/MAD로 구현이 쉽고 설명 가능 |
| feature-based unsupervised learning | IsolationForest | calendar/lag/rolling feature를 사용해 contextual pattern을 포착 가능 |
| sequence reconstruction | LSTM_AE | 비선형 temporal dependency와 gradual change 탐지 가능성 |
| forecasting residual | Prophet | trend/seasonality가 강한 비용 series에 적합 |

따라서 본 연구는 "어떤 단일 모델이 최고인가"보다 "평가 지표와 anomaly 유형에 따라 모델 선택이 어떻게 달라지는가"를 보는 데 초점을 둔다.

### 7.2 leakage-free protocol

평가 protocol은 다음과 같다.

1. Year 1에는 anomaly를 주입하지 않는다.
2. 모델 학습과 threshold calibration은 Year 1만 사용한다.
3. anomaly는 Year 2에만 존재한다.
4. 모든 성능 지표는 Year 2에서만 계산한다.
5. AUPRC는 binary alert가 아니라 raw anomaly score 기반으로 계산한다.

threshold는 모델별 score scale 차이를 맞추기 위해 Year 1 score percentile로 정한다.

| year1 false alarm target | percentile |
|---:|---:|
| 0.5% | 99.5 |
| 1.0% | 99.0 |
| 2.0% | 98.0 |

paper-ready primary setting은 1.0% Year-1 false alarm target이다.

운영 환산. Year 1의 정상 일수가 365이므로 1% Year-1 FAR은 service group 1개당 연 평균 약 3.65건의 false alert에 해당한다. 100개의 service group을 운영하는 조직이라면 false alert만으로 연 약 365건 (≈ 매일 1건)이 발생한다는 의미이며, 실 운영에서는 triage 비용·alert fatigue를 고려해 budget 선택을 조정해야 한다.

### 7.3 평가 지표

표준 지표:

- precision
- recall
- F1
- AUPRC
- false alarm rate

event-level 지표:

- event recall
- mean detection delay

FinOps 비용 가중 지표:

- cost-weighted recall: 전체 anomaly cost 중 탐지한 비용 비율
- mean MCTD: mean cost-to-detect, 탐지 전까지 누적된 비용 손실
- cost-to-detect ratio: 전체 anomaly cost 대비 탐지 전 손실 비용 비율
- avoided cost ratio: `1 - cost-to-detect ratio`; anomaly cost 중 탐지로 회피한 비용 비율의 근사값
- alert cost efficiency: alert 1개당 포착한 anomaly cost

FinOps 관점에서는 F1 하나만으로 모델을 고르면 부족하다. 예를 들어 alert 수가 적어 precision이 높아도 큰 비용 anomaly를 늦게 잡으면 운영상 좋지 않다. 반대로 F1은 낮아도 큰 비용 이벤트를 빨리 잡는 모델이 실무적으로 더 유용할 수 있다. `mean_mctd`는 비용 단위라 service 규모의 영향을 받으므로, 최종 산출물에는 scale-normalized 지표인 `cost_to_detect_ratio`와 `avoided_cost_ratio`도 함께 저장했다.

## 8. 실행한 실험

### 8.1 100k FOCUS relaxed calibrated benchmark

목적: service group 다양성을 늘린 보조 benchmark

실행 설정:

```powershell
python scripts/run_focus_benchmark.py `
  --url https://raw.githubusercontent.com/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/main/FOCUS-1.0/focus_sample_100000.csv.gz `
  --group-by ProviderName,ServiceCategory `
  --min-days 14 `
  --min-nonzero-days 7 `
  --min-mean-cost 0.1 `
  --n-seeds 5 `
  --output-dir outputs/results `
  --figure-dir outputs/figures `
  --n-events-per-combo 3
```

metadata:

| 항목 | 값 |
|---|---:|
| rows | 100,000 |
| service groups | 12 |
| seeds | 0, 1, 2, 3, 4 |
| budgets | 0.5%, 1.0%, 2.0% |
| compared models | 5 |
| event balance | 3 events per service × seed × type × intensity |
| runtime | 334.5 sec |
| output dir | `outputs/results` |
| figures | `outputs/figures` |

포함된 service groups:

- AWS / Compute
- AWS / Databases
- AWS / Other
- AWS / Storage
- Microsoft / AI and Machine Learning
- Microsoft / Compute
- Microsoft / Databases
- Microsoft / Networking
- Microsoft / Other
- Microsoft / Storage
- Microsoft / Web
- Oracle / Compute

주의점:

- 12개 group 중 7개 group은 관측일이 21일 미만이라 global fallback calibration을 사용했다.
- 5개 service-specific calibrated group 중 5개 모두 `monthly_growth`와 `noise_pct`가 clipping bound에 도달했다. 나머지 7개 fallback group은 global calibration을 사용했다.
- 따라서 100k relaxed 결과는 다양성 확인에는 유용하지만, service-specific conclusion은 조심해야 한다.

### 8.2 full FOCUS strict calibrated benchmark

목적: full public FOCUS sample을 사용한 더 보수적인 핵심 robustness benchmark

실행 설정:

```powershell
python scripts/run_focus_benchmark.py `
  --url https://media.githubusercontent.com/media/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/main/FOCUS-1.0/focus_data_table.csv.gz `
  --group-by ProviderName,ServiceCategory `
  --min-days 21 `
  --min-nonzero-days 14 `
  --min-mean-cost 1.0 `
  --n-seeds 3 `
  --output-dir outputs/results_full_strict `
  --figure-dir outputs/figures_full_strict `
  --n-events-per-combo 3
```

metadata:

| 항목 | 값 |
|---|---:|
| rows | 5,488,359 |
| service groups | 4 |
| seeds | 0, 1, 2 |
| budgets | 0.5%, 1.0%, 2.0% |
| compared models | 5 |
| event balance | 3 events per service × seed × type × intensity |
| runtime | 249.3 sec |
| output dir | `outputs/results_full_strict` |
| figures | `outputs/figures_full_strict` |

strict 기준으로 살아남은 service groups:

| service | days observed | nonzero days | base level | monthly growth | noise pct | fallback |
|---|---:|---:|---:|---:|---:|---|
| AWS / Compute | 30 | 19 | 156.4667 | 0.10 | 0.15 | False |
| AWS / Databases | 30 | 21 | 18.5000 | -0.02 | 0.15 | False |
| AWS / Other | 30 | 30 | 54.0427 | 0.10 | 0.15 | False |
| Oracle / Compute | 30 | 15 | 24.6667 | 0.10 | 0.15 | False |

4개 group 모두 `used_fallback=False`이며 (`focus_calibration_stats.csv`), 같은 사실이 `focus_run_metadata.json`의 `n_fallback_services: 0`에도 명시돼 있다. 이 실험은 fallback 없이 모든 group이 service-specific calibration을 사용했다는 점에서 100k relaxed 결과(`n_fallback_services: 7`)보다 보수적이고 방어 가능하다. 단, 최종 run에서는 4/4 group의 `monthly_growth`와 `noise_pct`가 clipping bound에 도달했으므로 trend/noise magnitude는 FOCUS raw estimate를 그대로 외삽한 것이 아니라 보수적으로 제한한 값이다.

### 8.3 raw full FOCUS sanity check

목적: 정답 라벨 없이 실제 FOCUS daily cost series에서 rolling z-score detector가 어떤 날짜를 flag하는지 확인

실행 설정:

```powershell
python scripts/run_focus_unsupervised.py `
  --url https://media.githubusercontent.com/media/FinOps-Open-Cost-and-Usage-Spec/FOCUS-Sample-Data/main/FOCUS-1.0/focus_data_table.csv.gz `
  --group-by ProviderName,ServiceCategory `
  --min-days 14 `
  --min-nonzero-days 7 `
  --min-mean-cost 0.1 `
  --window 7 `
  --sigma 2.5 `
  --output-prefix focus_unsupervised_full_relaxed `
  --figure-dir outputs/figures_full_strict
```

결과 파일:

- `outputs/results/focus_unsupervised_full_relaxed_alerts.csv`
- `outputs/results/focus_unsupervised_full_relaxed_summary.csv`

raw sanity check에서는 18개 service group이 살아남았다. 이 결과는 성능 평가가 아니라 실제 비용 데이터에서 detector가 의심 날짜를 찾아낼 수 있는지 확인하기 위한 보조 자료다.

추가로 raw FOCUS case-study plot 2개를 생성했다.

- `outputs/figures_full_strict/focus_unsupervised_full_relaxed_case_AWS___Compute.png`
- `outputs/figures_full_strict/focus_unsupervised_full_relaxed_case_Microsoft___Networking.png`

## 9. 결과

### 9.1 full FOCUS strict benchmark 전체 모델 순위

Primary setting: Year-1 FAR target = 1%. 값은 `mean ± std`이며, std는 service × seed (총 4 × 3 = 12 cell) pooled 표준편차이다. std 컬럼은 `outputs/results_full_strict/focus_overall_model_ranking_with_std.csv`에 함께 저장돼 있다.

| model | F1 | F1 rank | cost-weighted recall | dollar recall rank | alert cost efficiency | ACE rank | mean MCTD | MCTD rank | avoided cost ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prophet | 0.5798 ± 0.0820 | 1 | 0.9498 ± 0.0762 | 1 | 104.40 ± 100.85 | 4 | 19.66 ± 22.64 | 2 | 0.9063 ± 0.1054 |
| IsolationForest | 0.4353 ± 0.1232 | 2 | 0.8765 ± 0.1647 | 3 | 170.83 ± 176.03 | 3 | 44.50 ± 49.79 | 3 | 0.8252 ± 0.1730 |
| LSTM_AE | 0.3833 ± 0.1149 | 3 | 0.9076 ± 0.1447 | 2 | 43.49 ± 39.18 | 5 | 19.47 ± 15.68 | 1 | 0.8678 ± 0.1703 |
| SeasonalNaiveMAD | 0.2242 ± 0.0917 | 4 | 0.5966 ± 0.1660 | 4 | 487.52 ± 498.82 | 2 | 210.70 ± 233.59 | 4 | 0.5114 ± 0.1381 |
| EWMA | 0.1773 ± 0.0354 | 5 | 0.5154 ± 0.0786 | 5 | 511.36 ± 555.28 | 1 | 245.75 ± 256.17 | 5 | 0.4590 ± 0.0740 |

해석:

- Prophet은 F1과 cost-weighted recall에서 1위다. 다만 이 결과는 FOCUS-calibrated synthetic benchmark에서의 우위이며, Prophet의 trend/weekly seasonality 가정이 본 benchmark의 baseline 생성 구조와 잘 맞는다는 점을 함께 해석해야 한다.
- LSTM_AE는 mean MCTD가 가장 낮고 avoided cost ratio도 2위다. 즉 전체 F1은 Prophet보다 낮지만, 탐지된 이벤트에서 비용 손실을 빠르게 줄이는 경향이 있다.
- EWMA와 SeasonalNaiveMAD는 alert cost efficiency가 높지만, 이는 alert 수가 적은 데서 오는 ratio 효과일 수 있다. 두 모델 모두 cost-weighted recall과 avoided cost ratio가 낮아, 비용 손실 최소화 목적에는 위험하다.
- IsolationForest는 F1 기준 2위이며, contextual anomaly와 high-intensity spike에서 안정적인 성능을 보인다.

### 9.2 full FOCUS strict service-level 결과

| service | F1 mean | AUPRC mean | cost-weighted recall mean | mean MCTD | avoided cost ratio |
|---|---:|---:|---:|---:|---:|
| AWS / Compute | 0.3823 | 0.5538 | 0.8103 | 277.8097 | 0.7590 |
| AWS / Databases | 0.2504 | 0.5317 | 0.6342 | 12.9523 | 0.5571 |
| AWS / Other | 0.4079 | 0.5890 | 0.8003 | 96.7656 | 0.7607 |
| Oracle / Compute | 0.3993 | 0.5707 | 0.8317 | 44.5358 | 0.7790 |

해석:

- AWS / Other와 Oracle / Compute에서 평균 F1이 높다. 단, "Other"는 FOCUS `ServiceCategory`에서 다른 카테고리에 속하지 않는 잔여 service들을 묶은 heterogeneous bucket이므로, 이 결과는 단일 service 특성이라기보다 다양한 service pattern이 섞여 평균화된 결과로 해석해야 한다.
- AWS / Compute는 base level이 가장 높고 mean MCTD도 높아, 비용 규모가 큰 workload에서 늦은 탐지가 더 큰 손실로 이어질 수 있음을 보여준다.
- AWS / Databases는 전체 평균 F1과 dollar recall이 가장 낮다. 해당 service pattern에서는 anomaly와 normal variation의 구분이 더 어려웠을 가능성이 있다.

### 9.3 anomaly type별 결과

Primary setting: Year-1 FAR target = 1%

| model | anomaly type | n events | detection rate | mean detection delay | mean MCTD | avoided cost ratio | dollar recall |
|---|---|---:|---:|---:|---:|---:|---:|
| EWMA | spike | 108 | 0.4259 | 0.13 | 112.8210 | 0.7260 | 0.7607 |
| IsolationForest | spike | 108 | 0.7130 | 0.03 | 27.4138 | 0.9334 | 0.9483 |
| LSTM_AE | spike | 108 | 0.8333 | 0.02 | 4.4563 | 0.9892 | 0.9902 |
| Prophet | spike | 108 | 0.7870 | 0.01 | 17.0526 | 0.9586 | 0.9634 |
| SeasonalNaiveMAD | spike | 108 | 0.4444 | 0.04 | 77.1690 | 0.8126 | 0.8220 |
| EWMA | contextual | 108 | 0.4074 | 0.20 | 162.8730 | 0.4863 | 0.5177 |
| IsolationForest | contextual | 108 | 0.8704 | 0.00 | 4.3816 | 0.9862 | 0.9862 |
| LSTM_AE | contextual | 108 | 0.8704 | 0.03 | 1.7480 | 0.9945 | 0.9957 |
| Prophet | contextual | 108 | 0.9259 | 0.01 | 0.6649 | 0.9979 | 0.9980 |
| SeasonalNaiveMAD | contextual | 108 | 0.3889 | 0.00 | 112.0162 | 0.6467 | 0.6467 |
| EWMA | gradual | 108 | 0.2037 | 5.36 | 461.5614 | 0.2440 | 0.3375 |
| IsolationForest | gradual | 108 | 0.7315 | 3.19 | 101.6991 | 0.8334 | 0.9335 |
| LSTM_AE | gradual | 108 | 0.8056 | 1.80 | 52.2145 | 0.9145 | 0.9580 |
| Prophet | gradual | 108 | 0.8704 | 2.50 | 41.2631 | 0.9324 | 0.9905 |
| SeasonalNaiveMAD | gradual | 108 | 0.3241 | 6.11 | 442.9035 | 0.2745 | 0.4494 |

**intensity별 event count breakdown** (`focus_anomaly_intensity_results.csv` 기준):

| anomaly type | low | mid | high | total |
|---|---:|---:|---:|---:|
| spike | 36 | 36 | 36 | 108 |
| contextual | 36 | 36 | 36 | 108 |
| gradual | 36 | 36 | 36 | 108 |

최종 full strict run에서는 `--n-events-per-combo 3`을 적용해 intensity별 event count imbalance를 제거했다. 따라서 이전 run에서 gradual-high가 4건뿐이던 문제는 해결되었고, type × intensity 단위 비교의 표본 균형성이 개선됐다.

해석:

- Prophet은 세 anomaly type 모두에서 높은 detection rate와 dollar recall을 보인다.
- LSTM_AE는 spike에서 가장 낮은 MCTD를 보이고, gradual에서도 Prophet 다음으로 높은 avoided cost ratio를 보인다.
- EWMA와 SeasonalNaiveMAD는 gradual anomaly에 특히 취약하다. 두 모델 모두 detection rate와 avoided cost ratio가 낮아 점진적 비용 손실을 늦게 잡는다.
- Contextual anomaly에서는 Prophet, IsolationForest, LSTM_AE가 모두 높은 detection rate를 보였지만, EWMA와 SeasonalNaiveMAD는 상대적으로 낮았다.

### 9.4 100k relaxed benchmark 결과

이 결과는 현재 5-model configuration과 `--n-events-per-combo 3`으로 재실행한 보조 robustness run이다. Primary setting은 Year-1 FAR target = 1%이며, 값은 `mean ± std`이다. std는 service × seed (총 12 × 5 = 60 cell) pooled 표준편차이다.

| model | F1 | F1 rank | cost-weighted recall | dollar recall rank | alert cost efficiency | ACE rank | mean MCTD | MCTD rank | avoided cost ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prophet | 0.6000 ± 0.1209 | 1 | 0.9417 ± 0.0864 | 1 | 4.68 ± 11.35 | 4 | 1.16 ± 2.75 | 1 | 0.8709 ± 0.1177 |
| IsolationForest | 0.4266 ± 0.1178 | 2 | 0.8535 ± 0.1390 | 3 | 6.13 ± 13.58 | 3 | 1.89 ± 4.33 | 3 | 0.7866 ± 0.1465 |
| LSTM_AE | 0.3360 ± 0.0995 | 3 | 0.8621 ± 0.1695 | 2 | 1.71 ± 3.82 | 5 | 1.24 ± 2.55 | 2 | 0.8056 ± 0.1736 |
| SeasonalNaiveMAD | 0.2560 ± 0.0887 | 4 | 0.6161 ± 0.1841 | 4 | 14.85 ± 36.87 | 2 | 7.58 ± 19.67 | 4 | 0.5268 ± 0.1396 |
| EWMA | 0.1860 ± 0.0310 | 5 | 0.5267 ± 0.0731 | 5 | 19.16 ± 49.14 | 1 | 8.84 ± 21.83 | 5 | 0.4505 ± 0.0599 |

이 결과는 full strict benchmark와 같은 방향을 보인다.

- Prophet은 F1과 dollar recall에서 1위다.
- EWMA와 SeasonalNaiveMAD는 ACE는 높지만 F1, dollar recall, MCTD, avoided cost ratio가 낮다.
- IsolationForest와 LSTM_AE는 중간 순위에서 metric별 순위가 달라진다.

즉, sample size와 filtering 기준을 바꿔도 핵심 결론은 유지된다.

### 9.5 raw full FOCUS sanity check 결과

raw full FOCUS relaxed sanity check에서는 18개 service group이 살아남았다.

| service | days | nonzero days | mean daily cost | alert count | alert rate |
|---|---:|---:|---:|---:|---:|
| AWS / Analytics | 30 | 8 | 0.7667 | 3 | 0.100 |
| AWS / Compute | 30 | 19 | 33.0667 | 3 | 0.100 |
| AWS / Databases | 30 | 21 | 16.5000 | 2 | 0.067 |
| AWS / Other | 30 | 30 | 54.0427 | 1 | 0.033 |
| Microsoft / AI and Machine Learning | 19 | 19 | 13.2048 | 2 | 0.105 |
| Microsoft / Analytics | 19 | 19 | 2182.1710 | 4 | 0.211 |
| Microsoft / Compute | 19 | 19 | 513.4785 | 2 | 0.105 |
| Microsoft / Databases | 19 | 19 | 355.0404 | 2 | 0.105 |
| Microsoft / Integration | 19 | 19 | 108.7477 | 3 | 0.158 |
| Microsoft / Management and Governance | 19 | 19 | 15.6880 | 4 | 0.211 |
| Microsoft / Networking | 19 | 19 | 119.3848 | 6 | 0.316 |
| Microsoft / Other | 19 | 19 | 226.5997 | 3 | 0.158 |
| Microsoft / Security | 19 | 19 | 1.2742 | 1 | 0.053 |
| Microsoft / Storage | 19 | 19 | 37.6989 | 3 | 0.158 |
| Microsoft / Web | 19 | 19 | 215.4571 | 5 | 0.263 |
| Oracle / Compute | 30 | 15 | 24.6667 | 1 | 0.033 |
| Oracle / Web | 15 | 15 | 9.8667 | 3 | 0.200 |
| Oracle / nan | 15 | 15 | 2.4667 | 3 | 0.200 |

주의:

- 이 표는 실제 anomaly 정답이 아니다.
- alert count는 rolling z-score가 이상하게 본 날짜 수일 뿐이다.
- 일부 비용 값에는 credit, correction, refund 등으로 인해 음수 비용이 포함될 수 있다.
- 따라서 이 결과는 "실제 FOCUS 데이터에서도 비용 급변 날짜가 존재하며, 간단한 detector가 이를 flag할 수 있다"는 sanity check로만 사용한다.
- 위 표의 `mean_daily_cost`는 30일 raw 평균(0인 날 포함)이고, Section 8.2의 `base_level`은 calibration 단계의 detrend·smoothing 후 산출한 평균 수준이다. 따라서 같은 service에서도 두 값은 일치하지 않을 수 있다 (예: AWS / Databases는 raw 16.50, calibrated base 18.50).

## 10. 연구 질문과 연결

### RQ1. 탐지 패러다임별 anomaly type 성능 차이가 있는가

있다. full strict benchmark에서 Prophet은 contextual과 gradual에서 가장 높은 detection rate와 dollar recall을 보였고, LSTM_AE는 spike에서 가장 낮은 MCTD를 보였다. IsolationForest는 contextual anomaly에서 안정적이었다. EWMA와 SeasonalNaiveMAD는 gradual anomaly에서 특히 약했다. 이는 단순 residual 또는 robust seasonal baseline만으로는 점진적 비용 상승과 복합 맥락을 충분히 포착하기 어렵다는 점을 보여준다.

### RQ2. 표준 지표와 비용 가중 지표의 모델 순위가 다른가

다르다. full strict benchmark에서 F1과 dollar recall 기준 1위는 Prophet이지만, alert cost efficiency 기준 1위는 EWMA다. 그러나 EWMA는 dollar recall이 0.5154, avoided cost ratio가 0.4590으로 낮고 mean MCTD도 245.75로 가장 나쁘다. 즉 alert efficiency만 보면 EWMA가 좋아 보이지만, 실제 비용 손실을 줄이는 관점에서는 부적절할 수 있다. 이는 FinOps 비용 anomaly detection에서 단일 분류 지표나 단일 efficiency 지표만으로 모델을 선택하면 운영상 잘못된 결론을 낼 수 있음을 보여준다.

### RQ3. 탐지 메커니즘과 anomaly 유형의 mismatch가 있는가

있다. EWMA는 급격한 spike에는 어느 정도 반응하지만, contextual anomaly와 gradual anomaly에서는 취약하다. SeasonalNaiveMAD는 같은 요일 baseline을 쓰기 때문에 high-intensity spike에는 반응하지만, low-intensity contextual/gradual anomaly에는 약했다. IsolationForest는 calendar와 lag feature를 사용하기 때문에 contextual anomaly에서 강한 편이다. Prophet은 trend와 seasonality를 모델링하므로 baseline 대비 residual이 명확한 경우에 강하다. LSTM_AE는 정상 패턴 reconstruction을 기준으로 하므로 spike와 gradual 변화에서 비용 손실을 빠르게 줄이는 장점이 나타났다.

## 11. 발표용 핵심 주장

발표에서는 다음 문장을 중심으로 설명하는 것이 좋다.

1. 클라우드 비용 이상 탐지는 일반 anomaly detection과 달리 "얼마나 빨리, 얼마나 큰 비용 손실을 잡았는가"가 중요하다.
2. 실제 billing data는 민감하고 anomaly label이 없기 때문에 공개 labeled benchmark가 부족하다.
3. 본 연구는 FOCUS public sample에서 실제 비용 통계량을 추출해 synthetic benchmark를 보정했다.
4. 이 방식은 실제 데이터 realism과 label availability를 동시에 확보한다.
5. 실험 결과, F1 기준 모델 순위와 비용 가중 지표 기준 모델 순위가 달라졌다.
6. Prophet은 F1과 dollar recall에서 강했지만, LSTM_AE는 MCTD에서 강했고, EWMA와 SeasonalNaiveMAD는 alert cost efficiency만 높게 보일 수 있었다.
7. 따라서 FinOps 모델 선택에는 F1뿐 아니라 dollar recall, MCTD, avoided cost ratio, alert efficiency를 함께 봐야 한다.

## 12. 발표용 주의 문장

질문을 받을 가능성이 높은 부분에 대한 답변 문장은 다음과 같다.

질문: "이건 실제 데이터 실험인가, 합성 데이터 실험인가?"

답변:

> 둘의 중간입니다. raw FOCUS 데이터를 그대로 labeled benchmark로 쓴 것은 아닙니다. FOCUS에는 anomaly 정답 라벨이 없기 때문입니다. 대신 FOCUS에서 실제 비용 수준, 추세, 변동성, 요일 패턴을 추출해 synthetic benchmark baseline을 보정했고, anomaly는 통제해서 주입했습니다. 그래서 정량 평가는 FOCUS-calibrated synthetic benchmark이고, raw FOCUS는 sanity check로 사용했습니다.

질문: "왜 full FOCUS를 쓰고도 service group이 4개뿐인가?"

답변:

> full sample은 5.49M rows지만 관측 기간이 30일 내외로 짧고 provider/service별 sparsity가 큽니다. strict filter에서는 충분한 일수, nonzero day, 평균 비용 조건을 만족하는 안정적 group만 남겼기 때문에 4개가 되었습니다. relaxed setting에서는 18개 raw sanity group, 100k benchmark에서는 12개 group까지 확인했습니다.

질문: "raw FOCUS 결과에 precision이나 recall은 왜 없나?"

답변:

> raw FOCUS에는 anomaly 정답 label이 없기 때문입니다. 실제 비용 급등이 정상 workload인지 misconfiguration인지 알 수 없기 때문에 precision/recall을 계산하면 오히려 잘못된 주장이 됩니다. 대신 raw FOCUS는 detector가 실제 비용 series에서 의심 날짜를 찾는지 보는 sanity check로만 사용했습니다.

질문: "이 결과로 어떤 모델이 제일 좋다고 말할 수 있나?"

답변:

> 단일 답은 평가 목적에 따라 다릅니다. F1과 dollar recall 기준으로는 Prophet이 가장 강했습니다. MCTD 기준으로는 LSTM_AE가 좋은 모습을 보였습니다. EWMA와 SeasonalNaiveMAD는 alert cost efficiency는 높게 보일 수 있지만 놓치는 비용과 탐지 지연이 커서 FinOps 운영 목적에는 위험할 수 있습니다. 이 차이가 본 연구의 핵심입니다.

질문: "왜 Prophet 결과를 일반화하면 안 되나?"

답변:

> 본 benchmark의 baseline은 trend, weekly seasonality, monthly batch effect를 포함하도록 생성됩니다. Prophet도 trend와 seasonality를 명시적으로 모델링하므로 데이터 생성 구조와 모델 가정이 잘 맞습니다. 따라서 본 결과는 "FOCUS 1.0 sample로 보정한 trend/weekly benchmark에서 Prophet이 강하다"는 결론이지, 모든 실제 cloud billing 환경에서 Prophet이 항상 우월하다는 결론은 아닙니다.

## 13. 생성된 산출물

### 13.1 full strict benchmark

결과 디렉터리:

```text
outputs/results_full_strict/
```

핵심 파일:

| 파일 | 설명 |
|---|---|
| `focus_run_metadata.json` | 실행 설정, 데이터 URL, seed, runtime, warning |
| `focus_calibration_stats.csv` | service별 FOCUS calibration parameter, raw/clipped 값, clipping saturation 여부 |
| `focus_calibration_summary.csv` | calibration parameter와 clipping saturation의 cross-service 요약 |
| `focus_service_summary.csv` | service별 평균 성능 |
| `focus_core_metrics_by_service.csv` | service x model x budget별 mean/std |
| `focus_overall_model_ranking.csv` | 전체 모델 ranking (mean only) |
| `focus_overall_model_ranking_with_std.csv` | 전체 모델 ranking (mean ± std, pooled across service × seed) |
| `focus_anomaly_type_results.csv` | anomaly type별 detection 결과 |
| `focus_anomaly_intensity_results.csv` | anomaly type x intensity별 결과 |
| `focus_rank_reversal_by_service.csv` | metric별 model ranking reversal |
| `focus_rank_reversal_summary.csv` | model pair별 rank reversal rate |

그림 디렉터리:

```text
outputs/figures_full_strict/
```

생성된 그림:

- `focus_f1_bar.png`
- `focus_dollar_recall_bar.png`
- `focus_ace_bar.png`
- `focus_mctd_bar.png`
- `focus_f1_by_budget.png`
- `focus_mctd_by_budget.png`
- `focus_far_by_budget.png`
- `focus_metric_overview.png`
- `focus_detection_by_type.png`
- `focus_detection_heatmap.png`
- `focus_mctd_by_type.png`
- `focus_radar.png`
- `focus_rank_slope.png`
- `focus_unsupervised_full_relaxed_case_AWS___Compute.png`
- `focus_unsupervised_full_relaxed_case_Microsoft___Networking.png`

### 13.2 100k relaxed benchmark

결과 디렉터리:

```text
outputs/results/
```

해당 디렉터리의 `focus_run_metadata.json`은 100k sample relaxed benchmark 실행 정보를 담고 있다.

이 100k relaxed benchmark는 현재 5-model 설정으로 재실행했으며, `n_fallback_services: 7` 때문에 service-specific conclusion은 조심해야 한다. 최종 primary result는 여전히 더 보수적인 `outputs/results_full_strict/` full strict run이다.

### 13.3 raw FOCUS sanity check

결과 파일:

```text
outputs/results/focus_unsupervised_full_relaxed_alerts.csv
outputs/results/focus_unsupervised_full_relaxed_summary.csv
```

이 파일들은 raw full FOCUS daily cost series에 rolling z-score detector를 적용한 결과다.

`outputs/results/` 디렉터리에는 13.2의 100k benchmark 산출물과 본 13.3의 raw sanity check 산출물이 공존한다. 파일명 prefix로 구분 가능하다.

- 100k benchmark: `focus_run_metadata.json`, `focus_overall_model_ranking.csv`, `focus_metrics_*`, `focus_events_*`, `focus_anomaly_*`, `focus_service_summary.csv` 등
- raw sanity check: `focus_unsupervised_*` 로 시작하는 모든 파일

### 13.4 raw FOCUS data inventory

```text
outputs/focus_data_inventory.json
```

`scripts/build_focus_inventory.py`로 캐시된 FOCUS 파일(`.focus_cache/*.csv*`)에서 직접 산출한 인벤토리 파일이다. 각 sample 파일의 byte size, rows, columns, date range, unique billing days, provider별 row 수가 들어 있어, Section 4.3의 데이터 인벤토리 주장을 재현 검증할 수 있다.

## 14. 한계

본 연구의 한계는 다음과 같다.

1. FOCUS sample은 공개 sample data이므로 실제 기업 운영 환경 전체를 대표한다고 볼 수 없다.
2. FOCUS 원자료에는 anomaly label이 없어서 raw real-data benchmark는 불가능하다.
3. 관측 기간이 30일 내외로 짧아 장기 trend, quarterly seasonality, annual seasonality는 검증할 수 없다. 또한 FOCUS sample의 청구일 자체가 sparse (195일 캘린더 구간에 32 unique billing days) 하므로 weekly_factor 추정에는 잔여 노이즈가 남는다.
4. 일부 provider 또는 service는 row 수가 적어 filter를 통과하지 못했다. full strict benchmark는 4 service groups × 3 seeds = 12 cell에서 집계되므로 cross-service 일반화 주장은 thin sample 기반임을 명시한다.
5. calibration parameter는 짧은 관측 기간에서 추정되므로 clipping과 smoothing을 적용했다.
6. 최종 full strict run에서는 4/4 service group의 `monthly_growth`와 `noise_pct`가 clipping bound에 도달했다. 이는 raw FOCUS sample의 짧은 관측치에서 추정된 trend/noise가 매우 불안정하다는 신호이며, synthetic baseline이 실제 raw 변동성을 완전히 재현한다는 뜻이 아니다.
7. 본 실험은 FOCUS 1.0 sample data에 기반한다. FOCUS specification은 이후 1.3까지 확장되었으므로, 최신 schema/sample 또는 실제 조직의 FOCUS-exported billing data로 검증하는 후속 연구가 필요하다.
8. anomaly injection은 type × intensity event count를 균형화했지만, 실제 misconfiguration, traffic burst, discount correction 등 모든 현실 이벤트를 완벽히 반영하지는 않는다. spike/contextual/gradual 각 intensity multiplier도 config 상수이지 FOCUS 비용 분포에서 추정한 값이 아니므로 magnitude의 외부 정당성은 약하다.
9. Prophet, IsolationForest, LSTM_AE, EWMA, SeasonalNaiveMAD의 hyperparameter는 대표적 baseline으로 설정했으며, 각 모델의 최적 tuning 연구는 별도 과제다.
10. **모델-데이터 생성 구조 일치로 인한 내재 편향**. FOCUS-calibrated baseline은 *linear trend + multiplicative weekly factor + monthly batch effect + Gaussian noise* 구조로 생성된다 (`data.py`). 이는 Prophet의 additive trend + weekly seasonality + holiday/event 모델 가정과 거의 isomorphic하다. 따라서 본 실험에서 Prophet의 F1·dollar recall 우위는 모델 자체의 강점과 동시에 데이터 생성 과정이 Prophet의 가정과 잘 맞는 구조적 효과를 함께 반영한다. 따라서 본 보고서의 정확한 주장은 "linear trend와 weekly seasonality가 명확한 FOCUS-calibrated synthetic benchmark에서 Prophet이 강하다" 이며, 임의의 실제 cloud billing 시계열에서 Prophet이 보편적으로 우월하다고 일반화할 수는 없다.
11. raw real-time / streaming 환경에서의 sequential detection은 본 실험에 포함되지 않았다. 모든 평가는 Year 2 전체 구간에 대한 batch scoring이다.

이 한계에도 불구하고, 본 연구는 순수 합성 실험보다 실제 FinOps billing data와 더 강하게 연결되어 있다. 동시에 없는 label을 있다고 가정하지 않기 때문에 연구 정당성과 정직성을 함께 확보한다.

## 15. 최종 결론

본 연구의 최종 데이터 전략은 다음과 같이 정리할 수 있다.

```text
Primary quantitative result:
  FOCUS-calibrated synthetic benchmark

External realism source:
  Official FOCUS public billing sample data

Raw real-data use:
  Unsupervised sanity check only

Not claimed:
  Raw FOCUS labeled anomaly benchmark
```

이 전략은 실제 데이터 활용과 정량 평가 가능성 사이에서 가장 방어 가능한 선택이다. 실제 FOCUS full sample의 비용 패턴으로 benchmark baseline을 보정했기 때문에 순수 합성 데이터보다 현실성이 높고, anomaly label은 통제된 injection으로 확보했기 때문에 F1, recall, detection delay, MCTD, dollar recall, avoided cost ratio 같은 지표를 정당하게 계산할 수 있다.

결과적으로 최종 full strict run에서 Prophet은 F1과 cost-weighted recall에서 가장 강한 모델로 나타났고, LSTM_AE는 MCTD 측면에서 의미 있는 강점을 보였다. EWMA와 SeasonalNaiveMAD는 alert cost efficiency만으로는 좋아 보일 수 있지만, cost-weighted recall과 avoided cost ratio가 낮아 실제 비용 손실을 줄이는 목적에서는 취약했다. 따라서 FinOps 비용 이상 탐지 모델을 선택할 때는 표준 분류 지표와 비용 가중 운영 지표를 함께 고려해야 한다.

다만 Prophet의 우위에는 데이터 생성 구조와 모델 가정의 isomorphism이 일부 기여한다(Section 14, 한계 10). 따라서 본 결론은 "FOCUS 1.0 sample로 보정한 FOCUS-calibrated synthetic benchmark에서의 비교 결과"로 한정해 해석해야 하며, 실제 다양한 cloud billing 환경에서 동일한 ranking이 재현된다고 단정할 수는 없다.
