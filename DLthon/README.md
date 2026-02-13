# DKTC — 한국어 위험 대화 분류 프로젝트 회고

> AIFFEL DLthon · Team online16 · Kaggle Competition
> F1 0.72 → 0.79 → 0.72, 네 번의 실험에서 배운 것

---

## 프로젝트 개요

DKTC(Dangerous Korean Talk Classification)는 카카오톡 스타일의 한국어 대화를 5개 클래스로 분류하는 Kaggle 대회다.

| 클래스 | 라벨 |
|--------|------|
| 0 | 협박 (Threat) |
| 1 | 갈취 (Blackmail) |
| 2 | 직장내 괴롭힘 (Workplace Bullying) |
| 3 | 기타 괴롭힘 (Other Harassment) |
| 4 | 일반 대화 (Normal Conversation) |

4번의 버전을 거치면서 0.72에서 0.79까지 올렸다가, 마지막에 TAPT와 대규모 데이터 확장을 시도하면서 다시 0.72로 돌아왔다. 리더보드 1위는 0.88이었다.

```
v1:  0.72  KcELECTRA baseline, 클래스 4 데이터 없음
v2:  0.74  5-source 합성 데이터 + K-Fold
v3:  0.79  ⭐ HNM + Focal + FGM + R-Drop + Prior Calibration + Pseudo Label
v4:  0.72  ↓ TAPT + 23K data → 29:1 불균형 + 도메인 불일치
```

---

## 핵심 문제: train에 없는 클래스가 test의 75.6%

train.csv는 클래스 0~3(위험 대화) 약 800개로 구성되어 있다. **클래스 4(일반 대화)는 0개다.** 반면 test.csv의 약 75.6%가 클래스 4다. 학습 데이터에 없는 클래스가 테스트의 대부분을 차지하는 distribution shift 문제.

이 프로젝트의 핵심 과제:
1. **클래스 4의 학습 데이터를 어떻게 확보할 것인가** (data acquisition)
2. **train과 test의 클래스 분포 차이를 어떻게 보정할 것인가** (distribution calibration)

---

## v1 — Baseline: KcELECTRA fine-tuning (F1 = 0.72)

### 모델: `beomi/KcELECTRA-base-v2022`

DKTC 텍스트는 "야 뭐하냐 ㅋㅋ", "돈 안 갚으면 알지?" 같은 카톡 구어체다. 사전학습 모델 선택의 핵심 기준은 **사전학습 코퍼스와 task 도메인의 일치도**였다.

`klue/bert-base`는 뉴스·위키·댓글 등 혼합 코퍼스로 학습되어 범용성은 높지만, 비속어·줄임말·이모티콘이 섞인 카톡 대화체에는 약하다. `beomi/KcELECTRA-base-v2022`는 한국어 온라인 댓글 1억 1천만 문장으로 사전학습되어 DKTC 도메인과 가장 가까웠다.

ELECTRA는 BERT의 MLM(Masked Language Modeling)과 달리 RTD(Replaced Token Detection)를 사용한다. MLM은 입력 토큰의 15%만 마스킹하여 학습 신호가 sparse하지만, RTD는 모든 토큰에서 "원본인가 생성된 것인가"를 판별하므로 학습 효율이 높다 (Clark et al., 2020). 학습 데이터가 ~800개밖에 없는 low-resource 상황에서 이 효율 차이는 유의미하다.

### 결과

F1 = 0.72. 위험 대화 4개 클래스는 분류하지만, 클래스 4 예측이 불가능하다. test의 75.6%를 차지하는 클래스를 못 맞추니 macro F1이 낮을 수밖에 없다.

---

## v2 — 외부 데이터 조합 + K-Fold (F1 = 0.74)

### 데이터 조합 전략

클래스 4를 위한 일반 대화 데이터를 **5개 공개 데이터셋에서 수집**했다. LLM 생성 대신 기존 데이터를 사용한 이유는, (1) LLM이 만드는 대화는 오타·줄임말·이모티콘이 빠져 DKTC 도메인과 맞지 않고, (2) 1,000개 생성에 필요한 API 비용과 시간이 비효율적이었기 때문이다.

| 데이터 | 출처 | 수량 | 선택 이유 |
|--------|------|------|-----------|
| SmileStyle `informal` | Smilegate AI | 400 | 17가지 말투 병렬 코퍼스. 반말 컬럼이 DKTC 대화체와 register 거의 동일 |
| KakaoChatData | 카카오톡 대화 73K | 300 | 실제 카카오톡 Q-A 쌍. `contains_threat()` 필터링으로 위협 키워드 포함 문장 제거 |
| kor_unsmile `clean=1` | Smilegate AI | 200 | 혐오 분류 완료 데이터. clean=1은 비혐오 구어체 → 일반 대화로 노이즈 적음 |
| NSMC positive | 네이버 영화 리뷰 | 100 | 긍정 리뷰 중 일상 키워드 포함만 필터링. register 차이 있으나 무해한 구어체 확보용 |
| 경계 케이스 (수동 작성) | — | 25 | "야 죽을래 ㅋㅋ 아 진짜 웃겨서 죽겠다" 등 위협 키워드 포함 농담. 키워드 과적합 방지 |

v2에서는 이 문장들을 2~3개씩 **concatenation**하여 합성 샘플을 생성했다. 위험 대화 텍스트가 다수 문장으로 구성된 것에 길이를 맞추기 위해서였다.

### K-Fold Cross Validation (5-Fold)

학습 데이터가 ~800개인 상황에서 단순 80/20 split을 하면 validation set이 ~160개가 되어 평가 분산이 크다. 5-Fold CV를 적용하면 모든 샘플이 정확히 1번 validation에 사용되고, 5개 fold의 OOF(Out-of-Fold) 예측을 평균하여 최종 예측의 분산을 줄인다.

### 결과와 한계

F1 = 0.74 (+0.02). concatenation의 한계가 보였다. "안녕하세요. 날씨가 좋네요. 밥 먹었어?"처럼 맥락 없이 이어진 텍스트는 실제 대화와 분포가 다르다. 모델이 이 부자연스러운 패턴을 shortcut으로 학습하여, 자연스러운 일반 대화를 오히려 놓치는 현상이 발생했다.

---

## v3 — Hard Negative Mining + 11개 기법 종합 (F1 = 0.79) ⭐

v2의 실패 원인 분석:
1. Concatenation이 만든 distribution artifact
2. 랜덤 샘플링으로는 decision boundary 근처의 hard example을 학습하지 못함

### 데이터 파이프라인 재설계

**Concatenation → 개별 문장 샘플링으로 전환.** 문장을 그대로 사용하여 원본 데이터의 분포를 보존.

**Hard Negative Mining.** 핵심 아이디어: 랜덤 샘플링된 일반 대화 1,000개보다, 위험 대화와 embedding space에서 가까운 일반 대화 200개가 decision boundary를 더 정확하게 학습시킨다.

구현:
1. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`로 전체 텍스트를 384차원 벡터로 인코딩
2. 위험 대화 각 클래스(0~3)의 centroid 계산
3. 일반 대화 후보 풀(KakaoData 73,000개)에서 각 centroid와 cosine similarity가 높은 상위 N개를 선별
4. 클래스별 약 50개씩, 총 165개의 Hard Negative 선별

예시: "야 죽여버린다 이 게임" — 게임 대화(일반)이지만 `죽여버린다`가 포함되어 협박 클래스와 cosine similarity가 높다. 이런 샘플을 학습에 포함시키면 모델이 context를 보고 판단하게 된다.

**Cosine Deduplication.** 임베딩 공간에서 cosine similarity > 0.95인 쌍을 제거. 의미적 중복("안녕하세요"와 "안녕하세요!")을 제거하여 데이터 다양성 확보.

**Quality Filter.** 텍스트 길이 15자 미만/500자 초과 제거, 특수문자 비율 > 30% 제거, 동일 문자 5회 이상 연속 반복 제거.

최종 데이터: 일반 대화 ~2,665개 + 위험 대화 ~800개. **비율 약 4:1.**

### 학습 기법

#### Focal Loss (Lin et al., ICCV 2017)

클래스 4(일반 대화)가 ~2,665개로 다수 클래스이므로, 쉬운 일반 대화 샘플이 전체 gradient를 지배한다. Focal Loss는 이미 잘 맞추는 샘플의 loss를 줄인다.

```
FL(p_t) = -α_t(1-p_t)^γ · log(p_t)     γ = 2.0
```

#### LLRD — Layer-wise Learning Rate Decay

Transformer의 하위 레이어는 범용적 언어 특징(morphology, syntax)을, 상위 레이어는 task-specific semantics를 인코딩한다 (Jawahar et al., 2019). LLRD는 layer `i`에 `lr × decay^(L-i)`를 적용. classifier head는 base lr(2e-5), 최하위 레이어는 `2e-5 × 0.95^12 ≈ 1.08e-5`.

```python
def get_llrd_optimizer(model, lr=2e-5, weight_decay=0.01, llrd_factor=0.95):
    no_decay = ['bias', 'LayerNorm.weight']
    params = []
    for i in range(12):  # 12 encoder layers
        layer_lr = lr * (llrd_factor ** (12 - i))
        params += [
            {"params": [p for n, p in model.electra.encoder.layer[i].named_parameters()
                       if not any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": weight_decay},
            {"params": [p for n, p in model.electra.encoder.layer[i].named_parameters()
                       if any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": 0.0}
        ]
    params.append({"params": model.classifier.parameters(), "lr": lr})
    return AdamW(params)
```

#### FGM — Fast Gradient Method (Goodfellow et al., ICLR 2015)

임베딩 `e`에 대해 loss를 증가시키는 방향으로 perturbation을 가한다:

```
r_adv = ε · ∇_e L(θ, e, y) / ‖∇_e L‖     ε = 1.0
```

perturbed embedding `e + r_adv`에서 forward pass를 한 번 더 수행하고, adversarial loss를 합산.

```python
class FGM:
    def __init__(self, model, epsilon=1.0, emb_name='word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if self.emb_name in name and param.requires_grad:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
```

#### R-Drop (Wu et al., ICLR)

같은 입력 `x`를 두 번 forward pass하면 dropout mask가 다르기 때문에 출력 분포가 다르다. 두 출력 간의 bidirectional KL divergence를 regularization term으로 추가:

```
L = CE(P₁, y) + CE(P₂, y) + α · [KL(P₁‖P₂) + KL(P₂‖P₁)] / 2     α = 0.7
```

#### EMA — Exponential Moving Average (decay = 0.999)

학습 중 shadow parameter를 유지: `θ_ema = β·θ_ema + (1-β)·θ`. 추론 시 EMA 파라미터를 사용하면 학습 후반의 특정 mini-batch에 과적합된 가중치 대신, 전체 학습 궤적의 exponential average로 예측.

#### Prior Shift Calibration (Saerens et al., 2002)

train에서 클래스 4 비율 ~40%, test에서 ~75.6%. 베이즈 정리로 보정:

```
P_cal(y=k|x) ∝ P(y=k|x) · (π_test(k) / π_train(k))
```

```python
def calibrate_probs(probs, train_prior, test_prior):
    ratio = test_prior / train_prior
    calibrated = probs * ratio
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated
```

#### 기타

- **Dynamic Class Weight** — 1차 학습 후 test 예측 분포를 관찰하고, `estimated / predicted` 비율로 class weight 동적 조정 후 재학습
- **Confidence Fallback** — `max(P(y=0|x), ..., P(y=3|x)) < 0.15`이면 클래스 4로 강제 분류
- **Pseudo Labeling** — confidence ≥ 0.95인 test 예측을 pseudo label로 추가, 0.5~0.95 구간은 2x weight. 앙상블 비율 0.4:0.6

### Training Configuration

```python
MAX_LEN           = 384
BATCH_SIZE        = 16
EPOCHS            = 5
N_FOLDS           = 5
LR                = 2e-5
WEIGHT_DECAY      = 0.01
WARMUP_RATIO      = 0.1
MAX_GRAD_NORM     = 1.0
LLRD_FACTOR       = 0.95
FGM_EPSILON       = 1.0
LABEL_SMOOTHING   = 0.05
EMA_DECAY         = 0.999
RDROP_ALPHA       = 0.7
FOCAL_GAMMA       = 2.0
COSINE_SIM_THRESH = 0.95
PSEUDO_THRESHOLD  = 0.95
```

### 전체 파이프라인

```
┌─────────────────────────────────────────────┐
│  Data Acquisition                           │
│  5 public datasets → Quality Filter →       │
│  Cosine Dedup → Hard Negative Mining (SBERT)│
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  5-Fold Training                            │
│  KcELECTRA + Classification Head            │
│  Loss: Focal Loss + R-Drop KL              │
│  Regularization: FGM, EMA, LLRD            │
│  5 epochs × 5 folds → Best ckpt per fold   │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Post-Processing                            │
│  Weighted Ensemble (fold F1 기반) →          │
│  Prior Shift Calibration →                  │
│  Per-class Threshold Optimization →          │
│  Confidence Fallback                        │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│  Pseudo Labeling                            │
│  High conf (≥0.95) → pseudo train           │
│  Medium (0.5~0.95) → 2x weight retrain     │
│  Ensemble: 0.4 × 1st + 0.6 × pseudo model  │
└────────────────────┬────────────────────────┘
                     ↓
           submission.csv (F1 = 0.79)
```

### 결과

F1 = **0.79** (+0.05). 가장 큰 기여는 Hard Negative Mining이었다. 다만 11개 기법을 동시에 적용했기 때문에 각 기법의 개별 기여도를 정량적으로 분리하지 못한 것이 한계 (ablation study 미수행).

---

## v4 — TAPT + 대규모 데이터 확장 (F1 = 0.72)

### TAPT (Task-Adaptive Pre-Training)

Gururangan et al. "Don't Stop Pretraining" (ACL 2020)에 따르면, 사전학습 모델에 task domain 텍스트로 추가 MLM을 수행하면 downstream task에서 평균 3~5%p 향상된다.

**모델 변경: KcELECTRA → klue/bert-base.** ELECTRA의 RTD 구조는 Generator와 Discriminator가 분리되어 있어 Discriminator만으로 MLM을 수행할 수 없다. `klue/bert-base`는 KLUE 벤치마크(Park et al., 2021)에서 공개한 모델로, 뉴스·위키·댓글 등 62GB 한국어 코퍼스로 학습.

### 데이터 확장

| 소스 | 수량 |
|------|------|
| SmileStyle | 1,200 |
| kor_unsmile | 800 |
| KakaoChatData | 500 |
| NSMC | 500 |
| korean_safe_conversation | **10,000** |
| kor_nli | **10,000** |
| Hard Negative | 165 |
| **총 일반대화** | **~23,165** |

위험 대화: ~800. **비율 29:1.**

### 실패 원인 분석

**1. 극심한 클래스 불균형 (29:1).** v3의 4:1에서 29:1로 악화. Focal Loss와 Dynamic Class Weight를 적용해도, 이 수준의 불균형에서는 모델이 majority class로 collapse하는 경향을 막기 어렵다.

**2. kor_nli의 도메인 불일치.** NLI는 "한 남자가 공원에서 개를 산책시키고 있다. 따라서 남자는 밖에 있다."와 같은 형식적, 서술적 문체. DKTC의 대화체("야 뭐해?", "밥 먹자 ㅋㅋ")와 완전히 다른 register. 모델이 NLI의 형식적 패턴까지 "일반 대화"로 학습하면서 위험 대화의 형식적 표현도 일반으로 오분류.

**3. TAPT 오염.** 23,165개 중 10,000개가 NLI 문체. TAPT 과정에서 모델이 NLI의 형식적 패턴을 내재화하면서, 원래 사전학습된 대화체 표현이 오히려 희석.

---

## Data Augmentation

전통적 텍스트 augmentation(Back-Translation, EDA, paraphrasing)은 사용하지 않았다. 대신 세 가지 수준에서 데이터 다양성을 확보:

| 수준 | 기법 | 버전 | 설명 |
|------|------|------|------|
| 텍스트 레벨 | Concatenation | v2 | 외부 데이터 문장 2~3개 이어붙여 합성 샘플 생성. distribution artifact 문제로 v3에서 폐기 |
| 데이터 선별 | Hard Negative Mining | v3 | SBERT 임베딩 기반 decision boundary 근처 hard example 전략적 선별. 텍스트 변형이 아닌 **선별 최적화** |
| 학습 레벨 | R-Drop + FGM | v3~v4 | R-Drop: dropout randomness 활용한 implicit augmentation. FGM: embedding space adversarial perturbation |

---

## Ablation Study — 못 한 것과 해야 할 것

v3에서 11개 기법을 동시에 적용했다. 각 기법의 marginal contribution을 분리하지 못한 것이 명확한 한계다.

수행했어야 할 ablation:

| Exp | 설정 | 비교 포인트 |
|-----|------|-----------|
| 1 | v3 전체 (baseline) | 기준점 |
| 2 | Hard Negative Mining 제거 | HNM의 기여도 |
| 3 | Focal Loss → CE Loss | 불균형 대응 효과 |
| 4 | 일반대화 2,665 → 500개 축소 | 데이터 양의 영향 |
| 5 | R-Drop 제거 | regularization 기여도 |

"Hard Negative Mining이 가장 큰 기여를 했다"는 추측이지 증명이 아니다.

---

## 핵심 교훈

**데이터 양 ≠ 성능.** v3에서 2,665개로 0.79, v4에서 23,165개로 0.72. domain-matched small data > domain-mismatched large data.

**클래스 비율 관리.** 경험적으로 4:1~5:1이 적절. 29:1은 어떤 loss function으로도 극복하기 어렵다.

**Hard Negative Mining의 위력.** 랜덤 샘플링 1,000개보다 전략적 선별 200개가 효과적. decision boundary 근처에서의 학습이 핵심.

**모델보다 데이터가 중요하다.** v4에서 모델을 바꾸고 기법을 추가해도 데이터 전략이 잘못되니 성능이 하락했다.

---

## 참고 논문

| 논문 | 기법 | 적용 |
|------|------|------|
| Clark et al. (2020) | ELECTRA: Pre-training Text Encoders as Discriminators | 모델 선택 근거 |
| Lin et al. (ICCV 2017) | Focal Loss for Dense Object Detection | 불균형 대응 loss |
| Goodfellow et al. (ICLR 2015) | Explaining and Harnessing Adversarial Examples | FGM |
| Wu et al. (ICLR) | R-Drop: Regularized Dropout for Neural Networks | R-Drop |
| Saerens et al. (2002) | Adjusting the Outputs of a Classifier to New a Priori Probabilities | Prior Shift Calibration |
| Gururangan et al. (ACL 2020) | Don't Stop Pretraining | TAPT |
| Park et al. (2021) | KLUE: Korean Language Understanding Evaluation | klue/bert-base 출처 |
| Jawahar et al. (2019) | What Does BERT Learn about the Structure of Language? | LLRD 근거 |

---

## 버전별 진행 요약

| 버전 | 일반대화 데이터 | 모델 | Epochs | 추가 기법 | F1 |
|------|----------------|------|--------|----------|-----|
| v1 | 없음 | KcELECTRA-base-v2022 | 5 | — | 0.72 |
| v2 | SmileStyle 400<br>KakaoChatData 300<br>kor_unsmile 200<br>NSMC 100<br>경계케이스 25 | KcELECTRA-base-v2022 | 5 | K-Fold CV<br>Concatenation | 0.74 |
| v3 | SmileStyle 400<br>KakaoChatData 300<br>kor_unsmile 200<br>NSMC 100<br>경계케이스 165(HNM) | KcELECTRA-base-v2022 | 5 | K-Fold, Focal Loss, LLRD<br>EMA, FGM, R-Drop<br>Prior Shift Calibration<br>Hard Negative Mining<br>Dynamic Class Weight<br>Confidence Fallback<br>Pseudo Labeling | **0.79** |
| v4 | SmileStyle 1200<br>kor_unsmile 800<br>KakaoChatData 500<br>NSMC 500<br>korean_safe 10000<br>kor_nli 10000<br>HNM 165 | klue/bert-base | 5 | v3 전체 기법<br>+ TAPT | 0.72 |

---

## 파일 구성

```
DLthon/
├── README.md
├── figure/
├── ipynb/
│   ├── v1_baseline.ipynb
│   ├── v2_synthetic_data.ipynb
│   ├── v3_hard_negative_mining.ipynb
│   └── v4_tapt.ipynb
├── kaggle dataset/
├── submission/
└── train_data/
```

---

## 환경

| 항목 | 내용 |
|------|------|
| 플랫폼 | Google Colab, Jupyter Notebook |
| GPU | T4 |
| 주요 라이브러리 | transformers, torch, sklearn, sentence-transformers |
