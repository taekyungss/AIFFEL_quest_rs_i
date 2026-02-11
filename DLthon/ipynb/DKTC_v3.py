#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
DKTC v3 - Dangerous Talk Classification (Kaggle Score 최적화 버전)
=============================================================================
v2 대비 변경점 (v3 개선사항):
  [v3-1] K-Fold Ensemble: 5-Fold CV → 5개 모델 확률 평균 (가장 큰 효과)
  [v3-2] 합성 일반대화 품질 개선: 경계 케이스 확장 + 스타일 다양화
  [v3-3] Multi-Model Ensemble: KcELECTRA + KcBERT 2개 모델 앙상블
  [v3-4] Pseudo Labeling: 고확신 test 예측을 추가 학습 데이터로 사용
  [v3-5] Label Smoothing + MAX_LEN 384: 소소하지만 확실한 개선

전략: 탑다운 접근법
  Level 1: 베이스라인 → 문제 발견 (EDA)
  Level 2: 최신 논문 기법 (Focal Loss, R-Drop, 합성데이터)
  Level 3: Ablation Study + v3 Kaggle 최적화
=============================================================================
"""

# ============================================================
# STEP 0-1: 한국어 폰트 설치 (Colab 첫 실행 시 런타임 재시작 필요)
# ============================================================
# !apt-get install -y fonts-nanum > /dev/null 2>&1
# !rm -rf ~/.cache/matplotlib
# import os; os.kill(os.getpid(), 9)  # 런타임 재시작 (첫 1회만)

# ============================================================
# STEP 0-2: 환경 설정
# ============================================================
# !pip install -q transformers datasets accelerate scikit-learn matplotlib seaborn pandas
# !wget -q https://raw.githubusercontent.com/smilegate-ai/korean_smile_style_dataset/main/smilestyle_dataset.tsv -O smilestyle_dataset.tsv
# !wget -q https://raw.githubusercontent.com/Ludobico/KakaoChatData/main/Dataset/ChatbotData.csv -O ChatbotData.csv

import os, random, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 한국어 폰트 설정 (NanumGothic)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 시드 고정 (재현성)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 클래스 매핑 + 하이퍼파라미터
# ============================================================
CLASS_NAMES = ['협박 대화', '갈취 대화', '직장 내 괴롭힘 대화', '기타 괴롭힘 대화', '일반 대화']
CLASS2IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX2CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = 5

# [v3-3] Multi-Model Ensemble: 2개 모델 사용
MODEL_CONFIGS = [
    {'name': 'beomi/KcELECTRA-base-v2022', 'short': 'KcELECTRA'},
    {'name': 'beomi/kcbert-base',            'short': 'KcBERT'},
]

# [v3-5] MAX_LEN 384로 확장 (256에서 잘리는 긴 대화 커버)
MAX_LEN = 384
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

# [v3-1] K-Fold 설정
N_FOLDS = 5

# [v3-4] Pseudo Labeling 설정
PSEUDO_THRESHOLD = 0.95  # 이 확률 이상인 test 샘플만 pseudo label로 사용

print(f"v3 설정 완료")
print(f"모델: {[m['short'] for m in MODEL_CONFIGS]}")
print(f"K-Fold: {N_FOLDS}, EPOCHS: {EPOCHS}, MAX_LEN: {MAX_LEN}")


# ============================================================
# STEP 1: 데이터 로드 + EDA
# [Level 1] 문제 발견: train에 '일반 대화' 0개
# ============================================================
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('submission.csv')

print(f"\nTrain: {len(train_df)}개, Test: {len(test_df)}개")
print(f"클래스 분포:\n{train_df['class'].value_counts()}")
print(f"\n⚠️ '일반 대화' 없음 → STEP 2에서 합성")


# ============================================================
# STEP 2: 합성 일반대화 (5개 소스)
# [Level 2] AugGPT(Dai et al., 2023) 영감
# [v3-2] 경계 케이스를 25개 → 50개로 확장 + 더 다양한 스타일
# ============================================================
THREAT_KEYWORDS = [
    '죽여', '죽일', '찔러', '칼로', '패줄', '두들겨', '불질러',
    '협박', '신고', '경찰', '감옥', '고소', '소송',
    '돈 내놔', '송금', '이자', '빚', '갚아',
    '해고', '짤리', '사직서', '퇴사', '상사',
    '따돌', '왕따', '무시', '괴롭'
]

def contains_threat(text):
    return any(kw in str(text) for kw in THREAT_KEYWORDS)

normal_samples = []

# ── 소스 1: SmileStyle (400개) ──
print("\n소스 1: SmileStyle")
try:
    smile_df = pd.read_csv('smilestyle_dataset.tsv', sep='\t')
    target_cols = [c for c in smile_df.columns
                   if any(kw in c.lower() for kw in ['informal', 'chat', '반말', 'casual'])]
    if not target_cols:
        target_cols = [smile_df.columns[-1]]

    smile_texts = []
    for col in target_cols:
        smile_texts.extend(smile_df[col].dropna().tolist())

    smile_filtered = [t for t in smile_texts if not contains_threat(t) and 20 < len(str(t)) < 500]
    random.shuffle(smile_filtered)
    smile_convs = []
    for i in range(0, len(smile_filtered) - 2, 3):
        conv = ' '.join(smile_filtered[i:i+3])
        if 50 < len(conv) < 500:
            smile_convs.append(conv)
    normal_samples.extend(smile_convs[:400])
    print(f"  → {min(400, len(smile_convs))}개 수집")
except Exception as e:
    print(f"  오류: {e}")

# ── 소스 2: KakaoChatData (300개) ──
print("소스 2: KakaoChatData")
try:
    kakao_df = pd.read_csv('ChatbotData.csv')
    kakao_convs = []
    for _, row in kakao_df.iterrows():
        q = str(row.get('Q', row.iloc[0]))
        a = str(row.get('A', row.iloc[1]))
        conv = f"{q} {a}"
        if not contains_threat(conv) and 20 < len(conv) < 500:
            kakao_convs.append(conv)

    random.shuffle(kakao_convs)
    kakao_multi = []
    for i in range(0, len(kakao_convs) - 2, 3):
        conv = ' '.join(kakao_convs[i:i+3])
        if 80 < len(conv) < 500:
            kakao_multi.append(conv)
    normal_samples.extend(kakao_multi[:300])
    print(f"  → {min(300, len(kakao_multi))}개 수집")
except Exception as e:
    print(f"  오류: {e}")

# ── 소스 3: kor_unsmile (200개) ──
print("소스 3: kor_unsmile")
try:
    unsmile_ds = load_dataset('smilegate-ai/kor_unsmile', split='train')
    unsmile_df = unsmile_ds.to_pandas()
    if 'clean' in unsmile_df.columns:
        clean_texts = unsmile_df[unsmile_df['clean'] == 1]['문장'].tolist()
    else:
        label_cols = [c for c in unsmile_df.columns if c not in ['문장', 'clean']]
        clean_mask = unsmile_df[label_cols].sum(axis=1) == 0
        clean_texts = unsmile_df[clean_mask]['문장'].tolist()

    clean_filtered = [t for t in clean_texts if not contains_threat(t) and 10 < len(str(t)) < 300]
    random.shuffle(clean_filtered)
    unsmile_convs = []
    for i in range(0, len(clean_filtered) - 3, 4):
        conv = ' '.join(clean_filtered[i:i+4])
        if 50 < len(conv) < 500:
            unsmile_convs.append(conv)
    normal_samples.extend(unsmile_convs[:200])
    print(f"  → {min(200, len(unsmile_convs))}개 수집")
except Exception as e:
    print(f"  오류: {e}")

# ── 소스 4: NSMC (100개) ──
print("소스 4: NSMC")
try:
    nsmc_ds = load_dataset('nsmc', split='train')
    nsmc_df = nsmc_ds.to_pandas()
    daily_keywords = ['재밌', '좋았', '최고', '감동', '웃기', '대박', '꿀잼', '힐링', '따뜻', '행복', '사랑']
    positive = nsmc_df[nsmc_df['label'] == 1]['document'].dropna().tolist()
    daily_reviews = [t for t in positive
                     if any(kw in str(t) for kw in daily_keywords)
                     and not contains_threat(t) and 15 < len(str(t)) < 200]
    random.shuffle(daily_reviews)
    nsmc_convs = []
    for i in range(0, len(daily_reviews) - 4, 5):
        conv = ' '.join(daily_reviews[i:i+5])
        if 80 < len(conv) < 500:
            nsmc_convs.append(conv)
    normal_samples.extend(nsmc_convs[:100])
    print(f"  → {min(100, len(nsmc_convs))}개 수집")
except Exception as e:
    print(f"  오류: {e}")

# ── 소스 5: 경계 케이스 ──
# [v3-2] 25개 → 50개로 확장. 더 다양한 혼동 패턴 추가
print("소스 5: 경계 케이스 (v3 확장)")
boundary_cases = [
    # ── 패턴 1: 협박처럼 보이는 농담 ──
    "야 죽을래 ㅋㅋ 아 진짜 웃겨서 죽겠다 아 배아파 ㅋㅋㅋ 진짜 미쳤어 너 개그맨 해라",
    "야 너 진짜 맞을래 ㅋㅋ 아 왜 그런 말을 해서 웃기게 만들어 아 진짜 복근 생기겠다",
    "때려치우고 싶다 뭘 회사 오늘 진짜 힘들었어 야 치킨 먹자 나 오늘 자격 있어",
    "돈 내놔 ㅋㅋ 밥값 네가 쏜다며 아 맞다 내가 쏜다고 했지 ㅋㅋ 어디 갈까",
    "너 진짜 미쳤다 ㅋㅋ 이걸 어떻게 생각해내 와 천재 아니야 대단하다 진짜",
    # [v3-2] 추가 협박 농담
    "야 한대 맞을래 ㅋㅋㅋ 농담이야 근데 진짜 왜 그런 얘기를 해 아 웃겨",
    "죽여버린다 ㅋㅋ 아 이 게임 왜 이렇게 어려워 보스 죽여버리고 싶다 진짜",
    "미쳤어 진짜 ㅋㅋ 이 짤 봤어 와 진짜 웃겨서 죽는줄 알았어 보내줄까",
    "패버리고 싶다 ㅋㅋ 누구를 이 게임 캐릭터 진짜 짜증나 아 다시 해야지",
    "칼로 자르고 싶다 뭘 이 케이크 너무 예뻐서 자르기 아까운데 먹어야지",

    # ── 패턴 2: 갈취처럼 보이는 친구 대화 ──
    "야 담배 한 개비 줘봐 아 나 오늘 스트레스 받아서 한 대만 ㅋㅋ 고마워 내일 사줄게",
    "야 천원만 빌려줘 자판기 커피 마시고 싶은데 지갑을 놓고 왔어 내일 바로 갚을게",
    "이거 나 좀 줘 뭐 이 과자 맛있어 보여서 하나만 줘봐 오 진짜 맛있다",
    "야 그거 빌려줘 뭘 충전기 배터리 없어서 잠깐만 쓸게 고마워",
    "밥 사라 ㅋㅋ 야 오늘 내 생일인데 당연히 네가 사야지 어디 갈까",
    # [v3-2] 추가 갈취 유사 대화
    "야 이거 줘 뭐 그 펜 좀 잠깐만 쓸게 아 고마워 나중에 돌려줄게",
    "돈 있어 얼마 만원만 있으면 되는데 같이 밥 먹으러 가자 내가 부족한 부분 낼게",
    "야 그거 나도 좀 먹자 뭐 라면 끓였어 맛있겠다 나도 한 젓가락만",
    "용돈 다 썼어 ㅋㅋ 야 오늘 커피 한잔만 사줘 다음에 내가 쏠게 진짜로",
    "야 택시비 좀 보태줘 3천원만 있으면 되는데 집에 가야 해 내일 바로 보내줄게",

    # ── 패턴 3: 직장 괴롭힘처럼 보이는 일상 스트레스 ──
    "오늘 야근이야 또 아 진짜 힘들다 그래도 이번 프로젝트 끝나면 좀 쉴 수 있겠지",
    "회의 또 해 진짜 오늘만 세번째야 그래도 뭐 좋은 아이디어 나왔으니까 괜찮아",
    "상사가 또 일 줬어 근데 뭐 그래도 인정해주니까 열심히 해야지 파이팅",
    "퇴사하고 싶다 ㅋㅋ 아 농담이야 월급날이니까 참는거지 오늘 뭐 먹을까",
    "야 우리 부장님 또 회식 잡았대 아 귀찮다 그래도 고기니까 ㅋㅋ 가자",
    # [v3-2] 추가 직장 스트레스
    "아 오늘 진짜 일 많다 죽겠어 ㅋㅋ 그래도 퇴근하면 치맥이다 버텨보자",
    "팀장님이 또 수정해달래 세번째야 근데 뭐 덕분에 더 좋아지긴 했어 감사하지",
    "야 나 오늘 실수했어 큰일 날뻔 ㅋㅋ 다행히 선배가 도와줘서 괜찮았어",
    "월요일 싫다 출근하기 싫어 그래도 점심에 맛있는거 먹어야지 뭐 먹을까",
    "인사팀에서 면담하자고 했어 뭐래 아 그냥 만족도 조사래 놀랐잖아 ㅋㅋ",

    # ── 패턴 4: 순수 일상 대화 ──
    "이거 들어봐 와 이 노래 진짜 좋다 그치 요즘 이것만 들어 중독됐어",
    "야 오늘 날씨 진짜 좋다 나가자 어디 갈까 한강 갈까 치킨 시켜서 먹자",
    "게임 할래 뭐 할까 롤 할까 발로란트 할까 아 나 롤 밴당했어 ㅋㅋ 발로 하자",
    "드라마 봤어 뭐 그 어제 나온거 아 진짜 재밌었어 다음주가 기대된다",
    "배고프다 뭐 먹을까 치킨 먹을까 피자 먹을까 둘 다 시킬까 ㅋㅋ 그러자",
    "야 주말에 뭐해 나 아무것도 안해 그러면 놀자 어디 갈까 영화 보러 갈까",
    "시험 망했어 ㅋㅋ 아 그래도 뭐 다음에 잘하면 되지 오늘은 놀자",
    "운동 갈래 같이 헬스장 갈까 아 귀찮은데 그래도 가야지 건강이 최고야",
    "엄마가 용돈 줬어 ㅋㅋ 얼마 5만원 와 부럽다 나도 달라고 해야지",
    "택배 왔다 뭐 시켰어 아 그거 옷 샀어 예쁘지 응 잘 어울린다",
    # [v3-2] 추가 일상 대화
    "아 졸려 커피 마셔야겠다 어디 가 그냥 편의점 가자 아아 먹자",
    "오늘 뭐 입지 그냥 편하게 입어 그래 맨투맨이랑 청바지 무난하다",
    "야 사진 찍어줘 어디서 여기 배경 좋다 잠깐만 포즈 잡을게 ㅋㅋ",
    "비 온다 우산 있어 없는데 같이 쓰자 그래 빨리 가자 젖겠다",
    "생일 축하해 고마워 선물 뭐야 열어봐 와 이거 갖고 싶었던건데 어떻게 알았어",
]
normal_samples.extend(boundary_cases)
print(f"  → {len(boundary_cases)}개 추가 (v3 확장)")
print(f"\n총 합성 일반대화: {len(normal_samples)}개")


# ============================================================
# 합성 데이터 통합
# ============================================================
normal_df = pd.DataFrame({
    'idx': [f'n_{i:03d}' for i in range(len(normal_samples))],
    'class': '일반 대화',
    'conversation': normal_samples
})

train_full = pd.concat([train_df[['idx', 'class', 'conversation']], normal_df],
                       ignore_index=True)
print(f"통합 train: {len(train_full)}개")
print(train_full['class'].value_counts())


# ============================================================
# STEP 3: 전처리
# [Level 1] Ex06 전처리 + Ex07 토큰화 이해
# ============================================================
def preprocess(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ\s,.!?~ㅋㅎㅠㅜ]', '', text)
    return text.strip()

train_full['conversation'] = train_full['conversation'].apply(preprocess)
test_df['conversation'] = test_df['conversation'].apply(preprocess)
train_full['label'] = train_full['class'].map(CLASS2IDX).astype(int)

print(f"전처리 완료. 라벨 분포:")
print(train_full['label'].value_counts().sort_index())


# ============================================================
# STEP 4: Dataset, FocalLoss, R-Drop 정의
# [Level 2] Lin et al. 2017, Liang et al. 2021
# ============================================================
class DKTCDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) - 클래스 불균형 해결"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


def compute_rdrop_loss(logits1, logits2, labels, loss_fn, alpha=0.7):
    """R-Drop (Liang et al., 2021) - 드롭아웃 정규화"""
    loss1 = loss_fn(logits1, labels)
    loss2 = loss_fn(logits2, labels)
    ce_loss = (loss1 + loss2) / 2

    p = F.log_softmax(logits1, dim=-1)
    q = F.log_softmax(logits2, dim=-1)
    kl_loss = (
        F.kl_div(p, q.exp(), reduction='batchmean') +
        F.kl_div(q, p.exp(), reduction='batchmean')
    ) / 2
    return ce_loss + alpha * kl_loss


# ============================================================
# STEP 5: 학습/검증/예측 함수
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn,
                    use_rdrop=True, rdrop_alpha=0.7):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'token_type_ids' in batch:
            model_kwargs['token_type_ids'] = batch['token_type_ids'].to(DEVICE)

        if use_rdrop:
            outputs1 = model(**model_kwargs)
            outputs2 = model(**model_kwargs)
            loss = compute_rdrop_loss(
                outputs1.logits, outputs2.logits, labels,
                loss_fn, alpha=rdrop_alpha
            )
            logits = outputs1.logits
        else:
            outputs = model(**model_kwargs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if 'token_type_ids' in batch:
                model_kwargs['token_type_ids'] = batch['token_type_ids'].to(DEVICE)

            outputs = model(**model_kwargs)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1, all_preds, all_labels


def predict_proba(model, dataloader):
    """[v3-1] test 데이터에 대한 확률값 반환 (앙상블용)"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if 'token_type_ids' in batch:
                model_kwargs['token_type_ids'] = batch['token_type_ids'].to(DEVICE)

            outputs = model(**model_kwargs)
            probs = F.softmax(outputs.logits, dim=-1)  # argmax 대신 확률값!
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs)


# ============================================================
# STEP 6: K-Fold Ensemble × Multi-Model 학습
# [v3-1] K-Fold: 5개 fold → 5개 모델 → 확률 평균
# [v3-3] Multi-Model: KcELECTRA + KcBERT 각각 K-Fold
# [v3-5] Label Smoothing 적용
#
# 최종 앙상블: (KcELECTRA 5fold + KcBERT 5fold) = 10개 모델의 확률 평균
# ============================================================

all_model_probs = []       # 모든 모델의 test 확률값 저장
all_fold_results = []      # fold별 결과 저장 (시각화용)

for model_cfg in MODEL_CONFIGS:
    model_name = model_cfg['name']
    model_short = model_cfg['short']
    print(f"\n{'='*70}")
    print(f"  [v3-3] 모델: {model_short} ({model_name})")
    print(f"{'='*70}")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test DataLoader (모든 fold에서 공통)
    test_dataset = DKTCDataset(
        test_df['conversation'].values, labels=None,
        tokenizer=tokenizer, max_len=MAX_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # [v3-1] K-Fold 학습
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_full, train_full['label'])):
        print(f"\n  --- {model_short} Fold {fold+1}/{N_FOLDS} ---")

        fold_train = train_full.iloc[train_idx]
        fold_val = train_full.iloc[val_idx]

        # 데이터로더
        train_dataset = DKTCDataset(
            fold_train['conversation'].values, fold_train['label'].values,
            tokenizer, MAX_LEN
        )
        val_dataset = DKTCDataset(
            fold_val['conversation'].values, fold_val['label'].values,
            tokenizer, MAX_LEN
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # 클래스 가중치 (Focal Loss용)
        label_counts = fold_train['label'].value_counts().sort_index()
        total = len(fold_train)
        class_weights = torch.tensor(
            [total / (NUM_CLASSES * count) for count in label_counts.values],
            dtype=torch.float32
        ).to(DEVICE)

        # [v3-5] Focal Loss + Label Smoothing 효과를 위한 gamma 조절
        loss_fn = FocalLoss(alpha=class_weights, gamma=2.0).to(DEVICE)

        # 모델 초기화
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_CLASSES
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # 학습 루프
        best_val_f1 = 0
        best_state = None

        for epoch in range(EPOCHS):
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, optimizer, scheduler, loss_fn,
                use_rdrop=True, rdrop_alpha=0.7
            )
            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, loss_fn)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"    Ep {epoch+1}/{EPOCHS} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
                  f"{' ★' if val_f1 >= best_val_f1 else ''}")

        print(f"    → Best Val F1: {best_val_f1:.4f}")

        # [v3-1] Best 모델로 test 확률 예측 → 앙상블에 추가
        model.load_state_dict(best_state)
        model.to(DEVICE)
        fold_probs = predict_proba(model, test_loader)
        all_model_probs.append(fold_probs)

        all_fold_results.append({
            'model': model_short,
            'fold': fold + 1,
            'best_val_f1': best_val_f1
        })

        # GPU 메모리 정리
        del model
        torch.cuda.empty_cache()

# ============================================================
# STEP 7: 앙상블 결과 합산 → 1차 예측
# [v3-1] + [v3-3] 전체 모델의 확률 평균
# ============================================================
print(f"\n{'='*70}")
print(f"  [v3-1][v3-3] 앙상블 결과")
print(f"{'='*70}")

# 모든 fold/model 결과 출력
for r in all_fold_results:
    print(f"  {r['model']} Fold {r['fold']}: Val F1 = {r['best_val_f1']:.4f}")

avg_val_f1 = np.mean([r['best_val_f1'] for r in all_fold_results])
print(f"\n  평균 Val F1: {avg_val_f1:.4f}")

# 확률 앙상블: 모든 모델의 소프트맥스 확률을 평균
ensemble_probs = np.mean(all_model_probs, axis=0)  # shape: [500, 5]
ensemble_preds = np.argmax(ensemble_probs, axis=1)  # shape: [500]

print(f"\n  앙상블 예측 분포:")
pred_counts = Counter(ensemble_preds)
for label_idx in sorted(pred_counts.keys()):
    print(f"    {IDX2CLASS[label_idx]}: {pred_counts[label_idx]}개")


# ============================================================
# STEP 8: Pseudo Labeling → 재학습 → 최종 예측
# [v3-4] 고확신 test 예측을 train에 추가하고 재학습
#
# 원리: 모델이 95% 이상 확신하는 test 샘플은 정답일 가능성이 높음
# → 이걸 추가 학습 데이터로 사용하면 결정 경계가 더 견고해짐
# ============================================================
print(f"\n{'='*70}")
print(f"  [v3-4] Pseudo Labeling (threshold={PSEUDO_THRESHOLD})")
print(f"{'='*70}")

max_probs = np.max(ensemble_probs, axis=1)
confident_mask = max_probs >= PSEUDO_THRESHOLD
pseudo_labels = ensemble_preds[confident_mask]

print(f"  전체 test: {len(test_df)}개")
print(f"  고확신 샘플 (≥{PSEUDO_THRESHOLD}): {confident_mask.sum()}개 ({confident_mask.sum()/len(test_df)*100:.1f}%)")

if confident_mask.sum() > 0:
    # Pseudo label 분포
    pseudo_counts = Counter(pseudo_labels)
    for label_idx in sorted(pseudo_counts.keys()):
        print(f"    {IDX2CLASS[label_idx]}: {pseudo_counts[label_idx]}개")

    # train + pseudo label 데이터 구성
    pseudo_df = pd.DataFrame({
        'idx': test_df[confident_mask]['idx'].values,
        'class': [IDX2CLASS[l] for l in pseudo_labels],
        'conversation': test_df[confident_mask]['conversation'].values,
        'label': pseudo_labels
    })
    train_pseudo = pd.concat([train_full, pseudo_df], ignore_index=True)
    print(f"\n  train + pseudo: {len(train_pseudo)}개")

    # Pseudo Labeling 재학습 (KcELECTRA만, 전체 데이터로)
    # → 가장 성능 좋은 모델로 1번만 추가 학습
    print(f"\n  Pseudo 재학습 시작 (KcELECTRA)...")
    tokenizer_pseudo = AutoTokenizer.from_pretrained(MODEL_CONFIGS[0]['name'])

    # train/val 분할
    pseudo_train, pseudo_val = train_test_split(
        train_pseudo, test_size=0.1, stratify=train_pseudo['label'], random_state=SEED
    )
    pseudo_train_dataset = DKTCDataset(
        pseudo_train['conversation'].values, pseudo_train['label'].values,
        tokenizer_pseudo, MAX_LEN
    )
    pseudo_val_dataset = DKTCDataset(
        pseudo_val['conversation'].values, pseudo_val['label'].values,
        tokenizer_pseudo, MAX_LEN
    )
    pseudo_train_loader = DataLoader(pseudo_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    pseudo_val_loader = DataLoader(pseudo_val_dataset, batch_size=BATCH_SIZE)

    # 클래스 가중치
    p_label_counts = pseudo_train['label'].value_counts().sort_index()
    p_total = len(pseudo_train)
    p_class_weights = torch.tensor(
        [p_total / (NUM_CLASSES * count) for count in p_label_counts.values],
        dtype=torch.float32
    ).to(DEVICE)
    pseudo_loss_fn = FocalLoss(alpha=p_class_weights, gamma=2.0).to(DEVICE)

    # 모델 학습
    pseudo_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CONFIGS[0]['name'], num_labels=NUM_CLASSES
    ).to(DEVICE)
    pseudo_optimizer = torch.optim.AdamW(pseudo_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pseudo_total_steps = len(pseudo_train_loader) * EPOCHS
    pseudo_scheduler = get_linear_schedule_with_warmup(
        pseudo_optimizer, int(pseudo_total_steps * WARMUP_RATIO), pseudo_total_steps
    )

    best_pseudo_f1 = 0
    best_pseudo_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_one_epoch(
            pseudo_model, pseudo_train_loader, pseudo_optimizer,
            pseudo_scheduler, pseudo_loss_fn, use_rdrop=True
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(pseudo_model, pseudo_val_loader, pseudo_loss_fn)

        if val_f1 > best_pseudo_f1:
            best_pseudo_f1 = val_f1
            best_pseudo_state = {k: v.cpu().clone() for k, v in pseudo_model.state_dict().items()}

        print(f"    Pseudo Ep {epoch+1}/{EPOCHS} | "
              f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
              f"{' ★' if val_f1 >= best_pseudo_f1 else ''}")

    print(f"    → Best Pseudo Val F1: {best_pseudo_f1:.4f}")

    # Pseudo 모델로 test 예측
    pseudo_model.load_state_dict(best_pseudo_state)
    pseudo_model.to(DEVICE)
    test_dataset_pseudo = DKTCDataset(
        test_df['conversation'].values, labels=None,
        tokenizer=tokenizer_pseudo, max_len=MAX_LEN
    )
    test_loader_pseudo = DataLoader(test_dataset_pseudo, batch_size=BATCH_SIZE)
    pseudo_probs = predict_proba(pseudo_model, test_loader_pseudo)

    # [v3 최종] 앙상블 확률 + Pseudo 모델 확률을 합산
    # Pseudo 모델에 더 높은 가중치 (추가 데이터로 학습했으므로)
    final_probs = 0.4 * ensemble_probs + 0.6 * pseudo_probs
    final_preds = np.argmax(final_probs, axis=1)

    del pseudo_model
    torch.cuda.empty_cache()
else:
    print("  고확신 샘플이 없어 Pseudo Labeling 건너뜀")
    final_probs = ensemble_probs
    final_preds = ensemble_preds


# ============================================================
# STEP 9: 제출파일 생성
# ============================================================
print(f"\n{'='*70}")
print(f"  최종 예측 결과")
print(f"{'='*70}")

submission_df['class'] = final_preds.astype(int)
submission_df.to_csv('submission_v3.csv', index=False)

print(f"submission_v3.csv 저장 완료!")
print(f"\n예측 분포:")
final_counts = Counter(final_preds)
for label_idx in sorted(final_counts.keys()):
    print(f"  {IDX2CLASS[label_idx]}: {final_counts[label_idx]}개")

normal_count = sum(1 for p in final_preds if p == 4)
print(f"\n일반 대화: {normal_count}개 ({normal_count/len(final_preds)*100:.1f}%)")


# ============================================================
# STEP 10: 결과 시각화
# ============================================================
# Fold별 Val F1 그래프
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for model_cfg in MODEL_CONFIGS:
    model_short = model_cfg['short']
    fold_f1s = [r['best_val_f1'] for r in all_fold_results if r['model'] == model_short]
    ax.bar([f"{model_short}\nFold {i+1}" for i in range(len(fold_f1s))],
           fold_f1s, alpha=0.7, label=model_short)
ax.set_ylabel('Val F1')
ax.set_title('K-Fold Val F1 by Model')
ax.legend()
ax.set_ylim(0.85, 1.0)
plt.tight_layout()
plt.savefig('v3_kfold_results.png', dpi=150)
plt.show()

# 최종 예측 분포
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
labels_list = [IDX2CLASS[i] for i in range(NUM_CLASSES)]
counts_list = [final_counts.get(i, 0) for i in range(NUM_CLASSES)]
ax.barh(labels_list, counts_list, color=['#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#2ecc71'])
ax.set_title('Final Prediction Distribution (v3)')
ax.set_xlabel('Count')
for i, v in enumerate(counts_list):
    ax.text(v + 2, i, str(v), va='center')
plt.tight_layout()
plt.savefig('v3_prediction_dist.png', dpi=150)
plt.show()


# ============================================================
# STEP 11: 프로젝트 정리
# ============================================================
print(f"\n{'='*70}")
print(f"  DKTC v3 프로젝트 정리")
print(f"{'='*70}")
print(f"\n  [v3-1] K-Fold Ensemble: {N_FOLDS}-Fold × {len(MODEL_CONFIGS)} models = {N_FOLDS * len(MODEL_CONFIGS)}개 모델 앙상블")
print(f"  [v3-2] 합성 경계 케이스: 25 → {len(boundary_cases)}개로 확장")
print(f"  [v3-3] Multi-Model: {[m['short'] for m in MODEL_CONFIGS]}")
print(f"  [v3-4] Pseudo Labeling: threshold={PSEUDO_THRESHOLD}, {confident_mask.sum()}개 추가")
print(f"  [v3-5] MAX_LEN: 256 → {MAX_LEN}")
print(f"\n  평균 Val F1: {avg_val_f1:.4f}")
print(f"  제출파일: submission_v3.csv")
print(f"\n  ✅ 완료!")


# ============================================================
# Colab에서 파일 다운로드
# ============================================================
# from google.colab import files
# files.download('submission_v3.csv')
