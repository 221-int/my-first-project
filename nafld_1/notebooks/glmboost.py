import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier

VALIDATION_PATH = "data/microarray/GSE63067_disease_vs_control_ratios.txt"

# 5개 유전자 probe (GPL570 기준) — 여기서 EVI2B가 없을 수도 있음
PROBE_MAP = {
    "PRF1": ["204655_at"],
    "EVI2B": ["206013_at"],   # <- 이게 파일에 없을 수 있음 (없으면 자동으로 제외)
    "CST7": ["204225_at"],
    "GNG2": ["204287_at"],
    "KLHL24": ["227289_at"],
}

RANDOM_STATE = 42


def build_features_from_probes(df: pd.DataFrame, probe_map: dict):
    """
    df: rows=probe, cols=samples (값은 log ratio 등)
    probe_map: {gene: [probe1, probe2, ...]}

    반환:
      X_df: rows=samples, cols=genes (존재하는 gene만)
      found_genes: 실제로 매칭된 gene 리스트
      missing_genes: 매칭 못한 gene 리스트
    """
    found_genes = []
    missing_genes = []
    gene_series = {}

    for gene, probes in probe_map.items():
        existing = [p for p in probes if p in df.index]
        if not existing:
            missing_genes.append(gene)
            continue

        # probe가 여러 개면 평균 (토이 예제용)
        vals = df.loc[existing].astype(float)
        gene_series[gene] = vals.mean(axis=0)  # sample별 평균
        found_genes.append(gene)

    if not found_genes:
        return None, found_genes, missing_genes

    X_df = pd.DataFrame(gene_series)  # index: sample, columns: found_genes
    return X_df, found_genes, missing_genes


def infer_labels_from_columns(columns):
    """
    컬럼명에 'Control'이 들어가면 0, 아니면 1로 두는 단순 규칙.
    (네 파일 컬럼명 규칙이 다르면 여기만 바꾸면 됨)
    """
    y = []
    for col in columns:
        if "Control" in str(col):
            y.append(0)
        else:
            y.append(1)
    return np.array(y, dtype=int)


def main():
    print(f"파일을 읽어오는 중이다멍... 경로: {VALIDATION_PATH}")

    if not os.path.exists(VALIDATION_PATH):
        print(f"오류: 파일을 찾을 수 없다멍! 경로를 확인해라멍: {VALIDATION_PATH}")
        return

    # 1) 데이터 로드
    try:
        df = pd.read_csv(VALIDATION_PATH, sep="\t", skiprows=1, index_col=0)
    except Exception as e:
        print(f"데이터 로드 중 에러 발생: {e}")
        return

    print(f"로드 완료! shape={df.shape} (rows=probe, cols=samples)")
    print("probe 예시(상위 5개):", df.index[:5].tolist())
    print("sample 예시(상위 5개):", df.columns[:5].tolist())

    # 2) gene feature 만들기 (없는 유전자는 자동 제외)
    X_df, found_genes, missing_genes = build_features_from_probes(df, PROBE_MAP)

    print("\n[Probe 매칭 결과]")
    print("찾은 유전자:", found_genes)
    print("못 찾은 유전자:", missing_genes)

    if X_df is None or X_df.shape[1] == 0:
        print("\n오류: 매칭된 유전자가 0개라서 모델을 만들 수 없다멍.")
        print("PROBE_MAP이 파일 플랫폼(GPL)과 맞는지 확인해라멍.")
        return

    # 3) 라벨 만들기
    # X_df는 index가 sample이 아니라 DataFrame 생성 시 columns 기준이라 index는 기본(0..)
    # 그래서 sample 이름을 index로 다시 붙여줌
    X_df.index = df.columns
    y = infer_labels_from_columns(X_df.index)

    print(f"\n샘플 수: {X_df.shape[0]}, 사용 유전자 수: {X_df.shape[1]}")
    print("라벨 분포:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})

    if len(np.unique(y)) < 2:
        print("\n오류: 라벨이 한 클래스만 있다멍. 컬럼명 규칙( Control 포함 여부 )을 확인해라멍.")
        return

    # 4) 결측/비정상 값 처리
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.dropna(axis=0, how="any")  # NaN 있는 샘플 제거 (토이용)
    y = infer_labels_from_columns(X_df.index)

    if X_df.shape[0] < 6:
        print("\n오류: 사용할 수 있는 샘플이 너무 적다멍(결측 제거 후).")
        return

    # 5) Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # 6) glmBoost-like 모델 (logistic boosting 느낌)
    # - max_depth=1 : 약한(선형에 가까운) learner
    model = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # 7) 평가
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC (glmBoost-like, 사용 유전자={X_df.shape[1]}개): {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (glmBoost-like, {X_df.shape[1]} genes)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (선택) 어떤 유전자를 썼는지 출력
    print("\n사용한 유전자 컬럼:", list(X_df.columns))


if __name__ == "__main__":
    main()