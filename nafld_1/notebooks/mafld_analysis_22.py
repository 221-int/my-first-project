import gzip
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC

COUNTS_PATH = "data/bulk_rnaseq/GSE126848_Gene_counts_raw.txt"
META_PATH   = "data/bulk_rnaseq/metadata/GSE126848_series_matrix.txt.gz"

TARGET_GENES = ["PRF1", "EVI2B", "CST7", "GNG2", "KLHL24"]
RANDOM_STATE = 42

CV_FOLDS = 5


# 출력 제목 구분선
def print_head(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# 각 샘플이 어떤 샘플인지(제목/설명)을 읽어와서 표(DataFrame)로 정리하는 거
def get_sample_info_series_matrix(path: str) -> pd.DataFrame:
    ids = None
    titles = None
    ch1_lines = []

    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                ids = [x.replace('"', "") for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [x.replace('"', "") for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                ch1 = [x.replace('"', "") for x in line.strip().split("\t")[1:]]
                ch1_lines.append(ch1)

    if ids is None or titles is None:
        raise ValueError("series_matrix에서 !Sample_geo_accession 또는 !Sample_title 을 찾지 못함.")

    df = pd.DataFrame({"Title": titles}, index=ids)

    if ch1_lines:
        ch1_by_sample = []
        for i in range(len(ids)):
            parts = []
            for row in ch1_lines:
                if i < len(row):
                    parts.append(row[i])
            ch1_by_sample.append(" | ".join(parts))
        df["Characteristics"] = ch1_by_sample
    else:
        df["Characteristics"] = ""

    return df



# Characteristics 안에 "disease: NASH" 이런 글자가 존재,, 거기서 disease 뒤에 있는 단어만 뽑는 기능
def extract_disease_status(text: str) -> str:
    t = (text or "").lower()
    m = re.search(r"disease:\s*([^|]+)", t)
    if not m:
        return ""
    return m.group(1).strip()


# 방금 뽑은 라벨을 가지고 정답지 제작
def make_label_from_disease(df_meta: pd.DataFrame) -> pd.Series:
    labels = []
    status_list = []

    for _, row in df_meta.iterrows():
        status = extract_disease_status(row.get("Characteristics", ""))
        status_list.append(status)

        if status in {"healthy", "obese"}:
            labels.append(0)
        elif status in {"nafld", "nafl", "nash"}:
            labels.append(1)
        else:
            title = (row.get("Title", "") or "").lower()
            if ("normal-weight" in title) or ("healthy" in title):
                labels.append(0)
            elif ("obese" in title):
                labels.append(0)
            elif ("nafl" in title) or ("nafld" in title) or ("nash" in title):
                labels.append(1)
            else:
                labels.append(np.nan)

    s = pd.Series(labels, index=df_meta.index, name="Label")
    df_meta["DiseaseStatus"] = status_list
    return s


# 이름이 매칭이 안되면 순서로 매칭 시도
def align_samples(df_counts: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    if len(df_counts.columns.intersection(df_meta.index)) > 0:
        return df_meta

    if df_counts.shape[1] == df_meta.shape[0]:
        df_meta2 = df_meta.copy()
        df_meta2["CountsSampleID"] = df_counts.columns.tolist()
        df_meta2 = df_meta2.set_index("CountsSampleID")
        return df_meta2

    raise ValueError(
        "counts 샘플 수와 meta 샘플 수가 달라서 자동 정렬이 어려움.\n"
        f"- counts columns: {df_counts.shape[1]}\n"
        f"- meta samples: {df_meta.shape[0]}\n"
    )


# counts 파일은 EVI2B 처럼 이름이 없음 그래서 바꿔주고 겹치면 하나로 통합
def ensembl_to_symbol(df_counts: pd.DataFrame) -> pd.DataFrame:
    try:
        import mygene
    except Exception as e:
        raise ImportError(
            "pip install mygene\n"
            f"(원인: {e})"
        )

    mg = mygene.MyGeneInfo()
    ens_ids = df_counts.index.astype(str).tolist()

    res = mg.querymany(
        ens_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=False,
        returnall=False,
        verbose=False,
    )

    ens2sym = {}
    for r in res:
        if isinstance(r, dict) and (not r.get("notfound", False)) and ("query" in r) and ("symbol" in r):
            ens2sym[r["query"]] = r["symbol"]

    if len(ens2sym) == 0:
        raise RuntimeError("Ensembl → Symbol 매핑 결과가 비었음.")

    df = df_counts.copy()
    df["__symbol__"] = df.index.map(lambda x: ens2sym.get(str(x), None))
    df = df.dropna(subset=["__symbol__"]).set_index("__symbol__")
    df = df.groupby(df.index).sum()
    return df


def roc_table(y_true, y_prob, title="ROC Table", n_points=21, save_csv=None, verbose=True):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    df = pd.DataFrame({
        "threshold": thr,
        "FPR": fpr,
        "TPR": tpr,
    })
    df["Specificity"] = 1 - df["FPR"]
    df["YoudenJ"] = df["TPR"] - df["FPR"]

    df = df.sort_values("threshold", ascending=False).reset_index(drop=True)

    if n_points is not None and len(df) > n_points:
        idx = np.linspace(0, len(df) - 1, n_points).round().astype(int)
        idx = np.unique(idx)
        df = df.iloc[idx].reset_index(drop=True)

    if verbose:
        print_head(title)
        print(f"AUC: {roc_auc:.4f}")
        print(df.to_string(index=False))

        best = df.loc[df["YoudenJ"].idxmax()]
        print("\n[추천 cut-off (YoudenJ 최대)]")
        print(best.to_string())

    if save_csv:
        df.to_csv(save_csv, index=False)
        if verbose:
            print(f"\nROC 테이블 CSV 저장: {save_csv}")

    return df, roc_auc


def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return roc_auc


def metrics_table(y_true, y_pred, y_prob=None, title="Metrics Summary", save_csv=None, verbose=True):
    acc = accuracy_score(y_true, y_pred)

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    prec_macro = rep["macro avg"]["precision"]
    rec_macro  = rep["macro avg"]["recall"]
    f1_macro   = rep["macro avg"]["f1-score"]
    prec_w     = rep["weighted avg"]["precision"]
    rec_w      = rep["weighted avg"]["recall"]
    f1_w       = rep["weighted avg"]["f1-score"]

    auc_val = np.nan
    if y_prob is not None:
        try:
            auc_val = roc_auc_score(y_true, y_prob)
        except Exception:
            auc_val = np.nan

    df = pd.DataFrame([{
        "Accuracy": acc,
        "AUC": auc_val,
        "Precision_macro": prec_macro,
        "Recall_macro": rec_macro,
        "F1_macro": f1_macro,
        "Precision_weighted": prec_w,
        "Recall_weighted": rec_w,
        "F1_weighted": f1_w,
    }])

    if verbose:
        print_head(title)
        print(df.round(4).to_string(index=False))

    if save_csv:
        df.to_csv(save_csv, index=False)
        if verbose:
            print(f"\n지표 요약 CSV 저장: {save_csv}")

    return df


def main():
    print_head("실행 환경")
    print("현재 작업 디렉토리:", os.getcwd())
    print("COUNTS_PATH:", COUNTS_PATH)
    print("META_PATH:", META_PATH)

    # 1) 유전자 발현 값이 들어있는 파일 불러오기..
    print_head("1) Counts 로드")
    if not os.path.exists(COUNTS_PATH):
        raise FileNotFoundError(f"Counts 파일이 없음: {COUNTS_PATH}")

    df_counts = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
    print("counts shape:", df_counts.shape)
    print("counts index 예시:", df_counts.index[:5].tolist())
    print("counts columns 예시:", df_counts.columns[:5].tolist())

    # 2) 각 샘플이 정상인지 아닌지 라벨 생성
    print_head("2) Metadata 로드 + 라벨 생성 (disease 기반)")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata 파일이 없음: {META_PATH}")

    df_meta = get_sample_info_series_matrix(META_PATH)
    df_meta["Label"] = make_label_from_disease(df_meta)

    if df_meta["Label"].isna().any():
        bad = df_meta[df_meta["Label"].isna()][["Title", "Characteristics"]]
        print_head("경고: 라벨을 못 만든 샘플(제거됨)")
        print(bad.head(10))
        df_meta = df_meta.dropna(subset=["Label"])

    df_meta["Label"] = df_meta["Label"].astype(int)

    print("meta shape:", df_meta.shape)
    print("라벨 분포:\n", df_meta["Label"].value_counts())

    if "DiseaseStatus" in df_meta.columns:
        print("\nDiseaseStatus 빈도:\n", pd.Series(df_meta["DiseaseStatus"]).value_counts(dropna=False))

    if df_meta["Label"].nunique() < 2:
        raise ValueError("라벨이 한 클래스만 있음.")

    # 3) 샘플 정렬/매칭
    print_head("3) 샘플 정렬/매칭")
    df_meta_aligned = align_samples(df_counts, df_meta)

    X_all = df_counts.T
    common_samples = X_all.index.intersection(df_meta_aligned.index)
    print("교집합 샘플 수(counts raw 기준):", len(common_samples))
    if len(common_samples) == 0:
        raise ValueError("counts 샘플과 meta 샘플 교집합이 0. align_samples 확인하기.")

    #4) 유전자 ID를 PRF1처럼 변경
    print_head("4) Ensembl → Symbol 매핑")
    df_counts_sym = ensembl_to_symbol(df_counts)
    print("symbol counts shape:", df_counts_sym.shape)

    X_sym_all = df_counts_sym.T
    common_samples2 = X_sym_all.index.intersection(df_meta_aligned.index)
    print("교집합 샘플 수(symbol 기준):", len(common_samples2))
    if len(common_samples2) == 0:
        raise ValueError("symbol 변환 후 샘플 교집합이 0. meta 인덱스 정렬을 다시 확인.")

    X_sym_all = X_sym_all.loc[common_samples2]
    y2 = df_meta_aligned.loc[common_samples2, "Label"].astype(int)

    X_sym_log = np.log2(X_sym_all + 1)

    available_genes = [g for g in TARGET_GENES if g in X_sym_log.columns]
    print("TARGET_GENES:", TARGET_GENES)
    print("데이터셋에서 찾은 유전자:", available_genes)

    if len(available_genes) == 0:
        print("컬럼(심볼) 상위 50개 예시:\n", X_sym_log.columns[:50].tolist())
        raise ValueError("TARGET_GENES 중 데이터셋에 존재하는 유전자가 하나도 없음.")

    X_selected = X_sym_log[available_genes]
    print("X_selected shape:", X_selected.shape)

    # 5) Hold-out 평가 (train/test)
    print_head("5) Hold-out 평가 (train/test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y2.loc[X_selected.index],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y2.loc[X_selected.index],
    )
    # SVM으로 1차 분류 성능(정확도)만 빠르게 확인
    svm_model = SVC(probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_train, y_train)
    print(f"SVM Test Accuracy: {svm_model.score(X_test, y_test):.4f}")

    # (전체 데이터) 클래스별(정상/질환) 평균 유전자 발현을 히트맵으로 비교
    df_vis = X_selected.copy()
    df_vis["Label"] = y2
    df_vis["Label_Name"] = df_vis["Label"].map({0: "Control", 1: "Disease"})
    heatmap_data = df_vis.groupby("Label_Name")[available_genes].mean().T


    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="RdBu_r", center=0)
    plt.title("Average Gene Expression (Log2) - Group Mean")
    plt.tight_layout()
    plt.show()

    # (Hold-out) RandomForest로 본격 분류 모델 학습
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # (Hold-out) 혼동행렬(맞춘/틀린 개수) 계산
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred, target_names=["Control", "Disease"], zero_division=0))

    # (Hold-out) 혼동행렬을 히트맵으로 시각화
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Control", "Pred Disease"],
        yticklabels=["True Control", "True Disease"]
    )
    plt.title("Confusion Matrix (Hold-out)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y_test, y_pred)
    auc_holdout = roc_auc_score(y_test, y_prob)

    print("Test Accuracy:", round(acc, 4))
    print("Hold-out AUC:", round(auc_holdout, 4))

    plot_roc_curve(y_test, y_prob, title="ROC (Hold-out) - 5 genes")

    roc_df_holdout, _ = roc_table(
        y_test, y_prob,
        title="ROC Table (Hold-out) - 5 genes",
        n_points=21,
        save_csv="roc_holdout_table.csv",
        verbose=True
    )

    metrics_df_holdout = metrics_table(
        y_test, y_pred, y_prob=y_prob,
        title="Metrics Summary (Hold-out) - 5 genes",
        save_csv="metrics_holdout_summary.csv",
        verbose=True
    )

    print_head("Feature Importance (Hold-out model)")
    importances = pd.Series(model.feature_importances_, index=available_genes).sort_values()
    print(importances)

    # (Hold-out) Feature Importance 막대그래프로 시각화
    plt.figure(figsize=(8, 4))
    importances.plot(kind="barh")
    plt.title("Feature Importance (5 genes)")
    plt.tight_layout()
    plt.show()

    print_head(f"6) StratifiedKFold 교차검증 AUC (k={CV_FOLDS})")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    aucs = []
    y_all_true, y_all_prob, y_all_pred = [], [], []

    # (CV) 각 fold마다 RandomForest 학습 후 확률 예측해서 AUC 계산
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_selected, y2.loc[X_selected.index]), start=1):
        X_tr, X_te = X_selected.iloc[tr_idx], X_selected.iloc[te_idx]
        y_tr = y2.loc[X_selected.index].iloc[tr_idx]
        y_te = y2.loc[X_selected.index].iloc[te_idx]

        m = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        fold_auc = roc_auc_score(y_te, p)
        aucs.append(fold_auc)
        print(f"Fold {fold} AUC: {fold_auc:.4f}")

        pred = (p >= 0.5).astype(int)

        y_all_true.append(y_te.values)
        y_all_prob.append(p)
        y_all_pred.append(pred)

    aucs = np.array(aucs)
    print("\nCV AUC mean:", round(aucs.mean(), 4))
    print("CV AUC std :", round(aucs.std(ddof=1), 4))

    y_all_true = np.concatenate(y_all_true)
    y_all_prob = np.concatenate(y_all_prob)
    y_all_pred = np.concatenate(y_all_pred)

    # (CV merged) 합쳐진 예측으로 혼동행렬 계산
    print_head("Confusion Matrix (CV merged preds)")

    cm_cv = confusion_matrix(y_all_true, y_all_pred)
    print(cm_cv)

    print("\nClassification Report (CV merged preds)")
    print(classification_report(
        y_all_true,
        y_all_pred,
        target_names=["Control", "Disease"],
        zero_division=0
    ))

    # (CV merged) 합쳐진 혼동행렬 시각화
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm_cv,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Pred Control", "Pred Disease"],
        yticklabels=["True Control", "True Disease"]
    )
    plt.title(f"Confusion Matrix (CV merged preds, k={CV_FOLDS})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # (CV merged) 합쳐진 예측 확률로 ROC 커브 시각화
    plot_roc_curve(y_all_true, y_all_prob, title=f"ROC (CV merged preds) - 5 genes (k={CV_FOLDS})")


    # csv로 값 저장
    roc_df_cv, auc_cv_tbl = roc_table(
        y_all_true, y_all_prob,
        title=f"ROC Table (CV merged preds) - 5 genes (k={CV_FOLDS})",
        n_points=21,
        save_csv="roc_cv_merged_table.csv",
        verbose=True
    )

    metrics_df_cv = metrics_table(
        y_all_true, y_all_pred, y_prob=y_all_prob,
        title=f"Metrics Summary (CV merged preds) - 5 genes (k={CV_FOLDS})",
        save_csv="metrics_cv_merged_summary.csv",
        verbose=True
    )

    print(f"\nMerged CV AUC (from pooled fold preds): {auc_cv_tbl:.4f}")


if __name__ == "__main__":
    print("라이브러리 로드 완료!")
    main()