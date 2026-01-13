import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

VALIDATION_PATH = "data/microarray/GSE63067_disease_vs_control_ratios.txt"


# ------------------------------------------------------------------

def main():
    print(f"파일을 읽어오는 중이다멍... 경로: {VALIDATION_PATH}")

    if not os.path.exists(VALIDATION_PATH):
        print(f"오류: 파일을 찾을 수 없다멍! 경로를 확인해라멍: {VALIDATION_PATH}")
        return

    # 1. 5개 유전자의 탐침(Probe) ID 매핑 (GPL570 플랫폼 기준)
    PROBE_MAP = {
        'PRF1': ['204655_at'],
        'EVI2B': ['206013_at'],
        'CST7': ['204225_at'],
        'GNG2': ['204287_at'],
        'KLHL24': ['227289_at']
    }

    # 2. 데이터 불러오기
    # (파일 첫 줄에 주석(#)이 있을 수 있어서 skiprows=1 처리함)
    try:
        df = pd.read_csv(VALIDATION_PATH, sep='\t', skiprows=1, index_col=0)
    except Exception as e:
        print(f"데이터 로드 중 에러 발생: {e}")
        return

    # 3. 5개 유전자 데이터만 쏙 뽑기
    plot_data = []
    found_genes = 0

    for gene, probes in PROBE_MAP.items():
        for probe in probes:
            if probe in df.index:
                found_genes += 1
                # 샘플별 값을 가져와서 정리
                vals = df.loc[probe]
                for col, val in vals.items():
                    # 컬럼명(Steatosis1, NASH1...)에서 그룹 정보 추출
                    if isinstance(val, (int, float)):  # 숫자인지 확인
                        group = 'Steatosis' if 'Steatosis' in col else 'NASH'
                        plot_data.append({'Gene': gene, 'Group': group, 'LogRatio': val})

    if found_genes == 0:
        print("경고: 매칭되는 유전자를 하나도 못 찾았다멍! ID 형식이 맞는지 확인해라멍.")
        return

    df_plot = pd.DataFrame(plot_data)

    # 4. 결과 시각화 (박스플롯)
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # 0 = 정상인 기준선

    # 예쁘게 그리기
    sns.boxplot(data=df_plot, x='Gene', y='LogRatio', hue='Group', palette='Set2')

    plt.title("External Validation in GSE63067\n(Log Ratio > 0 : Upregulated in Disease)")
    plt.ylabel("Expression Ratio (vs Control)")
    plt.xlabel("Target Genes")
    plt.legend(title='Disease Stage')
    plt.tight_layout()
    plt.show()

    print("그래프가 잘 그려졌나멍? 0보다 위에 있으면 병이 있을 때 늘어난다는 뜻이다멍!")


if __name__ == "__main__":
    main()