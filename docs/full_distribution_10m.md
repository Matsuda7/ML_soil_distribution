# XGBoostの10 m全域分布

通常のc・φ・γ解析は `python run_analysis.py --targets phi c gamma` を使用します。
この文書は、独立した空間CVの結果からXGBoost全域分布を作成する経路の説明です。
コードはmodule/full_distribution.pyにあります。

## 1. 空間CV

```bash
python run_analysis.py --stage compare --models xgboost --comparison-config config/xgboost/evaluation.json
```

CVが完了した結果を次の--comparison-runへ指定します。
CVの説明変数はconfig/xgboost/evaluation.json、通常解析の説明変数は各物性値のfeatures.jsoncです。
両経路の条件を混同しないでください。

## 2. 全域分布の作成

```bash
# --planは解析・結果書き込みを行わない
python -m module.full_distribution --comparison-run results/YYYYMMDD/comparison/RUN_ID --output results/YYYYMMDD/distribution_10m_5m --stage prepare --plan

# XGBoost最終学習・地理特徴量キャッシュ・全域推論
python -m module.full_distribution --comparison-run results/YYYYMMDD/comparison/RUN_ID --output results/YYYYMMDD/distribution_10m_5m --stage prepare

# 保存済み特徴量からXGBoostを再計算する場合
python -m module.full_distribution --comparison-run results/YYYYMMDD/comparison/RUN_ID --output results/YYYYMMDD/distribution_10m_5m --stage xgboost
```

既存のXGBoost経路と同様、深度5 mでconfig/spatial.jsonの格子・マスクを使用します。
入力CSVはチャンク読み込みし、行列はmemmapへ書き込みます。空間CV完了と入力ハッシュの一致が必要です。
--stage allはprepareと同じ一連の準備・全域予測を行います。引数なしではヘルプのみ表示します。

## 3. 出力

results/YYYYMMDD/distribution_10m_5m/以下に保存します。

- 各対象/xgboost/raw.npy：補正前の分布行列。領域外はNaN。
- 各対象/xgboost/distribution.png：分布図。
- N/xgboost/phi_deg.npyとphi_distribution.png：Nから換算したφ。
- features/：チャンク単位の地理特徴量。
- prepare_status.json：全域処理の完了状態。

全域完了後、既存の監査・一覧図作成コマンドも使用できます。

```bash
python -m module.distribution_overview results/YYYYMMDD/distribution_10m_5m
```

cは表示用にmax(raw,0)、Nはclip(raw,0,50)の補正図を追加できます。元のraw.npyは保持します。
φは既存換算式を使用し、φの実測精度の評価とは扱いません。γは補正しません。
過去の比較結果・未完了出力・停止フラグ・モデル重みは今回のコード整理では変更していません。
今回、解析・推論・図の再生成・テスト・PBS投入は実行していません。
