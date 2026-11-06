# TabPFNの有効粘着力c：03 学習・検証

## 実行

02で完成したデータを使います。model.jsonのmodelはtabpfnにします。

```bash
python 03_train_validate.py --targets c
```

これだけで03のみを実行し、04の全域分布は実行しません。04は[分布作成ガイド](tabpfn_c_prediction.md)を参照してください。
run_analysis.pyを使うならpipeline.jsoncの01/02/04をfalse、03をtrue、targetsを[c]にします。
02もtrueなら、その実行で作成した学習表を03へ明示的に渡します。

## model.jsonのTabPFN設定

```json
"tabpfn": {
  "dataset_directory": null,
  "model_version": "v2",
  "model_path": "data/models/tabpfn/tabpfn-v2-regressor.ckpt",
  "device": "auto",
  "cpu_threads": null,
  "n_estimators": 8,
  "predict_batch_size": 256
}
```

- dataset_directoryがnullならdata/training/tabpfn/c/内の最新のprepared結果を選びます。固定したい場合はその実行ディレクトリを指定します。
- 現仮想環境のパッケージ情報は2.2.1だったためv2を既定にしました。プログラムは版を自動更新・切替しません。
- v2はtabpfn==2.2.1、明示的に選ぶv3.5-fastはtabpfn==9.2.0に対応します。Fastのローカル重みは公式ファイル名tabpfn-v3.5-fast-20260909.safetensorsを維持してください。
- 使用する重みは事前に配置してください。model_path=autoは認めず、ローカル重みがなければ停止します。インストールや重みダウンロードは実行しません。
- autoはCUDA優先・なければCPU。cuda指定時にCUDAがなければエラーです。
- cpu_threadsは正の整数またはnull。nullはPBS割当、次にOS/アフィニティから決定し、割当と利用可能数を上限にします。PyTorchとBLASの実行時スレッド数を制限し、前処理プロセス数は1です。
- v2はCPU1000行、CUDA10000行、特徴量500列を上限に検査します。最終学習の全行で検査し、超過時はデータを減らさず停止します。Fastも公式APIのチェックを無効化しません。
- predict_batch_sizeは検証・テストの推論分割数です。学習データ数を減らす設定ではありません。

## 学習表と前処理

02のstatus、モデル種別、単位、行ID、孔ID、目的値、出力ハッシュ、説明変数順、学習深度条件を検査します。
最新結果が壊れていても古い結果へ黙って切り替えません。
入力はmodel_dataset.csvとmetadata.jsonです。旧X.csv/y.csvは不要です。

数値は欠損を保ったまま入力します。カテゴリは訓練側だけでOrdinalEncoderをfitし、未知カテゴリを-1として処理します。欠損はOrdinalEncoderの規則に従い、訓練時に認識した欠損はNaNを使います。
列番号をcategorical_features_indicesへ渡すため、単なる連続量としてカテゴリの大小を学習させる意図ではありません。
外部の標準化・中央値補完・one-hotは行いません。TabPFN内部の前処理もfitへ渡す訓練データのみで作ります。
最終モデルには、全採用行で別にfitした前処理を保存します。

## 検証条件

共通config/validation.jsonの孔単位train/validation/test分割を、現行XGBoostのcと同じsplit_dataで作ります。
行単位分割は孔の混在を防ぐため拒否します。各分割に2行以上が必要です。
**現在のcは孔単位holdoutであり、10 km空間CVやOOFではありません。** n_splits=5だけでcを5foldに変更しません。
空間的な独立性を含む評価が必要なら、両モデルの共通分割を別途実装する必要があります。

評価用モデルにはtrainのみを渡し、validation/testの正解値はfitしません。固定パラメータを使用し、testで調整しません。
validation/testのR²・RMSE・MAE・Bias（予測−実測）を保存します。raw予測とmax(raw,0)の両方を区別します。
現行XGBoostのcは非負補正後で評価するため、同じ補正後の指標も出力します。
最後に全採用行で最終モデルを再学習しますが、そのモデルを独立テスト精度の算出には使いません。

N値は02で固定したN分布を使用します。上流Nモデルがcの評価孔から独立していることは未確認で、完全な独立精度とは記載しません。
同じ解析中の新しいphi分布を差し替える機能はこの03にはありません。必要なphi実行後に02を作ってください。

## 出力

```text
results/YYYYMMDD/c/
├── config.json / training_config.json # 使用した条件
├── training_data.json                 # 02の参照元・メタデータ・ハッシュ
├── environment.json                   # Git SHA/状態・コードハッシュ・パッケージ版・CPU/GPU・重みハッシュ
├── training_status.json / run_summary.json
├── evaluation/borehole/
│   ├── split.csv
│   ├── validation_predictions.csv / test_predictions.csv
│   ├── metrics.csv / metrics.json
│   ├── observed_vs_predicted.png / residual_distribution.png
│   ├── target_distribution.csv / target_distribution.png
│   └── evaluation_model.tabpfn_fit / preprocessor.joblib
└── model/
    ├── model.tabpfn_fit / preprocessor.joblib
    ├── metrics.json / artifacts.json
    └── target_distribution.csv / target_distribution.png
```

同日の既存c結果は従来どおりarchive/へ退避し、モデル種別を記録します。
学習時間・推論時間・分割数・乱数シード・使用特徴量を保存します。
TabPFNにはXGBoostのgain重要度がないため、同じ重要度を捏造せず未計算と記録します。
保存済みモデルの復元には同じパッケージ版、前処理、元のローカル重みが必要です。
04は保存モデル・前処理と02のN分布を使用して分布を作成します。

## 仕様確認と検証状況

- https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/regressor.py
- https://docs.priorlabs.ai/models/accessing-model-weights
- v2のAPIはローカルにインストールされた2.2.1のregressor.pyも静的に確認しました。

今回、学習・推論・テスト・前処理・重み取得・パッケージ導入は実行していません。構文確認のみです。
実測精度・速度・メモリ・モデル保存/復元の実環境動作は未検証です。
