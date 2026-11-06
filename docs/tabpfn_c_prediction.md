# TabPFNの有効粘着力c：04 分布作成

## 実行

03がtrainedで完了した後、model.jsonのmodel=tabpfnのまま実行します。

```bash
python 04_create_distributions.py --targets c
```

既存の03モデルで動作するようにしています。03のmetricsに学習深度範囲がない旧形式の場合、
03に記録された02の学習表のハッシュを検証して深度範囲だけを読みます。再学習はしません。
モデルの読み込み後にfitは呼びません。学習済み前処理はtransformだけを使います。
TabPFN内部の推論設定（low_memory）は03保存時のまま維持します。

## 使用する結果と条件

pipeline.jsoncのreuse_runs.cがnullなら、同じモデル・対象の最新03結果を選びます。
固定したい場合はresults/YYYYMMDD/cなど03の実行ディレクトリを指定します。
学習済み状態、前処理、元のローカル重み、02のN分布のSHA256を確認してから読み込みます。
重みのパスは保存済み状態のinit_paramsと一致する必要があります。自動ダウンロードは行いません。

- 説明変数、カテゴリの対応、モデル版、重み、推定器数は03保存時の条件。
- 現在のmodel.jsonのdevice・cpu_threads・predict_batch_sizeだけは実行資源として上書き可能。
- 予測深度は現在のmodel.json。チャンク件数・表示間隔・テキスト行列出力は現在のc/prediction.json。
- 格子・座標系・DEM・マスクは03保存条件と一致する必要があります。
- N値は02で保存したn_distribution/n_value.npyを使用。新たなN値推定や異なるphi結果への切替はしません。
- N分布深度とcの予測深度は別に記録します。固定深度N分布を使う前提は02と同じです。

整数の地理コードがCSVでは1101.0、格子では1101となる場合、保存済みカテゴリに一意に対応できる表記だけを合わせます。
予測対象からカテゴリを学習し直したり、曖昧なコードを推測して置き換えたりはしません。

## 全域の処理

既存cの格子読み込み・地理情報付与・prop非0マスク・座標順序・kPa単位・描画を再利用します。
input_chunk_sizeずつ読み、その中をpredict_batch_size（既定256）で推論します。
行列はmemmapで保存し、全対象点が埋まったことを確認してから図を作ります。
CPUでは長時間かかる可能性があります。解像度や学習行数を勝手に減らしません。

## 出力

03の結果ディレクトリ以下に作成します。

```text
results/YYYYMMDD/c/
├── config.json                    # 解決済み予測条件
├── prediction_environment.json    # デバイス・実スレッド数・バッチサイズ等
├── run_summary.json               # 完了/未完了/失敗
└── grid/
    ├── c_effective.png            # 非負補正後の分布図
    ├── c_effective_kpa.npy        # max(raw,0)の行列
    ├── c_effective_raw_kpa.npy    # 補正前の行列
    ├── c_effective_kpa.txt.gz     # 有効設定時のテキスト行列
    ├── chunks/                   # 座標・深度・補正前後のc・使用N値等のCSV.gz
    ├── c_observed_overlay.csv     # 図へ重ねた実測点
    ├── prediction_summary.json
    └── prediction_progress.json
```

図だけはdisplay_spacing_mで設定した表示間隔を使い、数値格子の解像度は変更しません。
元のraw値と負値を0へ補正した結果を区別し、補正件数を記録します。
進捗は推論点数・バッチ数・秒数・点/秒を記録します。モデルが予測済みの点数であり、行列保存完了数とは別です。
途中再開の機能はありません。中断後の再実行は既存grid・設定・環境記録をprediction_archiveへ退避して最初から処理します。
03のモデルとtraining_config.jsonは変更しません。

今回、実際のモデル復元・推論・図作成・テスト・PBS投入は実行していません。コードと構文確認のみです。
