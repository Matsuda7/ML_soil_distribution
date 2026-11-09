# TabPFNの有効粘着力c：04 分布作成

## 実行

03がtrainedで完了した後、model.jsonのmodel=tabpfnのまま実行します。

```bash
python 04_create_distributions.py --targets c
```

既存の03モデルで動作するようにしています。03のmetricsに学習深度範囲がない旧形式の場合、
03に記録された02の学習表のハッシュを検証して深度範囲だけを読みます。再学習はしません。
`config/model.json` の `tabpfn.prediction_fit_mode` で04の推論方式を指定します。

```json
"prediction_fit_mode": "fit_with_cache"
```

現在の設定はfit_with_cacheです。未指定の場合は従来のlow_memoryを維持します。
03の学習・検証は引き続きlow_memoryで、変更しません。
キャッシュ方式では保存モデルを復元後、元のinit paramsのfit_modeだけを変更した別モデルを作り、
SHA256検証済みの全学習行でfitしてキャッシュを構築します。重みの最適化やCVの再実行ではありません。
学習済み外側前処理はtransformだけを使い、seed・特徴量順・アンサンブル数・重み・memory_saving_mode=Trueは維持します。
保存済みmodel.tabpfn_fitとpreprocessorは変更しません。キャッシュはメモリ内だけで、実行ごとに構築します。
元の02学習CSVが必要です。欠損・改変・OOM時はエラーとし、行削減や自動フォールバックはしません。
メモリ消費は増えます。元モデルの保存先を退避する前に構築を完了します。
キャッシュ構築時間・前処理時間・使用方式をprediction_environment.json、推論進捗と完了サマリーへ記録します。
キャッシュ構築中は標準出力に開始・完了を表示し、まだ予測進捗ファイルは更新しません。
現時点のキャッシュ対応はTabPFN 2.2.1のみです。
`low_memory`へ戻せば、従来どおり保存モデルを復元してpredictだけを実行します。
バッチサイズは256を維持しています。ベンチマークで確認した条件に合わせて別途設定してください。
キャッシュとバッチ分割で丸め差が出る可能性があるため、同じ条件のベンチマークの予測差も確認してください。
実行中のジョブには設定変更は反映されません。再実行は最初からで、途中再開ではありません。

## 使用する結果と条件

pipeline.jsoncのreuse_runs.cがnullなら、同じモデル・対象の最新03結果を選びます。
固定したい場合はresults/YYYYMMDD/cなど03の実行ディレクトリを指定します。
学習済み状態、前処理、元のローカル重み、02のN分布のSHA256を確認してから読み込みます。
重みのパスは保存済み状態のinit_paramsと一致する必要があります。自動ダウンロードは行いません。

- 説明変数、カテゴリの対応、モデル版、重み、推定器数は03保存時の条件。
- 現在のmodel.jsonのdevice・cpu_threads・predict_batch_size・prediction_fit_modeは実行資源として上書き可能。
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

実モデルによるキャッシュ構築・分布推論・PBS投入はローカルで実行していません。軽量テストで接続と不変条件を確認します。
