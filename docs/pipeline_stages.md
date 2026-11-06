# 番号付き工程とモデルの切り替え

## 標準の実行

```bash
python run_analysis.py
```

config/model.jsonのmodelでモデルを指定します。現在はxgboostの全工程と、tabpfnの有効粘着力cの02前処理・03学習検証・04分布作成に対応しています。
tabpfnのphi/gammaは未実装です。別モデルによる代替実行は行いません。
[TabPFN用c学習表](tabpfn_c_preparation.md)を参照してください。
config/model.json、共通のvalidation.json・spatial.json、data/はモデル間で共有できる配置です。

## 工程

| 番号 | ファイル | 内容 |
| --- | --- | --- |
| 01 | 01_collect_data.py | データ収集・配置 |
| 02 | 02_prepare_data.py | 前処理と学習表の作成 |
| 03 | 03_train_validate.py | 学習・検証・最終モデルと前処理の保存 |
| 04 | 04_create_distributions.py | 学習済みモデルを読み込み、分布CSV・行列・図を作成 |

02・03・04はmodule/model_backend.pyを通してモデル固有のフックへ接続します。
現在の接続先はmodule/backends/xgboost.pyです。共通の収集・出力・工程制御はmodule/にあります。
将来、モデル固有の前処理が必要ならそのモデルのprepareを実装し、共通前処理は再利用します。
旧xgboost_analysis/ディレクトリは廃止しました。

## 不要な工程をスキップ

config/pipeline.jsoncで指定します。ファイル自体は削除しません。

```json
"stages": {
  "01_collect": false,
  "02_prepare": false,
  "03_train": true,
  "04_distribute": true
}
```

この例は収集済み・前処理済みデータを使い、学習と分布作成を行います。
既存の準備署名と設定が一致しなければ停止します。スキップした工程を自動で実行し直しません。

03もfalseにすると、04は保存モデルを読むだけです。再学習や前処理のfitはしません。

```json
"stages": {
  "01_collect": false,
  "02_prepare": false,
  "03_train": false,
  "04_distribute": true
},
"reuse_runs": {
  "phi": "results/YYYYMMDD/phi",
  "c": null,
  "gamma": null
}
```

reuse_runsのnullは同じ対象・モデルで03が完了した最新結果を選択します。
保存先のtraining_status.jsonを検証し、異なるモデルやγの湿潤/乾燥の取り違えを防ぎます。
旧結果は、完了状態・config.json・最終モデル・前処理・metrics.jsonの最終学習件数を確認できる場合に限り再利用します。
確認できなければ03を一度実行してください。再利用した旧結果には学習設定のコピーと互換記録を保存します。

既存のモデル・前処理・説明変数は保存時の設定を使い、予測深度・チャンク件数・表示条件などは現在の設定を使います。
格子・DEM・マスク・座標系が学習保存時の条件と違う場合は、黙って無視せず停止します。
同じ結果ディレクトリで04を再実行する場合、以前のgridと設定・状態をprediction_archive/へ退避します。

## φとcの順序

φ・cを同時指定した場合、φの03→04を終え、そのN値分布を使ってcの03→04へ進みます。
番号順は対象ごとの処理順です。この依存関係のため、全対象の03をまとめて先に実行する構成にはしていません。
φの04を無効にした状態で同じ実行のcを進めると、必要なN値分布がないためcを停止します。
cだけを指定すれば既存の最新完了N値分布を03で使用します。
04だけの再利用時、φも指定されていれば今回のφ分布をcの推論に渡します。
φを指定しなければ、c学習時に保存したN値分布を再利用します。学習元と推論元の記録を残します。

## 単独工程と実行計画

```bash
python run_analysis.py --plan
python 01_collect_data.py
python 02_prepare_data.py
python 03_train_validate.py --targets phi
python 04_create_distributions.py --targets phi
```

番号付きファイルと--stageは、設定のtrue/falseに関係なくその工程だけを明示実行します。
--planは設定を表示するだけで、学習・推論・結果書き込みをしません。
通常の入口はrun_analysis.pyです。run_distribution.pyとplot_test_locations.pyは削除しました。

今回、解析・テスト・図の作成・PBS投入は実行していません。
