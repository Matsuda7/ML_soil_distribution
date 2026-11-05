# 設定ファイルの編集先

共通条件をこの階層、物性値ごとの設定をc/・phi/・gamma/に配置しています。

| ファイル | 変更できる内容 |
| --- | --- |
| model.json | 使用モデル（model）、学習深度範囲、N=50の扱い、予測深度、実測点の深度許容差 |
| validation.json | 通常のN値・c・γの検証方式、分割比率、乱数シード。n_splitsはNモデルのCVで使用 |
| spatial.json | 予測点CSV、DEM、DEMキャッシュ、座標系、格子間隔、マスク、対象範囲 |
| pipeline.jsonc | 実行対象、01～04の有効/無効、03省略時の再利用先、取り込み設定 |
| c/・phi/・gamma/ | 各物性値の説明変数、チャンク処理、表示、モデル固有設定 |
| xgboost/evaluation.json | 通常解析とは独立した空間CVの条件 |

## 共通の検証条件

通常解析はvalidation.jsonを使用します。以前のphi/validation.json、c/validation.json、
gamma/prediction.json内の同じ検証設定を集約しました。既存の方式・分割比率・シードの値は維持しています。
通常c・γはholdout、NモデルはCV等を行うため、同じ設定ファイルを使っても評価手順がすべて同じになるわけではありません。
二段階γモデルの実験設定と独立した空間CVは、各専用設定を使用します。

独自設定でvalidation_configやvalidationの上書きを指定する旧形式も読み込み可能です。
このリポジトリの標準設定には上書きを置かず、共通ファイルを編集します。

## 空間・予測の設定

共通のprediction.jsonはspatial.jsonを指すだけだったため廃止しました。
通常の読み込みはspatial.jsonを直接読み、対象別prediction.jsonを重ねます。
独自のspatial_config指定がある場合はそのファイルを読みます。
予測深度・実測点の深度許容差は最後にmodel.jsonの値を適用します。

対象別のprediction.jsonは必要です。チャンク件数・図の表示間隔・出力形式や、
γの湿潤/乾燥・学習パラメータなどを変更するためのファイルです。

## 土壌図と試験地点図

土壌データの入力先は固定のため、module/solphyj_features.pyの
SOIL_ARCHIVE・REPRESENTATIVE_PROPERTIESに定義しています。土壌区分を使うc・φ・γで共用します。
代表物性CSVは代表砂分・シルト分・粘土分を選択した場合に読み込み、分類だけなら読みません。

試験地点図の入力・出力先はmodule/test_locations.pyの既定値を使います。
独立コマンド `python -m module.test_locations` の標準入力はdata/raw/soiltest/strength、
標準出力先はresults/test_locations、空間条件はconfig/spatial.jsonです。
必要時のみ--xml-dir・--output-dir・--spatial-configで上書きできます。専用JSONは使用しません。

学習深度・説明変数・土壌図等を変えた場合は、対応する準備・学習をやり直してください。
今回、既存データ・モデル・解析結果は更新していません。

工程別の構成とスキップ設定は[工程ガイド](../docs/pipeline_stages.md)を参照してください。
