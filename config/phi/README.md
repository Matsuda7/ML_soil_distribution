# φ（内部摩擦角）・N値の設定

既存の解析は、XGBoostでN値を推定し、既存の換算式でφを求めます。
今回はファイル名と説明を整理しただけで、説明変数・学習条件・換算式は変更していません。

```text
config/phi/
├── README.md
├── features.jsonc       # N値モデルの説明変数
└── prediction.json     # 入出力・表示・チャンク処理
```

## 普段の編集先

| 変更内容 | ファイル・項目 |
| --- | --- |
| N値モデルの説明変数 | `features.jsonc` の numeric_features / categorical_features |
| 評価方式・分割比率・シード・CV分割数 | 共通の `config/validation.json` |
| 予測チャンク件数 | `prediction.json` の input_chunk_size |
| 図の表示間隔・N値の表示色範囲 | `prediction.json` の display_spacing_m / n_plot_min / n_plot_max |
| 既存チャンクの扱い | `prediction.json` の existing_chunks |
| ボーリングCSV・DEM外地点の表示設定 | `prediction.json` |
| 土壌図・代表物性表 | 固定入力。module/solphyj_features.py内の定数 |
| 学習深度範囲・N=50の扱い・予測深度 | 共通の `config/model.json` |
| 数値格子の解像度・領域・DEM | 共通の `config/spatial.json` |

features.jsoncはφの実測値を学習する設定ではなく、N値モデルの説明変数設定です。
現在は数値8変数・カテゴリ3変数で、n_value自体はコメントアウトされています。
XGBoostの既定学習パラメータは従来どおりmodule/phi_training.pyのXGB_PARAMETERSです。

`n_plot_max` は図の表示上限で、N値の予測上限を変更する設定ではありません。
`display_spacing_m` も表示の間隔で、数値格子の解像度は共通の空間設定で指定します。

## 土壌データの設定

入力はmodule/solphyj_features.py内のSOIL_ARCHIVE（土壌図ZIP）とREPRESENTATIVE_PROPERTIES（SSG代表物性CSV）で固定しています。
土壌区分だけを使用する場合、代表物性CSVは読み込みません。
この設定は共通の土壌特徴量処理からc・γにも参照されます。φだけに影響する設定ではありません。

## 実行コマンド（今回未実行）

```bash
python run_analysis.py --targets phi
```

cも指定する場合、φを先に実行し、その実行で保存したN値分布をcへ渡します。

```bash
python run_analysis.py --targets phi c
```

XGBoost単独の空間CV（--stage compare）はconfig/xgboost/evaluation.jsonを使用します。
このディレクトリのfeatures.jsoncを変えても、比較経路の説明変数は変わりません。

## 旧名との対応

- model.jsonc → features.jsonc
- 旧土壌入力JSON → module/solphyj_features.py内の定数へ移動
- validation.json → 共通のconfig/validation.json
- prediction.json → 変更なし

旧名は複製せず移動しました。通常コマンドとコード内の参照は更新済みです。
独自スクリプトの旧パス参照は更新してください。設定内のパスはリポジトリ直下からの相対パスです。
今回、学習・推論・解析・テストは実行していません。既存の結果・モデルは変更していません。
