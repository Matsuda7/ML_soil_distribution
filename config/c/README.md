# c（有効粘着力）の設定

XGBoostのcモデルA用です。独立した空間CV（`--stage compare`）はconfig/xgboost/evaluation.jsonを使用します。

```text
config/c/
├── features.jsonc       # cの説明変数
├── prediction.json     # チャンク・表示・XGBoost等
├── n_distribution.json # 使用する保存済みN値分布
└── README.md
```

## N値は保存済み分布から取得

c専用のN値推定モデルは学習しません。旧n_estimation/設定は廃止しました。
φ解析が保存した `grid/chunks/N_value_10m_*.csv.gz` の `predicted_n_value` を使います。
φの角度からN値を逆算することはありません。

`n_distribution.json` の設定は一つです。

```json
{"run_directory": null}
```

- `null`：results/YYYYMMDD/phi/と同日archive/phi_*から、完了済みrun_summary.jsonの更新時刻が最新の実行を選ぶ。
- パス文字列：例えば `"results/20261029/phi"`。このφ実行のN値分布を固定して使う。
- φとcを同じ実行で指定：上の設定より今回のφ結果を優先する。φを先に実行し、失敗時はcを停止する。古い結果へ切り替えない。

参照先には完了状態、config.json、N値CSVが必要です。比較用ディレクトリや未完了の部分分布は対象外です。
コピーで更新時刻が変わった場合や再現実験では、run_directoryで保存元を固定してください。

## 位置・深度の対応

各c試料に対し平面位置の最寄り格子からN値を取得します。同距離の境界では添字の大きい格子を選びます。
**全試料が同じ固定深度のN分布を参照し、各試料の深度に合わせたN値再推定はしません。**
N分布の深度とc分布の予測深度が異なる場合も、指定された固定深度のN分布を使用して両方を記録します。
座標系・解像度・格子範囲はcと一致する必要があります。元N値CSVの領域外格子は切り出して件数を記録します。c試料が範囲外・欠損格子に当たる場合は、補間や試料削減をせず停止します。
全域を一括CSVとして読み込まず、チャンクからc実行内のmemmapへ保存します。

## 変更するファイル

| 内容 | 編集先 |
| --- | --- |
| cの説明変数 | features.jsonc。現在の選択は変更していない |
| c評価の方式・分割比率・シード | 共通のconfig/validation.json |
| 入力チャンク・表示間隔・テキスト出力・XGBoostパラメータ上書き | prediction.json |
| 使用するN値分布 | n_distribution.json |
| 学習深度・c分布の予測深度 | 共通のconfig/model.json |
| 格子・領域・DEM | 共通のconfig/spatial.json |

設定中のパスはリポジトリ直下からの相対パスです。prediction.jsonのmodel_configは説明変数ファイルを指します。

## コマンド（今回未実行）

```bash
# 保存済み最新N値分布を使用
python run_analysis.py --targets c

# φ側のN値・φ分布を先に作り、そのN値分布を使用
python run_analysis.py --targets phi c
```

`--targets c phi` の順でも内部でφを先にします。準備工程ではcのN値列は未設定とし、
分布工程でφ完了後にN値を取り込み、cの評価・最終学習・分布推定へ同じ分布を使用します。
旧方式の準備署名は変わるため、初回は準備を含む上記コマンドを使ってください。

## 評価と保存内容

cの学習・検証・テスト分割は維持します。ただし保存済みN分布の学習にc評価地点の実測Nが
含まれている可能性があります。保存済みN分布を条件としたcの評価であり、上流N値モデルまで含む
独立した精度評価とは扱いません。旧方式の交差適合Nとの精度を無条件に比較しないでください。
実測N条件との比較は行いません。

results/YYYYMMDD/c/以下に保存します。

- `n_distribution/source.json`：参照元、N分布深度、元設定、CSVのSHA256、選択方法。
- `n_distribution/n_value.npy`：そのc実行で使用したN値分布のコピー。
- `final_training_n_features.csv`：各試料のN値、N分布深度、最近傍格子への距離。
- `evaluation/<mode>/metrics.json`：c評価値と、保存済みN分布を使用した評価条件。
- `grid/chunks/`：c予測に加えて使用N値・N分布深度・格子対応距離。

既存の解析結果・学習データはこのコード変更では削除・上書きしていません。
