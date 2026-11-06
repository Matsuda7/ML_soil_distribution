# モデルA：有効粘着力c′の空間予測

cの分布作成はφと同じ統合パイプラインから実行します。実行時にc試料の再準備、評価・最終学習、指定深度の格子予測を行います。

```bash
python run_analysis.py --targets c
```

準備済みデータを使って分布工程だけを実行する場合:

```bash
python run_analysis.py --stage distribute --targets c
```

cの説明変数は`config/c/features.jsonc`、評価は`config/validation.json`、N値分布の保存元は`config/c/n_distribution.json`で指定します。深度は共通の`config/model.json`です。
この呼び出しでは実行時に学習と分布予測が行われます。

## 目的値と対象試験

CŪ（CUb、B0523）・CD（B0524）の有効粘着力c′を使用します。
電子納品要領の土質試験結果一覧表では「せん断強さ（有効応力）」はc′、単位kN/m²です。
[項目定義：付属資料6、付6-2](https://www.pref.yamaguchi.lg.jp/uploaded/attachment/189665.pdf)
この標準XMLの項目は任意の応力条件下のせん断応力τではなく、強度定数の切片です。
以前の「この項目は粘着力として扱えない」「粘着力が0件」という解釈を訂正します。
旧試験地点図・旧集計結果は再生成していないため、この訂正がまだ反映されていません。

SOILTESTLISTの標準XMLであることを確認して、shear_strength_effectiveをc′に対応付けます。
明示的c_effectiveがあれば優先し、両者が矛盾する試料は除外します。
全応力の値、UU/CU、一軸圧縮強さからの換算値は混ぜません。
有効粘着力欠損・負値・不正深度・不正座標・重複・DEM範囲外を除外し、理由を保存します。
ゼロの粘着力は有効データとして残します。CŪ/CDの全111試験が学習可能とは限りません。
モデル予測が負値の場合は0へ制限し、件数を報告します。N値の上限50は粘着力へ適用しません。

## 説明変数と領域

config/c/features.jsoncで説明変数を指定します。現在は数値7変数（depth、sample_elevation、slope、curvature、jshis_avs30、jshis_arv、n_value）とカテゴリ3変数（symbol、jshis_jcode、soil_inventory_ssg）が有効です。x、yはコメントアウトされており使用しません。
sample_elevationは学習時にDEM標高−試料中央深度、予測時にDEM標高−指定深度です。
depthは地表からの深度（m）です。地表標高surface_zはcの直接の説明変数に入れません。
n_valueは完了済みφ解析が保存したN値分布から取得します。φ角からの逆算は行いません。
cだけなら最新の完了結果を使用し、φと同時に実行するときはφを先に実行してその結果を使います。
固定深度のN分布を、c試料の深度によらず平面位置の最寄り格子で参照します。
範囲外・欠損格子は補完せず停止します。c専用のN値学習・交差適合・実測N比較は行いません。
保存済みNモデルがc評価孔のN値を学習済みである可能性があるため、評価は保存済み分布を条件とするものです。
上流Nモデルまで含む独立した汎化精度とは扱いません。
詳細は[最新の説明変数](soil_predictors.md)を参照してください。
config/spatial.jsonをφ・γと共用し、同じ領域・10m格子・prop非0を予測します。
config/model.jsonのprediction_depth_mで予測深度を指定します。画像の表示間隔はconfig/c/prediction.json、数値格子はconfig/spatial.jsonで指定します。

## 評価・学習

config/validation.jsonで孔単位・空間評価を切り替えます。
異なるXML名でも同じ平面位置にある試料は同じグループとして扱います。
主評価のvalidationで木数を決め、全採用試料で最終モデルを再学習します。
空間評価は独立した追加評価です。主評価の木数選択には使いません。
評価図の左上にR²を表示します。model/の散布図も主評価のholdoutモデルの結果であり、最終再学習モデルの独立評価ではありません。

## 保存先

- data/training/xgboost/c/model_A/model_dataset.csv：学習表、元項目・試験種別・元XMLを保持
- data/training/xgboost/c/model_A/excluded_records.csv：除外記録
- data/training/xgboost/c/model_A/preparation_summary.json：採用件数・孔数・目的値範囲・出典
- results/YYYYMMDD/c/evaluation/：孔単位・空間評価
- 同model/：最終モデル・前処理・重要度・評価への参照
- 同grid/：c_effective.png、c_effective_kpa.npy、圧縮CSVチャンク
- results/YYYYMMDD/c/run_summary.json：当該実行の状態

単位はkPa（kN/m²と同じ）。図は英語タイトルで、正式な平面直角座標系の軸名に従い横Y・縦X、単位m。

## 使用するN値分布

`config/c/n_distribution.json` の `run_directory: null` は、results/YYYYMMDD/phi/および同日archive/phi_*の
完了済みrun_summary.jsonの更新時刻で最新の実行を選びます。固定したい場合はφの実行ディレクトリを指定します。
φと同時指定した場合は今回のφ結果を優先し、φの失敗時はcを停止します。
詳細は[設定の案内](../config/c/README.md)を参照してください。

```bash
python run_analysis.py --targets phi c
```

旧方式の準備データとは署名が異なるため、次回は準備を含めて実行してください。
今回、学習・推論・解析・テスト・PBS投入は実行していません。
