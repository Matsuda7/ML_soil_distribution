# XGB_soil_distribution

c（粘着力）、φ（内部摩擦角）、γ（単位体積重量）の分布作成を、1つのGitリポジトリで管理します。
統合先は `XGB_soil_distribution` フォルダです。

## 解析の流れ

```text
python run_analysis.py
        │
        ├─ 1. collect     元データ収集・配置・不足確認
        ├─ 2. prepare     共通前処理、c・φ・γの学習用データ整理
        ├─ 3. train       モデル学習・検証・保存
        ├─ 4. distribute  保存モデルから分布作成
        └─ results/YYYYMMDD/  対象ごとの結果と run_summary.json
```

処理は順番に実行します。c・γ共通のXML抽出は1回です。
φの既存学習・格子予測を移植済みです。γはN値を使わないモデルを実装しました（解析・テスト未実行）。cはCŪ・CDを使うモデルAを実装済みです（未実行）。
未実装・不足データ・失敗を成功扱いせず、結果JSONに記録し、終了コード2を返します。
c（モデルA）・φ・γをtargetsで選択して実行するコードを整備しています。変更後の通し実行は未確認です。

## ディレクトリ

```text
run_analysis.py              # pipeline.jsoncに従って工程を実行
01_collect_data.py           # 収集
02_prepare_data.py           # 前処理
03_train_validate.py         # 学習・検証・モデル保存
04_create_distributions.py   # 保存モデルから分布作成
module/                      # 共通処理と各工程の実装
  backends/xgboost.py        # XGBoost固有の前処理・学習・推論の接続
  model_backend.py           # model.jsonによるモデル選択
config/                      # モデル間で共有する設定
  model.json                 # 使用モデル・学習深度・予測深度
  pipeline.jsonc             # 対象・実行工程・再利用モデル
  validation.json            # 共通の検証条件
  spatial.json               # 格子・DEM・座標系
  c/、phi/、gamma/            # 物性値別の設定
```

[番号付き工程とスキップの設定](docs/pipeline_stages.md)を参照してください。

## 使い方

Python環境へ `requirements.txt` の依存を導入します。

```bash
python -m pip install -r requirements.txt
python run_analysis.py --plan
python run_analysis.py
```

出力対象は `config/pipeline.jsonc` の `targets` で指定します。現在の設定はc（モデルA）・gamma・phiです。
既存の `sources`・`soiltest_bbox` はそのままにして、この項目だけ編集してください。

```json
"targets": ["c", "phi", "gamma"]
```

φのみなら `["phi"]`、φとγなら `["phi", "gamma"]` とします。
`python run_analysis.py` と段階別の入口は、この設定に従ってデータ整理・分布作成を行います。
第1段階の収集は引き続き `sources` に登録した共通データを取り込みます。
対象を外しても過去に出力したファイルは削除しません。

`--targets` を指定した場合は、設定ファイルより優先します。
`--plan` でも同じ優先順位で対象を確認できます。
`targets` 未記載の旧設定は3種類すべてとして扱います。
空リストや `c`・`phi`・`gamma` 以外の名前は解析開始前にエラーになります。
重複指定は1回にまとめます。c・γの分布モデルが未実装である点は変わりません。

段階・対象を指定する例：

```bash
python run_analysis.py --stage collect
python run_analysis.py --stage prepare --targets c gamma
python run_analysis.py --stage prepare --targets phi
python run_analysis.py --stage distribute --targets phi
```

`01_collect_data.py` などの入口も同じ関数を呼びます。
解析パスはプロジェクトの場所から解決するため、別の作業ディレクトリから実行できます。
`--plan` はデータ取り込みもモデル学習も行いません。

## 第1段階：収集とディスク共有

`config/pipeline.jsonc` の `sources` は初期取り込み完了後に空にしています。
通常の収集段階では統合先 `data/raw/` の配置を確認します。追加の取込元がある場合は `sources` に登録してください。
取り込みはハードリンク方式なので、同一ファイルシステム上では大容量データの実体を複製しません。
統合先には通常のファイルとして存在し、元のフォルダ名に依存するシンボリックリンクではありません。
元のファイル名を移動・削除しても、統合先のファイルは残ります。
既存取込先と取込元の内容が異なる場合は、上書きせず停止します。
異なるファイルシステムではハードリンクできないため、自動コピーに切り替えずエラーにします。

元データは不変として扱ってください。ハードリンク先を直接書き換えると同じ実体に影響します。
XMLダウンローダーは一時ファイルから置き換える方式で、既存リンクの実体を直接上書きしません。
収集履歴 `data/raw/collection_manifest.json` には取込元・サイズ・SHA-256を記録します。
元データ取り込み後の解析は、このプロジェクト内のファイルだけを参照します。
旧 `XGB_c_setting`・`XGB_phi_setting` はユーザーの指示で削除済みです。
統合済み384ファイルは削除前後にSHA-256を検証しています。収集履歴の旧パスは来歴として保持します。
取得一覧は `data/boring/boring_list.csv` に保存します。互換用の `data/input_xml` リンクは `data/raw/soiltest` を参照します（XML実体は移動しません）。

KuniJibanのXMLを追加取得する場合:

```bash
python run_analysis.py --stage collect --download
```

通常の収集は既存データを再利用します。DEM・地質図・J-SHIS・ボーリングCSVは既存のローカル資産を取り込みます。
出所や変換仕様が確定していないデータの自動ダウンロードは実装していません。
予測格子 `output4.txt` と領域マスク `prop.txt` は今回の確認時に見つからず、配置待ちです。
`data/raw/grid/` に配置するか、`config/pipeline.jsonc` に取込元を登録してください。
DEMの元ファイル `Z.txt` は統合先では `z.txt` とします。

## 前処理・学習・分布工程の現状

| 対象 | 学習データ整理 | 分布作成 |
|---|---|---|
| φ | 元のSPT測定値整理を移植 | 元のXGBoost N値予測→φ換算を移植。格子・マスクの配置が必要 |
| c | 真の粘着力項目だけを候補表へ。現在は0件 | 定義・単位・教師データ・モデルを次段階で決定 |
| γ | 湿潤・乾燥密度から各targetを作成 | 説明変数・空間予測モデルを次段階で決定 |

γは密度をg/cm³と仮定し、`γ = 密度 × 9.80665` によりkN/m³へ変換します。
せん断強さを粘着力へ読み替える処理はしません。
φ側の既存手法（N値の上限50、孔単位分割、RBFの交差適合、φへの参考換算）は維持します。
詳しい元の解析仕様は [docs/phi_original.md](docs/phi_original.md) に保存しています。
同文書の旧パス・実行コマンドは移植前の記録です。統合後の入口とパスは本READMEが正です。

## Git管理

このフォルダの既存Gitを統合リポジトリとして使います。内側に別の `.git` は作りません。
コード・設定・説明書・テストとデータ用ディレクトリの空プレースホルダーを管理し、元データ・学習データ・モデル・図・ログは除外します。
GitHubの既存remoteは変更していません。コミット・push・GitHub側の名前変更はまだ行っていません。

## 検証

```bash
python -m unittest discover -s tests -v
python -m unittest -v test_extract_strength.py
```

旧互換入口 download_kunijiban.py・extract_strength.py・make_distributions.py・shared_data.py は削除しました。本体は module/ に集約し、テストも本体を直接参照します。
直下の `output/` は廃止しました。過去の生成結果は削除済みです。新しい結果は `results/YYYYMMDD/` に保存します。

## φ予測の再実行

`config/phi/prediction.json` の `existing_chunks` は `"archive"` に設定しています。
予測チャンクが既にある場合、`chunks_previous_*/chunks/` にフォルダを退避し、
通常の `chunks/` に新しく予測します。退避は移動なのでファイル内容を複製しませんが、
新しい予測結果を保存する分のディスク容量は必要です。
`"error"` にすると既存チャンクがある場合は従来どおり停止します。

統合コマンドは再学習するため、旧チャンクの自動再利用は行いません。
再実行は `python run_analysis.py --stage distribute --targets phi` です。
このコマンドでも再学習し、全格子を計算し直します。
既存の `--resume` は同じモデル・入力・設定での継続を利用者が確認した場合だけの機能です。
過去の実行ですでに再学習済みの場合、残った旧チャンクと現在のモデルの一致は保証できません。

## c・γの土質試験地点を可視化

```bash
python -m module.test_locations
```

元XML・出力先は `module/test_locations.py` の既定値を使用します。変更が必要な場合は `--xml-dir`・`--output-dir`・`--spatial-config` で指定できます。
強度試験専用の抽出CSVでは欠落する密度試験も含めるため、元XMLを直接読み込みます。
`results/test_locations/c_gamma_test_locations.png` とPDFに、c関連の強度試験地点とγの密度試験地点を並べて出力します。
`test_locations.csv` に地点別の試験有無・座標、`summary.json` に集計、`excluded_records.csv` に座標欠損による除外を保存します。

輪郭は共通の `config/spatial.json` のprop配置・範囲・格子に従い、0と非0の境界を線にします。
負値や10以上の値も非0として扱い、元格子を間引きません。範囲外の地点は中抜きで表示します。
この線はpropの境界であり、別途取得した海岸線データではありません。
XMLの測地系コード00（日本測地系）・01（JGD2000）・02（JGD2011）を読み、同じ平面座標系に変換します。
コードの出典：[電子納品要領・土質試験結果一覧表データ 表2-4](https://www.maff.go.jp/j/nousin/seko/nouhin_youryou/attach/pdf/doboku-40.pdf)。
不明な測地系は推測せずエラーとします。

初回作成時点：c関連193地点、粘着力cそのもの0地点、湿潤・乾燥密度各184地点。
propの範囲外は全体で24地点。試料深度別の試験は同じ孔・位置に集約します。
図は予測分布ではなく試験地点図です。cはCŪ・CDを使うモデルAを実装済みです（未実行）。γは以下のN値を使わないモデルを実装しました（動作未検証）。


## N値を使わない乾燥・湿潤γモデル

コードを実装しましたが、今回の変更後の解析・学習・テストは未実行です。
`python run_analysis.py` で収集・前処理・学習検証・分布作成を設定に従って実行します。
現在の `config/pipeline.jsonc` の targets は `["c"]` です。γのみなら `["gamma"]` に変更します。
φも作成する場合は `["phi", "gamma"]` に変更します。cはCŪ・CDを使うモデルAを実装済みです（未実行）。

説明変数は `config/gamma/features.jsonc` で選択します。現在のno_n経路では、
depth、sample_elevation、slope、curvatureと、symbol、jshis_jcode、soil_inventory_ssgを使用します。
同ファイルにn_valueがあっても、no_n経路では除外されます。
`config/model.json` の prediction_depth_mを変更できます。
湿潤・乾燥密度[g/cm³]を9.80665倍して単位体積重量[kN/m³]に変換し、`gamma_kind` で選んだ側だけ学習します。
密度のみのXML試験も収集し、欠損・不正値・DEM範囲外の除外記録を残します。
学習表は data/training/gamma/、モデルと結果は results/YYYYMMDD/gamma_wet/ または gamma_dry/ に保存します。
選択した側の grid/gamma_wet.png または gamma_dry.png に、propの0/非0境界を重ねた分布図を出力します。
数値は10m格子のnpyと圧縮CSV、図の表示間隔は初期値100mです。
各実行は新規ディレクトリに保存し、既存結果を上書きしません。
処理状態は対象ディレクトリ内の run_summary.json に保存します。
乾燥γが湿潤γを上回る予測は補正せず、件数をprediction_summary.jsonに記録します。

## N値・γのvalidation切り替え

通常のN値・c・γの検証条件は共通の `config/validation.json` を編集します。

```json
{"mode": "borehole", "test_size": 0.2,
 "validation_size": 0.2, "random_state": 42}
```

- mode=borehole：孔単位の分割。初期設定です。
- mode=row：行単位。同じ孔が学習・評価に混在し得るため未知孔への評価とは区別します。

validation_sizeはtestを除いた残りに対する割合です。分割可能なグループが不足すれば停止します。
主評価のvalidationで木数を選び、その木数で全データに再学習します。
空間検証は削除済みです。以前の評価ファイルは履歴として保持します。


## configの配置

```text
config/
  pipeline.jsonc          # 出力対象・収集設定
  spatial.json           # 共通の格子・DEM・prop・座標系
  validation.json        # N値・c・γの共通検証条件
  c/                     # 詳細はconfig/c/README.md
    features.jsonc       # cの説明変数
    prediction.json      # cの実行設定・他設定への参照
    n_distribution.json  # 保存済みN分布の選択（nullなら最新の完了結果）
  phi/
    features.jsonc       # N値モデルの説明変数
    prediction.json
  gamma/
    features.jsonc
    prediction.json      # γの予測・学習・validation設定
    n_comparison/        # N値条件比較
    stage2/              # 二段階モデルの独立設定
```

φ・c・γはconfig/spatial.jsonを共有します。共通の学習深度・予測深度はconfig/model.jsonで指定します。
設定項目と編集先は[cの案内](config/c/README.md)、[γの案内](config/gamma/README.md)、[φ・Nの案内](config/phi/README.md)を参照してください。
γのobserved_vs_predicted.pngには評価用データのR²を左上に表示します（φには既に表示あり）。
γ分布図のタイトルは英語、軸はメートルです。北を上にした図で、平面直角座標系VIIの
正式な軸名に従い横軸Y（東西）、縦軸X（南北）とします。
内部のデータ列x=東西、y=南北は従来どおりです。図・評価結果の再生成は行っていません。

## 保存先の整理

入力・中間・学習データはdata/、モデル・評価・分布図はresults/、実行ログはlogs/です。
[保存先と旧outputからの移動一覧](docs/output_layout.md)を参照してください。

## N値の導入比較

[対応ルール・比較条件・信頼性指標・実行設定](docs/n_comparison.md)を追加しました。
config/gamma/n_comparison/settings.jsonのenabledで切り替えます。今回の変更後の解析・テストは未実行です。
比較のみの場合はconfig/gamma/prediction.jsonのcreate_distributionをfalseにしてください。

γの採用モデルは `config/gamma/prediction.json` の `distribution_model` で指定します。現在は `no_n`、N値比較は無効です。比較経路の説明は[比較・採用モデルの説明](docs/n_comparison.md)を参照してください。

## c：モデルA

CŪ・CD試験から有効粘着力c′を予測する[モデルA](docs/c_model_a.md)を実装しました。
現在のtargetsはcです。解析・学習・テストは未実行です。c全体を実装待ちとしていた記述はモデルAには当てはまりません。

## モデル作成時の必須図表

寄与度の表・棒グラフと、学習目的値の度数分布図を必ず保存します。
[出力ファイルと算出方法](docs/model_reporting.md)を参照してください。今回の変更後の解析・テストは未実行です。

## 最新のc・γ説明変数

c・γとも試料標高と推定N値を含む9変数に変更しました。[最新設定と標高の定義](docs/soil_predictors.md)を参照してください。旧説明変数の記述よりこちらを優先します。

## γの湿潤・乾燥の選択

`config/gamma/prediction.json` の `gamma_kind` を `"wet"`（湿潤、初期設定）または
`"dry"`（乾燥）に設定します。N値比較・評価・モデル学習・分布作成は選択した側だけ実行します。
共通の前処理では、再利用できるよう湿潤・乾燥両方の学習用CSVを作成します。
実行時に上書きする場合は次のように指定します。

```bash
python run_analysis.py --targets gamma --gamma-kind wet
python run_analysis.py --targets gamma --gamma-kind dry
```

c・φも含めて最初から実行する場合は `python run_analysis.py --gamma-kind wet` を使います。
前処理済みなら `python 04_create_distributions.py --targets gamma --gamma-kind dry` で分布作成だけを実行できます。
選択した種類は各実行の `config.json` と `run_summary.json` に記録されます。


保存例とファイルの役割は [出力構成](docs/output_layout.md) を参照してください。
基盤面の補間・図・テキスト・元点データの作成は廃止しました。

## 深度を指定する二段階の湿潤γモデル

深度を加えた第一段階B・C・Dの疑似γを用い、第二段階のF4_AVS_JCODEにも深度を追加する独立コマンドです。
`config/model.json` の `prediction_depth_m` に地表からの深度（m）を指定します。

```bash
python -m module.gamma_stage2_distribution --stage all
```

学習済みモデルから別の深度の分布を出す場合は、設定を変更して `--stage predict` を指定します。
説明変数・評価結果・出力先は [二段階γの深度指定分布](docs/gamma_stage2_distribution.md) を参照してください。


### 分布図の実測点

N値・γ・cの予測深度は `config/model.json` の `prediction_depth_m`（m）で共通指定します。
`observed_depth_tolerance_m`（m、初期値1.0）は、実測点を重ねる深度の許容幅です。
予測深度±許容幅の両端を含め、同一座標では予測深度に最も近い試料を選び、同距離なら平均します。
学習用に絞る前の元実測データを使用します。ただし値・深度・座標が不正な試料は除外し、cは有効粘着力、γは密度から換算した対象の湿潤／乾燥単位体積重量を使用します。
地図内の実測値を背景と同じカラーマップ・色範囲の黒縁●で表示します。色範囲外の値もCSVには元の値を保存します。
表示点は図と同じディレクトリの `*_observed_overlay.csv` に保存します。これは独立した精度検証ではありません。
既存の予測結果はその計算深度を確認してから再描画してください。深度が異なる場合は予測の再計算が必要です。

## 工程別の実行とモデル選択

`config/model.json` の `model` を `"xgboost"` に設定します。
将来の別モデルはmodule/backends/へ実装し、共通工程の入口はそのまま使用する設計です。
現在はXGBoostのみ実装済みで、未実装モデルを指定すると解析開始前にエラーにします。

```bash
python run_analysis.py
```

実行工程はconfig/pipeline.jsoncのstagesで選択できます。詳細は[工程設定](docs/pipeline_stages.md)を参照してください。
XGBoost単独の独立した空間CVは従来の--stage compareとconfig/xgboost/evaluation.jsonで使用できます。
旧run_distribution.py・plot_test_locations.pyは削除しました。通常の分布作成は04の工程に統一しています。
独立した全域CV結果の分布は `python -m module.full_distribution`、試験地点図は `python -m module.test_locations` で必要時に使用できます。

### cへのN値分布の入力

c専用の上流N値学習は廃止し、完了済みφ解析のN値分布を再利用します。
`python run_analysis.py --targets phi c` はφを先に実行し、その実行で保存したN値分布をcへ渡します。
cだけなら最新の完了済みφ結果を選びます。各試料の深度によらず、固定深度のN分布を平面位置で参照します。
保存元・N分布深度・最近傍格子への距離を記録します。
これは保存済みN分布を条件としたcの評価で、上流Nモデルまで含む独立評価ではありません。
詳細と出力は[config/cの案内](config/c/README.md)を参照してください。
