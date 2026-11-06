# TabPFN用の有効粘着力学習表（02のみ）

config/model.jsonのmodelをtabpfnに変更し、リポジトリ直下で次を実行します。

```json
"model": "tabpfn"
```

```bash
python 02_prepare_data.py --targets c
```

02だけを明示実行するため、pipeline.jsoncの03・04のフラグに関係なく学習・分布作成はしません。
02の前処理にはTabPFNのパッケージ・重みは不要で、
インストール・認証・ダウンロードは行いません。03は[学習・検証ガイド](tabpfn_c_training.md)を参照してください。04は[分布作成ガイド](tabpfn_c_prediction.md)を参照してください。phi/gammaは未実装です。
通常のrun_analysis.pyで実行するならstagesの01/03/04をfalse、02だけtrue、targetsを[c]にしてください。

## 共通データの作成

既存c_model_a.prepareを出力先だけ変えて再利用します。CU-bar/CDの実測有効粘着力（kPa）、
不正値・重複・DEM外・学習深度の除外、地形/地質/土壌区分の付与と孔グループを共通化しています。
共有設定はconfig/c/features.jsonc・config/model.json・config/validation.json・config/spatial.jsonです。
説明変数を勝手に増減せず、合成データや疑似目的変数も追加しません。

n_valueが選択されていれば、config/c/n_distribution.jsonで指定する完了済みN分布を参照します。
最新の完了済み分布がなければ停止します。固定深度のN分布をXY位置で対応させ、深度と参照元を記録します。
各試料深度に合わせたN再推定ではありません。元N値CSVが現在の格子より広い場合は、範囲外のN格子だけを切り出し対象から外します。c試料を削減する処理ではありません。
c試料の位置に対応するN格子がなければ、補間や試料削減をせず停止します。
N分布の上流学習がcの評価孔から独立しているとは限らず、独立精度を保証する学習表ではありません。

## 出力

```text
data/training/tabpfn/c/YYYYMMDD_HHMMSS_ID/
├── model_dataset.csv   # 03へ渡す唯一の学習表（目的値・説明変数・出典）
├── metadata.json       # 条件・列情報・準備サマリー・N分布情報・ハッシュを集約
├── excluded_records.csv # 除外した試料と理由
├── status.json         # prepared / insufficient_data / failed
└── n_distribution/
    └── n_value.npy     # 同じN分布を04でも使うため保持（N値を選択した場合）
```

model_dataset.csvのうち、metadata.jsonのfeature_columnsを説明変数、target_columnを目的変数として使用します。
row_id・boring_id・座標・出典等は表に残しますが、指定していない列を勝手に追加の説明変数にはしません。
カテゴリ列はmetadataのcategorical_featuresに従って文字列として読みます。欠損は空欄のままです。
数値変換不能値・無限大は欠損に統一し、補完・標準化・one-hotのfitは行いません。
検証分割は02では作りません。03実装時に孔・空間分割し、学習型前処理は訓練側だけでfitします。
カテゴリ列番号はfeature_columnsの順序に対応し、row_idを数えません。

X.csv・y.csv・row_metadata.csvは重複するため作成しません。
shared/の中間学習表・準備サマリーは、完成表とmetadataへ統合後に削除します。
N分布のsource.json・read_summary.jsonもmetadata内のn_distribution_sourceへ集約後に削除します。
入力CSVごとのハッシュ・行数はn_distribution_source.chunks、全体の読み込み/範囲外件数はn_distribution_source.read_summaryに残します。
保存したN行列のハッシュもoutput_sha256に記録します。

既存の旧出力ディレクトリは変更しません。今後の実行で作る新しいディレクトリに適用します。
失敗した場合はmetadataに失敗条件を記録し、中間ファイル・診断JSON/CSVを残します。
完成した学習表のstatusがpreparedであることを確認してから次工程へ渡してください。
03はこの学習表とmetadataを入力に使います。モデル版・ローカル重みはconfig/model.jsonのtabpfnで指定します。

今回、前処理・テスト・学習・推論・解析は実行していません。

## N値分布の検査エラー

N分布CSVが条件を満たさない場合、画面に失敗条件別の件数・期待範囲・読み込んだ値の範囲を表示します。
作成中の出力ディレクトリのn_distribution/に次を保存します。

- validation_error.json：参照CSV、失敗した読み込みバッチ、条件別件数、期待値、実際の範囲、各条件の具体例（最大5件）。
- validation_error_rows.csv：条件名・元CSVの行番号（ヘッダーを1行目と数える）・x・y・予測N値。

対象は、数値でない座標/非有限座標、対象範囲内の格子からのずれ、数値でないN/非有限N、
N<0、N>50、同じバッチ内の重複、既に読み込んだバッチとの重複です。
最初に異常を検出したバッチで停止するため、件数・範囲はファイル全体の集計ではありません。
検査基準を緩めたり、異常値を補正・削除して処理を続けたりはしません。

## 元N値CSVが対象領域より広い場合

φのN値CSVは入力格子全体を含み、現在のprop範囲より広い場合があります。
現在の対象格子に入らない既知の座標は読み込み対象から外し、処理を継続します。
対象領域内のN値・座標・重複の検査基準は維持し、数値化不能/非有限の座標は範囲外と決めつけずエラーにします。

入力行数・使用格子数・領域外格子数は、metadata.json内のn_distribution_source.read_summaryへ保存します。
処理途中・失敗時はn_distribution/read_summary.jsonやsource.jsonが残る場合があります。
失敗時のvalidation_error.jsonでは、ignored_countsに領域外として読み飛ばした件数、countsに失敗条件を記録します。
元CSVの行番号は切り出し前の番号を維持します。診断の値の範囲は読み込みバッチ全体の有限値の範囲です。
N分布が対象領域と一切重ならなければ停止します。c試料が範囲外・欠損格子に当たる場合も従来どおり停止します。
