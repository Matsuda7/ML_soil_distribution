# γ（単位体積重量）の設定

通常の解析と二段階モデルの設定を分けています。今回は配置・名前・説明だけを整理し、設定値は変更していません。

```text
config/gamma/
├── README.md
├── features.jsonc           # 通常のγモデルの説明変数
├── prediction.json          # 湿潤/乾燥・学習・評価・分布出力
├── n_comparison/
│   └── settings.json        # N値の入力条件比較（現在は無効）
└── stage2/                  # 独立した二段階モデル用
    ├── baseline.json        # 基準モデル
    ├── terrain.json         # 地形特徴量の比較
    ├── geology.json         # 地質特徴量の比較
    ├── solphyj.json         # 土壌代表物性の比較
    ├── integrated.json      # 統合した特徴量の比較
    ├── f3_distribution.json # F3＋深度の分布推定
    └── f4_distribution.json # F4_AVS_JCODE＋深度の分布推定（既定）
```

## 普段の編集先

| 変更内容 | ファイル・項目 |
| --- | --- |
| 説明変数 | `features.jsonc` の numeric_features / categorical_features |
| 湿潤・乾燥 | `prediction.json` の gamma_kind（wet / dry） |
| γの評価分割・シード | 共通の `config/validation.json` |
| XGBoostの学習条件 | `prediction.json` の xgb_parameters |
| 分布作成の有無 | `prediction.json` の create_distribution |
| 分布に採用するモデル | `prediction.json` の distribution_model |
| チャンク件数・表示間隔・テキスト出力 | `prediction.json` |
| N値条件比較 | `n_comparison/settings.json` |
| 学習深度・予測深度 | 共通の `config/model.json` |
| 格子・領域・DEM | 共通の `config/spatial.json` |

現在は `gamma_kind: wet`、`distribution_model: no_n`、N比較は `enabled: false`、分布作成は有効です。
`features.jsonc` にn_valueがあっても、現在のno_n経路ではN値を除外します。
現在の有効な通常モデルは数値4変数（depth、sample_elevation、slope、curvature）とカテゴリ3変数です。
N値比較を有効にする場合、上流Nモデルは `config/phi/features.jsonc` を参照します。
cで導入した「最新の保存済みN分布を使う方式」へγを変更したわけではありません。

## 通常の実行コマンド（今回未実行）

```bash
python run_analysis.py --targets gamma --gamma-kind wet
python run_analysis.py --targets gamma --gamma-kind dry
```

`--stage compare` によるXGBoost単独の空間CVは、config/xgboost/evaluation.jsonを使います。
`config/gamma/features.jsonc` の変更は比較経路の説明変数には反映されません。

## 二段階モデル（stage2/）

通常の `--targets gamma` がこのディレクトリ内のモデルをすべて実行するわけではありません。
それぞれ独立したモジュールから使用します。

```bash
# 基準モデル
python -m module.gamma_stage2 --config config/gamma/stage2/baseline.json

# F4＋深度の分布推定（--config省略時も同じ設定）
python -m module.gamma_stage2_distribution --config config/gamma/stage2/f4_distribution.json --stage all

# F3＋深度の分布推定
python -m module.gamma_stage2_distribution --config config/gamma/stage2/f3_distribution.json --stage all
```

各設定内のinput・output・説明変数・学習条件は移動前と同じです。通常モデルのfeatures.jsoncとは独立です。
出力先は各設定のoutputを参照してください。
詳しくは[二段階モデル](../../docs/gamma_stage2.md)・[深度指定分布](../../docs/gamma_stage2_distribution.md)を参照してください。

## 旧名との対応と注意

- model.jsonc → features.jsonc
- n_comparison.json → n_comparison/settings.json
- stage2.json → stage2/baseline.json
- stage2_distribution.json → stage2/f4_distribution.json
- stage2_f3_distribution.json → stage2/f3_distribution.json
- stage2_terrain/geology/solphyj/integrated.json → stage2/内の同名ファイル（stage2_を除く）

パスはすべてリポジトリ直下からの相対パスです。旧名は複製せず移動しました。
独自コマンドで旧--configパスを使っている場合は更新してください。既定コマンドの参照先は更新済みです。
γの準備署名にはN値比較設定の参照パスも含まれるため、既存の準備済みデータで署名不一致が出た場合は
準備を含む通常コマンドを使用してください。今回、データ・結果の再生成やテスト実行はしていません。
