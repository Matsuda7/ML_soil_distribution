# TabPFN 2.2.1 CPU推論ベンチマーク

本番のPython・設定・学習済みモデル・分布・進捗・run_summaryを変更しない独立機能です。
自動ダウンロード、パッケージ更新、本番ジョブ停止、ジョブ投入は行いません。
本体とPBSスクリプトはPBS_JOBIDがない環境では測定開始を拒否します（--helpと軽量テストは可能）。
本番と同じノードへの同時配置は避けてください。PBSのキュー・ノード指定はorcaの運用に従います。

## 実装調査と比較方法

実際の設定は `config/model.json`（`config/c/model.json` は存在しません）。
本番04は `prediction_config` → `load_assets` → `predict_batches` を使い、
保存済み `.tabpfn_fit` のexecutorと前処理器を復元します。
本ベンチマークも同じ復元・ハッシュ確認・カテゴリ処理・CPU割当・バッチ推論を再利用します。
本番の `predict_cohesion` / `predict_grid` は呼びません。

インストール済み公式TabPFN **2.2.1** の以下を静的に確認しました。

- `regressor.py`: `low_memory` は推論ごとに前処理、`fit_with_cache` はfit時に前処理とKVキャッシュを構築。
- `model_loading.py:load_fitted_tabpfn_model`: init paramsだけでなく保存されたexecutorを復元。
  復元後に `fit_mode` 属性だけ変えてもexecutorは切り替わりません。
- `base.py:create_inference_engine` / `inference.py:InferenceEngineCacheKV.prepare`:
  CPUを排除する分岐はなく、device=cpuで構築する経路があります。
  元の重みからアンサンブルごとにモデルをdeepcopyするため、メモリが増えます。
- `memory_saving_mode=True` はキャッシュengineにも渡され、推論時のメモリ節約処理に使われます。
  キャッシュ構築時には学習コンテキスト全体が必要で、KVキャッシュ自体のメモリをなくすものではありません。

確認対象はローカル `.venv-tabpfn/lib/python3.12/site-packages/tabpfn/` の2.2.1ソースです。
公式配布元: https://pypi.org/project/tabpfn/2.2.1/ 、公式リポジトリ: https://github.com/PriorLabs/TabPFN 。
新しい版のAPI仕様を2.2.1に流用しません。CPUでの実動作・必要RAM・数値一致はorca実測で判定します。

安全なモード変換の公開APIは確認できないため、キャッシュ条件は別モデルを作ります。
復元した元モデルの `get_params(deep=False)` をコピーし、**fit_modeだけ**を変更します。
重み・カテゴリ列インデックス・乱数seed・n_estimators・memory_saving_mode・精度設定などは保存時のままです。
学習CSVのSHA256と全行数を確認し、03の保存済み最終前処理器で **transformだけ**を行い、
その全行とtargetで別モデルをfitします。前処理器の再fit・学習行の削減・CV・精度の再評価はしません。
本番の保存状態を直接編集せず、キャッシュモデルを本番パスへ保存することもありません。

## 実験

A: low_memoryで256 / 512 / 1024 / 2048点バッチを比較。
B: 同じ各バッチについてlow_memory / fit_with_cacheを比較。

最初に独立したlow_memory / 256の基準予測を作り、各条件を3回ずつ測ります。
反復ごとに8条件をseed固定でシャッフルします。基準取得だけは比較のため先行します。
各条件を**独立プロセスで順番に**実行し、キャッシュの持ち越しとメモリ高水準値の混入を避けます。
並列のモデル推論プロセスは増やしません。モデル内部は既存PyTorch CPUスレッド並列です。
ウォームアップは既定256点で、測定推論とは分けます。

## 実際の予測点の抽出

本番の格子入力をチャンクで走査し、同じgrid_indicesとprop非0マスクで対象を選別します。
全域から乱数priority方式で既定10,000セルを抽出し、全条件で座標・行順・前処理後配列を固定します。
格子走査は全域を読みますが、全座標を一括で保持しません。格子サイズのマスクとseen配列は保持します。
重複座標・対象セルの欠落は本番同様エラーです。
本番と同じDEM・地質・J-SHIS・土壌情報・02で固定したN分布から特徴量を生成します。
DEMキャッシュの作成先も専用出力内に変更するため、本番キャッシュに書き込みません。
全域の特徴量付与や本番分布出力は実行しません。

現在の本番04は**指定深度の2次元格子**です。三次元全域を自動生成しているわけではありません。
`depths_m: null` は現在の本番設定の深度1つ、例えば5 mを使用します。
複数深度を比較したい場合は `"depths_m": [1, 5, 10]` のように指定すると、
抽出セルへ均等に深度を割り振り、同じ特徴量生成を使います。約10,000点は全深度の合計です。
Nは元の固定深度分布のままで、深度ごとのNを新規推定しません。
`total_prediction_rows` は概算の対象件数で、三次元全域の正確な総数が分かれば変更してください。

## 設定

`benchmark/config.json`:

- `run`: orcaにある03の結果ディレクトリ。既定 `results/20261106/c`。
- `cpu_threads`: nullならPBS割当とCPUアフィニティから決定。整数指定も割当を超えません。
- `prediction_rows`: 10000。学習行数とは別で、学習データは減らしません。
- `batch_sizes`: [256, 512, 1024, 2048]。基準256は必須。
- `fit_modes`: ["low_memory", "fit_with_cache"]。low_memoryは必須。
- `repeats`: 3、`warmup_rows`: 256、`seed`: 42。
- `total_prediction_rows`: 20000000、`timeout_seconds`: 各子処理7200秒。
- `atol` / `rtol`: 予測値一致判定の許容差。既定各1e-5。小さな丸め差を無条件にゼロ扱いしません。

CLIから `--run`, `--cpu-threads`, `--prediction-rows`, `--batch-sizes` も上書きできます。
例（**計算ノードで実行**）:

```bash
.venv-tabpfn/bin/python -u benchmark/benchmark_tabpfn.py \
  --config benchmark/config.json --cpu-threads 16 --prediction-rows 10000 \
  --batch-sizes 256 512 1024 2048
```

本番モデルと02データ・N分布・ローカル重み・入力格子などが必要です。
保存された絶対パスがorcaで有効であることも必要です。欠けている場合は失敗を記録し、代替モデルで成功扱いしません。

## 出力と解釈

毎回新規の `results/benchmarks/tabpfn/YYYYMMDDTHHMMSSZ_UUID/` に保存します。
日時はUTCで、ディレクトリ名末尾ZがUTCを示します。

- `results.csv`: 基準＋全条件の測定値。各条件終了時に更新。
- `benchmark_config.json`, `benchmark_environment.json`: 実行条件、Git SHA、Python、PBS、CPU、コードハッシュ。
- `sample_manifest.json`: 元モデル・重み・前処理・固定入力のハッシュ、特徴量順、深度、対象セル数、環境。
- `sample_coordinates.csv`, `sample_features.csv`, `input.npy`: 同一比較入力。
- `jobs.json`: 実行順序と反復番号。
- `reference_predictions.npy`, `trial_*_predictions.npy`: 生の予測値。
- `*_result.json`, `*.log`, `benchmark_status.json`: 条件別結果・例外・完了状態。
- `dem_cache.npy`: ベンチマーク専用のDEMキャッシュ。元DEMサイズ分のディスク容量が必要。

CSVには要求項目に加え、反復番号、ウォームアップ時間、キャッシュ用再前処理時間、
平均絶対差・差のRMSE・Bias・非負補正後の最大差・一致判定・2,000万点の時間換算を保存します。
`status=prediction_mismatch` は処理できたが許容差を超えた条件であり、成功扱いしません。
`failed` は例外、タイムアウト、OSによる強制終了など。基準取得に失敗したら以後の比較を中止します。
失敗条件を除外して勝者を自動採用したり、本番設定へ反映したりはしません。

`model_load_seconds` は既存load_assetsによる復元全体（ハッシュ検証・前処理・N分布・重みの確認を含む）です。
`cache_build_seconds` は別モデルの生成＋fitのみ、全行のtransformは`rebuild_preprocessing_seconds`で分離します。
`peak_memory_mb` は各独立workerのOS `ru_maxrss` をMiBへ換算した値です。
モデル復元・ライブラリ・キャッシュ構築・ウォームアップ・推論全体のピークで、
推論単独のメモリや純粋なKVキャッシュサイズではありません。
`cache_build_peak_memory_mb` も構築までの累積ピークです。基準条件との差は参考であり、純キャッシュ容量とは断定できません。

`estimated_total_seconds_20m = model_load_seconds + cache_build_seconds
+ rebuild_preprocessing_seconds + total_prediction_rows / points_per_second`。
これは**推論主体の時間概算**です。全域の地理情報付与・ファイル保存・描画・キュー待ち・ウォームアップは含みません。
小規模サンプル、入力分布、CPU競合、温度、OSファイルキャッシュによっても変わります。
条件ごとの反復の中央値とばらつき、メモリ、予測差を確認してください。

## ローカルで許可された軽量検証

本番データやTabPFNを使わない偽モデル・小配列のテストだけです。

```bash
python -m unittest discover -s benchmark -p 'test_benchmark.py' -v
python -c "import ast,pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('benchmark').glob('*.py')]; print('syntax OK')"
bash -n benchmark/benchmark_tabpfn.sh
```

ローカルに03モデルがなくても実行できます。ベンチマーク本体はローカルでは実行しません。

## GitHub → orca（すべてユーザーが手動で実施）

ローカル:

```bash
git status --short
git remote -v                         # 意図したリポジトリか必ず確認
git add benchmark/benchmark_tabpfn.py benchmark/benchmark_tabpfn.sh \
  benchmark/config.json benchmark/test_benchmark.py benchmark/README.md
git diff --cached --stat              # モデル・データ・結果が含まれないことを確認
git commit -m "Add isolated TabPFN CPU inference benchmarks"
git push origin HEAD
```

調査時のローカルoriginは `Matsuda7/XGB_gamma_setting.git` で、依頼文の `XGB_soil_setting` と異なります。
意図した保存先を確認してからpushしてください。本実装はremoteを変更しません。
出力は既存の `results/**` ignore内に保存し、コードだけを明示的にaddします。

orca:

```bash
ssh matsuda@orca
cd /home/matsuda/XGB_soil_setting_orca
git status --short
git pull --ff-only                    # 本番で使用中のコードの更新タイミングにも注意
source .venv-tabpfn/bin/activate
python -m pip show tabpfn torch        # 更新・インストールはしない
```

`benchmark/config.json` のrunと設定を確認し、PBSスクリプトのppn・walltimeを編集します。
初回は `nodes=1:ppn=16`。128コア版は同じスクリプトを `ppn=128` へ変更し、
`cpu_threads=null` のまま別ジョブとして実行します。128コアが速いとは限りません。
8条件×3反復＋基準＋サンプル準備のため、walltimeは単一条件より長く必要です。
必要ならキュー・メモリ要求をorca運用に合わせて追加してください。CPU要求だけでは本番との同居を防げません。
本番same03と異なるノードへの割当を管理者の指定方法で確保してください。

```bash
qsub benchmark/benchmark_tabpfn.sh
qstat -u matsuda
tail -f logs/benchmark/<JOB_ID>.log
```

条件別の進捗と結果:

```bash
ls -td results/benchmarks/tabpfn/* | head
cat results/benchmarks/tabpfn/<RUN_ID>/benchmark_status.json
cat results/benchmarks/tabpfn/<RUN_ID>/results.csv
```

本番の `prediction_progress.json` / `run_summary.json` へは一切書き込みません。
