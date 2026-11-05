# c・γの説明変数（最新設定）

cの説明変数はconfig/c/features.jsoncが正本です。現在は以下の10変数が有効で、x・yはコメントアウトされています。γの選択はconfig/gamma/features.jsoncを参照してください。

| 列 | 意味 |
|---|---|
| sample_elevation | 学習：DEM地表標高−試料中央深度。予測：DEM地表標高−設定深度 |
| depth | 地表からの深度（m） |
| jshis_arv | 地盤増幅率 |
| slope | 傾斜 |
| curvature | 曲率 |
| jshis_avs30 | AVS30 |
| symbol | 地質区分 |
| soil_inventory_ssg | 土壌インベントリ区分 |
| jshis_jcode | 地形区分 |
| n_value | 同じ試料標高／予測標高に対応する推定N値（上限50） |

surface_zは標高計算用の補助列として保持します。cにはdepthとsample_elevationの両方を説明変数として入力します。
現在の予測面の指定はprediction_depth_mです。一定標高面を直接指定する設定ではありません。
cとγはそれぞれconfig/c/features.jsonc、config/gamma/features.jsoncで管理します。
cのn_valueは保存済みの最新N値分布から平面位置で取得します。設定はconfig/c/n_distribution.jsonです。
φと同じ実行ならφを先に実行し、そのN値分布を使います。固定深度のN分布なので、試料標高に合わせたN値再推定はしません。

γは従来どおり主要試料のみでpredicted_nを学習します。比較用no_n条件ではn_valueを外します。
cは有効粘着力が得られるCŪ・CD試料を使用し、実測N値との対応有無では学習対象を限定しません。
c専用の上流N値学習は行いません。保存元・N分布深度・対応距離を記録します。
γの上流N値推定は従来の処理を維持します。
寄与度・目的値度数分布・R²付き評価図の出力は継続します。

今回の変更後の解析・学習・テストは未実行です。既存結果は旧設定のままです。
前処理の設定署名を変更したため、次回はprepareを含む全段階を実行してください。
`python run_analysis.py --targets c gamma`


## N値条件の評価

cでは実測Nと推定Nの条件比較を廃止し、保存済みN分布を入力したcモデルを評価します。
c自身の分割は維持しますが、N分布を作ったモデルの学習データとの重複は除外していないため、
上流Nモデルまで含む独立評価ではありません。評価JSONにもこの条件を記録します。
γのN値条件比較は変更していません。

試料標高の説明変数名は、c・γ・φおよび上流N値モデルで `sample_elevation` に統一します。計算式は `surface_z - depth` です。旧列名で作成した学習データ・保存モデルを使う場合は、前処理と学習を再実行してください。
