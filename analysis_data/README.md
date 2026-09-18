# analysis_data：分析に使うデータ（最終決定）

ハッカソンのプロトタイプ（動画から、①走り方のタイプとそのタイプのケガの多さ、②部位ごとの負担を出す）の分析は、**このフォルダのデータだけ**で行う。

- 各フォルダの中身は、公開元から取得したファイルをそのままコピーしたもの（2026-09-17）。加工はしていない
- 取得時の置き場所は `BioHackathon2026/data/` で、ここにある内容と一致することを確認済み
- `SHA256SUMS` にデータファイル（この README を除く）のハッシュ値がある。確認するときは、このフォルダで `sha256sum -c SHA256SUMS`
- 分析のスクリプトは `BioHackathon2026/analysis/` にあり、入力はすべてこのフォルダを参照する

## データの一覧

| フォルダ | データ | 規模 | 分析での役割 | 公開元・ライセンス |
|---|---|---|---|---|
| `npj2026_wu/` | Wu ほか 2026, *Multidisciplinary prediction of running-related injuries using machine learning*, npj Digital Medicine | 持久系ランナー142名、12か月の前向き追跡、週ごと6,181行 | ①リズムのタイプ（参加時の床反力のタイミング）と、その後のケガの多さ。発表の裏付け（動画だけの予測と、ケガ歴・練習量を足した予測の比較） | 論文の補足資料（Europe PMC, PMC12987969）。CC BY 4.0。https://doi.org/10.1038/s41746-026-02413-y |
| `loh2025_prospective/` | Loh ほか 2025, *Predicting running-related injuries from functional, kinetic and kinematic data*, Int J Sports Med | ランナー81名（12か月でケガ26名）、前向き | ①フォームのタイプ（参加時の2D動画の角度）と、その後のケガの割合 | NIE Data Repository（シンガポール）。CC BY-NC。https://doi.org/10.1055/a-2706-5516 |
| `lohkong2026_cross_sectional/` | Loh & Kong 2026, *Discriminant Efficacy of Video-Based Gait Analysis in Running-Related Injuries*, J Med Biol Eng | 受傷者154名（202脚、ケガの種類つき）、健常44名、横断 | 発表の裏付け（②の負担の推定が、ケガの種類と矛盾しないか）。角度の定義と分布の参照 | NIE Data Repository（シンガポール）。CC BY-NC |
| `fukuchi2017_running/` | Fukuchi ほか 2017, *A public data set of running biomechanics and the effects of running speed on lower extremity kinematics and kinetics*, PeerJ | ランナー39名、トレッドミル 2.5/3.5/4.5 m/s、3D動作と床反力 | ②動きから関節にかかる力を推定するモデルの学習。npj 2026 のタイミングを元の単位に戻すときの基準 | figshare 4543435（第5版）。CC BY 4.0。https://doi.org/10.6084/m9.figshare.4543435 |

## 各フォルダのファイル

- `npj2026_wu/europepmc_PMC12987969_supplementaryFiles.zip`：MOESM2（40列）、MOESM3（258列）、MOESM1（補足資料のPDF：質問票、変数の定義、表1）
- `loh2025_prospective/Loh 2025_Running prospective study data.xlsx`：シート Data_left, Data_right, Readme
- `lohkong2026_cross_sectional/Loh and Kong_2026_data_discriminant efficacy of video-based gait analysis.xlsx`：健常群、受傷者の一覧、受傷した脚の角度（左右）、ケガの種類の集計
- `fukuchi2017_running/`：`RBDSinfo.txt`（参加者情報）、`RBDSxxxprocessed.txt`（平均波形）、`RBDSxxxrunT{25,35,45}forces.txt`（床反力 300 Hz）、`RBDSxxxrunT{25,35,45}markers.txt`・`*.c3d`（マーカーの生データ）、`manifest.txt`（ダウンロードしたファイルの一覧とサイズ）

## 使わないことにしたデータ

`BioHackathon2026/data/` と `before_injury_data/` にある他のデータ（Running Injury Clinic、183名の動作スクリーニング、SoccerMon など）は、この分析では使わない。
データの注意点と、著者に確認せずに置いた前提は `analysis/step1_data/README.md` にまとめている。
