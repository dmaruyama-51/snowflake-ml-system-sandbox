# Snowflake ML System Sandbox

## Overview

Snowflakeを活用した機械学習システムの実装例 & 新機能調査のsandbox

### Usecase

- オンラインショッピングにおける顧客の購買意図を予測・スコア算出し、スコアを元にマーケティング施策を展開する
- このシステムでは Daily のバッチ処理で、ユーザーごとに購入意欲スコアをSnowflakeのScoresテーブルに保存する処理までを範囲とする。（本来はその先にMA等への連携などを想定）


### Dataset Information
UCI Machine Learning Repositoryの[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)を加工して使用。元のデータセットに SessionDate と UID を追加している。

| カラム名                   | 説明                                                                                   |
|-----------------------------|----------------------------------------------------------------------------------------|
| Administrative              | 訪問者が閲覧した管理関連ページの数                                                      |
| Administrative_Duration     | 管理関連ページに費やした総時間（秒）                                                    |
| Informational               | 訪問者が閲覧した情報提供ページの数                                                      |
| Informational_Duration      | 情報提供ページに費やした総時間（秒）                                                    |
| ProductRelated              | 訪問者が閲覧した商品関連ページの数                                                      |
| ProductRelated_Duration     | 商品関連ページに費やした総時間（秒）                                                    |
| BounceRates                 | 訪問者が最初のページから他のページに移動せずに離脱した割合                              |
| ExitRates                   | 特定のページがセッション中の最後のページとなった割合                                    |
| PageValues                  | eコマース取引を完了する前に訪問されたページの平均価値                                   |
| SpecialDay                  | 訪問日が特定の特別な日にどれだけ近いかを示す指標（例：母の日、バレンタインデー）         |
| Month                       | 訪問月（例：Jan, Feb, Mar）                                                             |
| OperatingSystems            | 訪問者が使用したオペレーティングシステムの種類                                          |
| Browser                     | 訪問者が使用したブラウザの種類                                                          |
| Region                      | 訪問者の地理的な地域                                                                    |
| TrafficType                 | 訪問者がウェブサイトにアクセスしたトラフィックの種類                                     |
| VisitorType                 | 訪問者が新規かリピーターかを示す（'New_Visitor', 'Returning_Visitor', 'Other'）         |
| Weekend                     | 訪問が週末に行われたかどうかを示す（True または False）                                 |
| Revenue                     | セッションが購入に至ったかどうかを示す（True または False）                             |
| SessionDate                     | 訪問日（Monthを元に 2024-xx-01 の日付として生成）                |
| UID                     | 訪問者ID（UUIDで生成）                |

※ UID と SessionDate でユニーク



## Setup

### 環境設定
以下のコマンドで必要なパッケージをインストール
```bash
poetry install
```

### Snowflakeの接続情報

Snowflakeの接続情報は`connection_parameters.json`に記載してルートディレクトリに配置
```json
{
    "account": "",
    "user": "",
    "password": "",
    "role": "",
    "warehouse": "",
    "database": "",
    "schema": ""
}
```

### データセットの用意
以下のコマンドでデータを生成してSnowflakeへアップロード

```bash
poetry run python src/adhoc/prepare_dataset.py
```