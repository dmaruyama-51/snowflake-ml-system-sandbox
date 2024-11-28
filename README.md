# snowflake_machine_learning_system

## Overview

Snowflakeを活用した機械学習システムの実装例


## Setup
以下のコマンドで必要なパッケージをインストール
```bash
poetry install
```

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


## Dataset Information
UCI Machine Learning Repositoryの[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)を使用

以下のコマンドでSnowflakeにデータをアップロード

```bash
poetry run python src/adhoc/prepare_dataset.py
```
