from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from src.models.pipeline import create_model_pipeline


def test_create_model_pipeline():
    """モデルパイプライン作成のテスト"""
    # パイプラインの作成
    pipeline = create_model_pipeline(random_state=42)

    # 戻り値の型チェック
    assert isinstance(pipeline, Pipeline), "戻り値がPipelineインスタンスではありません"

    # パイプラインのステップ数チェック
    assert len(pipeline.steps) == 2, "パイプラインのステップ数が想定と異なります"

    # 各ステップの名前と型のチェック
    preprocessor_step = pipeline.steps[0]
    classifier_step = pipeline.steps[1]

    # 前処理ステップのチェック
    assert preprocessor_step[0] == "preprocessor", "前処理ステップの名前が不正です"
    assert isinstance(
        preprocessor_step[1], ColumnTransformer
    ), "前処理ステップの型が不正です"

    # 分類器ステップのチェック
    assert classifier_step[0] == "classifier", "分類器ステップの名前が不正です"
    assert isinstance(
        classifier_step[1], RandomForestClassifier
    ), "分類器ステップの型が不正です"
