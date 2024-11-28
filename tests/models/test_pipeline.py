from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from src.models.pipeline import create_model_pipeline


def test_create_model_pipeline_structure():
    """パイプラインの基本構造をテスト"""
    pipeline = create_model_pipeline()

    # パイプラインの型チェック
    assert isinstance(pipeline, Pipeline)

    # ステップ数の確認
    assert len(pipeline.steps) == 2

    # ステップ名の確認
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["preprocessor", "classifier"]

    # 各コンポーネントの型チェック
    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)
    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)
