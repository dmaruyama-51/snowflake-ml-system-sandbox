import numpy as np
import pandas as pd
import pytest
from snowflake.snowpark import Session

from src.pipelines.sproc_offline_testing import sproc_offline_testing


@pytest.mark.parametrize("challenger_better", [True, False])
def test_sproc_offline_testing_success(mocker, challenger_better):
    """
    sproc_offline_testingの正常系テスト

    Args:
        mocker: pytest-mockのフィクスチャ
        challenger_better: チャレンジャーモデルが優れているかどうかのフラグ
    """
    # モックセッションの作成
    mock_session = mocker.Mock(spec=Session)

    # モデルバージョンのモック
    champion_model = mocker.Mock()
    champion_model.version = "V_250130_121116"

    challenger_model = mocker.Mock()
    challenger_model.version = "V_250202_121116"

    # テストデータの準備
    test_data = pd.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "REVENUE": np.random.choice([0, 1], size=100),
            "UID": range(100),
        }
    )

    # 各依存関数のモック化
    mock_load_default = mocker.patch(
        "src.pipelines.sproc_offline_testing.load_default_model_version"
    )
    mock_load_default.return_value = champion_model

    mock_load_latest = mocker.patch(
        "src.pipelines.sproc_offline_testing.load_latest_model_version"
    )
    mock_load_latest.return_value = challenger_model

    mock_fetch_test = mocker.patch(
        "src.pipelines.sproc_offline_testing.fetch_test_dataset"
    )
    mock_fetch_test.return_value = test_data

    mock_predict_proba = mocker.patch(
        "src.pipelines.sproc_offline_testing.predict_proba"
    )
    mock_predict_proba.return_value = np.random.rand(100)

    mock_predict_label = mocker.patch(
        "src.pipelines.sproc_offline_testing.predict_label"
    )
    mock_predict_label.return_value = np.random.randint(0, 2, 100)

    mock_calc_metrics = mocker.patch(
        "src.pipelines.sproc_offline_testing.calc_evaluation_metrics"
    )
    if challenger_better:
        challenger_scores = {"PR-AUC": 0.9}
        champion_scores = {"PR-AUC": 0.8}
    else:
        challenger_scores = {"PR-AUC": 0.8}
        champion_scores = {"PR-AUC": 0.9}
    mock_calc_metrics.side_effect = [challenger_scores, champion_scores]

    # Registryのモック
    mock_registry = mocker.patch("src.pipelines.sproc_offline_testing.Registry")
    mock_registry_instance = mocker.Mock()
    mock_registry.return_value = mock_registry_instance
    mock_model = mocker.Mock()
    mock_registry_instance.get_model.return_value = mock_model

    # テスト実行
    result = sproc_offline_testing(mock_session)

    # アサーション
    assert result == 1
    mock_load_default.assert_called_once_with(mock_session)
    mock_load_latest.assert_called_once_with(mock_session)
    mock_fetch_test.assert_called_once_with(mock_session, challenger_model)

    if challenger_better:
        # チャレンジャーモデルが優れている場合、デフォルトバージョンが更新されることを確認
        mock_registry_instance.get_model.assert_called_once_with("random_forest")
        assert mock_model.default == challenger_model
    else:
        # チャレンジャーモデルが劣っている場合、デフォルトバージョンが更新されないことを確認
        mock_registry_instance.get_model.assert_not_called()


def test_sproc_offline_testing_error(mocker):
    """
    エラー発生時のテスト
    """
    # モックセッションの作成
    mock_session = mocker.Mock(spec=Session)

    # load_default_model_versionでエラーを発生させる
    mock_load_default = mocker.patch(
        "src.pipelines.sproc_offline_testing.load_default_model_version"
    )
    mock_load_default.side_effect = Exception("テストエラー")

    # エラーが発生することを確認
    with pytest.raises(Exception, match="テストエラー"):
        sproc_offline_testing(mock_session)
