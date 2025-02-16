from unittest.mock import Mock, patch

import pytest
from snowflake.ml.registry import Registry

from src.models.rollback import rollback_model


@pytest.fixture
def mock_session():
    return Mock()


@pytest.fixture
def mock_registry():
    registry = Mock(spec=Registry)
    model_ref = Mock()

    # モデルの現在のバージョンをモック
    current_version = Mock()
    current_version.version = "v_current"
    model_ref.default = current_version

    # バージョンメソッドをモック
    target_version = Mock()
    model_ref.version.return_value = target_version

    registry.get_model.return_value = model_ref
    return registry


def test_rollback_model_success(mock_session, mock_registry):
    """正常系: モデルのロールバックが成功するケース"""
    with patch("src.models.rollback.Registry", return_value=mock_registry):
        rollback_model(mock_session, "v_target")

        # Registryが正しく初期化されたことを確認
        mock_registry.get_model.assert_called_once_with("random_forest")

        # 指定したバージョンが取得されたことを確認
        model_ref = mock_registry.get_model.return_value
        model_ref.version.assert_called_once_with("v_target")


def test_rollback_model_version_not_found(mock_session, mock_registry):
    """異常系: 指定したバージョンが存在しないケース"""
    mock_registry.get_model.return_value.version.side_effect = Exception(
        "Version not found"
    )

    with patch("src.models.rollback.Registry", return_value=mock_registry):
        with pytest.raises(ValueError, match="Specified version v_target not found"):
            rollback_model(mock_session, "v_target")


def test_rollback_model_registry_error(mock_session):
    """異常系: Registryの初期化に失敗するケース"""
    with patch("src.models.rollback.Registry", side_effect=Exception("Registry error")):
        with pytest.raises(Exception, match="Registry error"):
            rollback_model(mock_session, "v_target")
