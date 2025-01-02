from unittest.mock import Mock, patch

import pytest
from bittensor.core.config import Config

from infinite_games.sandbox.validator.utils.config import get_config


class TestConfig:
    @pytest.fixture
    def mock_dependencies(self):
        """Fixture to mock all external dependencies"""
        with (
            patch("argparse.ArgumentParser") as mock_parser_class,
            patch(
                "infinite_games.sandbox.validator.utils.config.Subtensor.add_args"
            ) as mock_subtensor_add_args,
            patch(
                "infinite_games.sandbox.validator.utils.config.LoggingMachine.add_args"
            ) as mock_logging_add_args,
            patch(
                "infinite_games.sandbox.validator.utils.config.Wallet.add_args"
            ) as mock_wallet_add_args,
            patch("infinite_games.sandbox.validator.utils.config.Config", spec=True),
        ):
            # Setup mock parser
            mock_parser = mock_parser_class.return_value

            # Return all mocked objects
            yield {
                "parser": mock_parser,
                "subtensor_add_args": mock_subtensor_add_args,
                "logging_add_args": mock_logging_add_args,
                "wallet_add_args": mock_wallet_add_args,
            }

    def create_mock_args(self, netuid: int, network: str) -> Mock:
        """Helper function to create mock args with specified values"""
        mock_args = Mock()
        mock_args.__getattribute__ = lambda x: {"netuid": netuid, "subtensor.network": network}.get(
            x
        )
        return mock_args

    def test_argument_parser_setup(self, mock_dependencies):
        """Test that argument parser is set up correctly"""
        mock_parser = mock_dependencies["parser"]
        mock_subtensor_add_args = mock_dependencies["subtensor_add_args"]
        mock_logging_add_args = mock_dependencies["logging_add_args"]
        mock_wallet_add_args = mock_dependencies["wallet_add_args"]

        # Mock successful parsing
        mock_args = self.create_mock_args(netuid=6, network="finney")
        mock_parser.parse_args.return_value = mock_args

        get_config()

        # Verify parser setup
        mock_parser.add_argument.assert_called_once_with(
            "--netuid", type=int, help="Subnet netuid", choices=[6, 155], required=False
        )

        # Verify all add_args methods were called
        mock_subtensor_add_args.assert_called_once_with(parser=mock_parser)
        mock_logging_add_args.assert_called_once_with(parser=mock_parser)
        mock_wallet_add_args.assert_called_once_with(parser=mock_parser)

    def test_valid_finney_config(self, mock_dependencies):
        """Test valid configuration with netuid 6 and finney network"""
        mock_parser = mock_dependencies["parser"]
        mock_args = self.create_mock_args(netuid=6, network="finney")
        mock_parser.parse_args.return_value = mock_args

        config = get_config()

        assert isinstance(config, Config)

    def test_valid_test_config(self, mock_dependencies):
        """Test valid configuration with netuid 155 and test network"""
        mock_parser = mock_dependencies["parser"]
        mock_args = self.create_mock_args(netuid=155, network="test")
        mock_parser.parse_args.return_value = mock_args

        config = get_config()

        assert isinstance(config, Config)

    def test_invalid_netuid_network_combination(self, mock_dependencies):
        """Test invalid combinations of netuid and network"""
        invalid_combinations = [
            (6, "test"),
            (155, "finney"),
            (1, "finney"),
            (200, "test"),
        ]

        for netuid, network in invalid_combinations:
            mock_parser = mock_dependencies["parser"]
            mock_args = self.create_mock_args(netuid=netuid, network=network)
            mock_parser.parse_args.return_value = mock_args

            with pytest.raises(
                ValueError, match=f"Invalid netuid {netuid} and network {network} combination."
            ):
                get_config()
