class TestValidator:
    def test_validator(self):
        file_path = "neurons/validator.py"

        with open(file_path, "r") as file:
            lines = file.readlines()

        # Assert the expected content
        expected_start = [
            "# -- DO NOT TOUCH BELOW - ENV SET --\n",
            "# flake8: noqa: E402\n",
            "import os\n",
            "import sys\n",
            "\n",
            "# Force torch - must be set before importing bittensor\n",
            'os.environ["USE_TORCH"] = "1"\n',
            "\n",
            "# Add the parent directory of the script to PYTHONPATH\n",
            "script_dir = os.path.dirname(os.path.abspath(__file__))\n",
            "parent_dir = os.path.dirname(script_dir)\n",
            "sys.path.append(parent_dir)\n",
            "# -- DO NOT TOUCH ABOVE --\n",
        ]

        # Check the expected lines
        for i, expected_line in enumerate(expected_start):
            assert lines[i] == expected_line
