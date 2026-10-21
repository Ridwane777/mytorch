from analyzer.cli import main


def test_unimplemented_cli_exits_with_84():
    assert main([]) == 84
