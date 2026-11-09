from qotair.tui.app import headless_contract


def test_headless_contract_exposes_workflow_screens():
    contract = headless_contract()
    assert contract["mode"] == "headless"
    assert "Home" in contract["screens"]
    assert "Plan Preview" in contract["screens"]
    assert "Run Monitor" in contract["screens"]
    assert "History" in contract["screens"]
    assert "plan" in contract["actions"]
