# ---------------------------------------------------------------------------
# Regression tests for the three verifier bugs found on the held-out reports
# ---------------------------------------------------------------------------
from triage.verifier.verify import check_field, check_baseline, ID_TOKENS, NUMBERISH

EMAIL = {"id": "E1", "source": "email", "to": "a@x.com;b@x.com;c@x.com"}


# Bug 1: to/cc/bcc only accepted a single address -------------------------------
def test_list_field_single_recipient():
    assert check_field(EMAIL, "to", "b@x.com") is None


def test_list_field_full_list_same_order():
    assert check_field(EMAIL, "to", "a@x.com;b@x.com;c@x.com") is None


def test_list_field_subset_any_order_and_spaces():
    assert check_field(EMAIL, "to", "c@x.com; a@x.com") is None


def test_list_field_case_insensitive():
    assert check_field(EMAIL, "to", "A@X.com") is None


def test_list_field_rejects_one_wrong_address():
    assert check_field(EMAIL, "to", "a@x.com;z@x.com") is not None   # would pass with any()


def test_list_field_rejects_partial_address():
    jim = {"id": "E2", "source": "email", "to": "jimbob@x.com;carol@x.com"}
    assert check_field(jim, "to", "bob@x.com") is not None           # would pass with substring
    assert check_field(jim, "to", "x.com") is not None


def test_list_field_rejects_empty_claim():
    assert check_field(EMAIL, "to", "") is not None
    assert check_field(EMAIL, "to", ";") is not None


# Bug 2: 114.0 from JSON was treated as one-decimal precision -------------------
def test_aggregate_whole_number_float_passes():
    assert check_baseline({"avg_http": 114.18}, "avg_http", 114.0) is None


def test_aggregate_plain_int_passes():
    assert check_baseline({"avg_http": 114.18}, "avg_http", 114) is None


def test_aggregate_decimal_precision_still_applies():
    assert check_baseline({"avg_http": 114.18}, "avg_http", 114.2) is None
    assert check_baseline({"avg_http": 114.18}, "avg_http", 114.1) is not None


def test_aggregate_wrong_whole_number_fails():
    assert check_baseline({"avg_http": 114.18}, "avg_http", 120.0) is not None


# Bug 3: user/PC ids triggered the "unchecked numbers" warning ------------------
def _warns(text):
    return bool(NUMBERISH.search(ID_TOKENS.sub("", text)))


def test_number_warning_ignores_user_id():
    assert not _warns("KLH0596 visited two websites outside their usual working hours.")


def test_number_warning_ignores_pc_name():
    assert not _warns("The user logged on to PC-8486 late at night.")


def test_number_warning_still_catches_real_numbers():
    assert _warns("KLH0596 visited 30 sites.")
    assert _warns("The session lasted 506 minutes.")
    assert _warns("The session started at 19:54.")