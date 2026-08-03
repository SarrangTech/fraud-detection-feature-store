from streaming.producer import simulate_user_id, to_event


def test_simulate_user_id_is_deterministic():
    a = simulate_user_id(row_index=42, time_offset_seconds=1000.0, pool_size=5000)
    b = simulate_user_id(row_index=42, time_offset_seconds=1000.0, pool_size=5000)
    assert a == b


def test_simulate_user_id_spreads_across_pool():
    ids = {simulate_user_id(i, float(i), pool_size=100) for i in range(1000)}
    # With 1000 samples into a pool of 100, expect broad coverage, not everything
    # collapsing onto a handful of buckets.
    assert len(ids) > 50


def test_to_event_maps_kaggle_row_correctly():
    row = {"Time": "10.0", "Amount": "125.50", "Class": "1", **{f"V{i}": "0.1" for i in range(1, 29)}}
    event = to_event(row_index=7, row=row, pool_size=100)

    assert event["transaction_id"] == "txn_00000007"
    assert event["amount"] == 125.50
    assert event["is_fraud"] == 1
    assert event["time_offset_seconds"] == 10.0
    assert event["V1"] == 0.1
    assert event["V28"] == 0.1
    assert event["user_id"].startswith("user_")
