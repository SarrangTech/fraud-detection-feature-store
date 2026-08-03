from serving.feature_mapping import FEATURE_COLS, build_feature_vector


def test_build_feature_vector_returns_all_expected_columns_in_order():
    txn = {"transaction_id": "txn_1", "user_id": "user_00001", "amount": 100.0}
    vector = build_feature_vector(txn, user_proxy_features=None)

    assert list(vector.keys()) == FEATURE_COLS
    assert all(isinstance(v, float) for v in vector.values())


def test_cold_start_user_falls_back_to_transaction_amount():
    txn = {"transaction_id": "txn_1", "user_id": "user_new", "amount": 250.0}
    vector = build_feature_vector(txn, user_proxy_features=None)

    # With no history, "recent average" collapses to this transaction's own amount,
    # so the ratio features start neutral (~1.0) rather than spuriously extreme.
    assert vector["avg_amount_24h"] == 250.0
    assert round(vector["amount_vs_avg_24h"], 2) == round(250.0 / 250.01, 2)


def test_known_user_uses_redis_proxy_features():
    txn = {"transaction_id": "txn_2", "user_id": "user_00042", "amount": 50.0}
    proxy = {
        "avg_velocity_1h": 2.0,
        "avg_velocity_24h": 5.0,
        "avg_daily_spend": 400.0,
        "avg_amount_all_time": 80.0,
        "max_amount_all_time": 300.0,
    }
    vector = build_feature_vector(txn, proxy)

    assert vector["txn_count_1h"] == 2.0
    assert vector["txn_count_24h"] == 5.0
    assert vector["total_amount_24h"] == 400.0
    assert vector["avg_amount_24h"] == 80.0
