from app.policies.simple_memory_policy import SimpleMemoryPolicy


def test_should_remember_returns_true_for_allowed_key():
    policy = SimpleMemoryPolicy()

    result = policy.should_remember(
        "user_name",
        "Frank",
    )

    assert result is True


def test_should_remember_returns_false_for_unknown_key():
    policy = SimpleMemoryPolicy()

    result = policy.should_remember(
        "current_emotion",
        "Tired",
    )

    assert result is False
