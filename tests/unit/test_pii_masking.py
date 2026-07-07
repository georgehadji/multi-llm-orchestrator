import pytest
from orchestrator.safety.pii_masking_etl import PIIMaskingETL

pytestmark = pytest.mark.unit


def test_pii_masking_etl_email_and_ip():
    etl = PIIMaskingETL()

    raw_payload = (
        "The user john.doe@example.com connected from 192.168.1.1. "
        "Contact them at admin@test.org or 10.0.0.5."
    )

    sanitized = etl.transform(raw_payload)

    assert "john.doe@example.com" not in sanitized
    assert "admin@test.org" not in sanitized
    assert "192.168.1.1" not in sanitized
    assert "10.0.0.5" not in sanitized

    assert "<EMAIL_MASKED>" in sanitized
    assert "<IPV4_MASKED>" in sanitized
    assert sanitized.count("<EMAIL_MASKED>") == 2
    assert sanitized.count("<IPV4_MASKED>") == 2


def test_pii_masking_etl_api_keys():
    etl = PIIMaskingETL()

    raw_payload = "Here is my secret key: sk-1234567890abcdef1234567890abcdef12345678"
    sanitized = etl.transform(raw_payload)

    assert "sk-1234567890abcdef1234567890abcdef12345678" not in sanitized
    assert "<API_KEY_MASKED>" in sanitized


def test_pii_masking_etl_phone_numbers():
    etl = PIIMaskingETL()

    raw_payload = "Call me at +1-555-555-5555 or 800-123-4567."
    sanitized = etl.transform(raw_payload)

    assert "+1-555-555-5555" not in sanitized
    assert "800-123-4567" not in sanitized
    assert "<PHONE_MASKED>" in sanitized
    assert sanitized.count("<PHONE_MASKED>") == 2


def test_pii_masking_process_dict():
    etl = PIIMaskingETL()

    data = {
        "user": "test@test.com",
        "ip": "1.1.1.1",
        "nested": "not string",  # Assuming shallow dict for this method
    }

    sanitized = etl.process_dict(data)
    assert sanitized["user"] == "<EMAIL_MASKED>"
    assert sanitized["ip"] == "<IPV4_MASKED>"
    assert sanitized["nested"] == "not string"
