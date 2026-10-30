from datetime import datetime

from agent.fhir_rules import (
    calculate_age,
    magnesium_dose,
    potassium_dose_meq,
    resource_url_from_location,
    split_name,
)


def test_age_rounds_down_before_birthday():
    assert calculate_age(datetime(2000, 11, 14)) == 22
    assert calculate_age(datetime(2000, 11, 13)) == 23
    assert calculate_age(datetime(1932, 12, 29)) == 90


def test_magnesium_bands():
    assert magnesium_dose(2.0) is None
    assert magnesium_dose(1.9) == (1, 1)
    assert magnesium_dose(1.5) == (1, 1)
    assert magnesium_dose(1.4) == (2, 2)
    assert magnesium_dose(1.0) == (2, 2)
    assert magnesium_dose(0.9) == (4, 4)


def test_potassium_dose():
    assert potassium_dose_meq(3.5) is None
    assert potassium_dose_meq(3.4) == 10
    assert potassium_dose_meq(3.0) == 50


def test_location_strips_history():
    url, resource_id = resource_url_from_location(
        "http://localhost:8080/fhir/Observation/786002/_history/1",
        "http://localhost:8080/fhir/",
    )
    assert url == "http://localhost:8080/fhir/Observation/786002"
    assert resource_id == "786002"


def test_split_name_last_token_is_family():
    assert split_name("Peter Stafford") == ("Peter", "Stafford")
