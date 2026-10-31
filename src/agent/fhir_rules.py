from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

FROZEN_NOW = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
AUTHORED_ON = "2023-11-13T10:15:00+00:00"
NEXT_MORNING = "2023-11-14T08:00:00+00:00"
ONE_YEAR_BEFORE = datetime.fromisoformat("2022-11-13T10:15:00+00:00")

_NAME_DOB = re.compile(r"name (.+?) and DOB of (\d{4}-\d{2}-\d{2})")
_MRN = re.compile(r"\b(S\d+)\b")
_QUOTED = re.compile(r'"([^"]*)"')
_NDC = re.compile(r"\b\d{4,5}-\d{3,4}-\d{2}\b")
_LOINC = re.compile(r"\b\d{3,5}-\d\b")
_SNOMED = re.compile(r"\b\d{6,18}\b")
_BP_CODE = re.compile(r"flowsheet ID for blood pressure is (\w+)")


def calculate_age(dob: datetime, today: datetime = FROZEN_NOW) -> int:
    age = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    return age


# TODO(decision): last token is the family name; vendored task1 names are two tokens.
def split_name(full_name: str) -> tuple[str, str]:

    parts = full_name.strip().split()
    if len(parts) < 2:
        raise ValueError(f"cannot split patient name {full_name!r}")
    return " ".join(parts[:-1]), parts[-1]


def parse_name_and_dob(instruction: str) -> tuple[str, str, str]:
    match = _NAME_DOB.search(instruction)
    if not match:
        raise ValueError("instruction has no name/DOB")
    given, family = split_name(match.group(1))
    return given, family, match.group(2)


def parse_mrn(instruction: str) -> str:
    match = _MRN.search(instruction)
    if not match:
        raise ValueError("instruction has no MRN")
    return match.group(1)


def first_quoted(text: str) -> str:
    match = _QUOTED.search(text)
    if not match:
        raise ValueError("no quoted string")
    return match.group(1)


def require_ndc(context: str) -> str:
    found = _NDC.findall(context)
    if not found:
        raise ValueError("context has no NDC")
    return found[0]


def require_loinc(context: str) -> str:
    found = _LOINC.findall(context)
    if not found:
        raise ValueError("context has no LOINC-like code")
    return found[0]


def require_snomed(context: str) -> str:
    found = _SNOMED.findall(context)
    if not found:
        raise ValueError("context has no SNOMED code")
    return found[0]


def require_bp_code(context: str) -> str:
    match = _BP_CODE.search(context)
    if not match:
        raise ValueError("context has no blood pressure flowsheet id")
    return match.group(1)


class IncompleteSearchError(Exception):
    """The server reported more matches than the pages we were able to read."""


def bundle_resources(body: Any) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return []
    return [entry["resource"] for entry in body.get("entry") or [] if "resource" in entry]


def next_page_url(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    for link in body.get("link") or []:
        if isinstance(link, dict) and link.get("relation") == "next" and link.get("url"):
            return str(link["url"])
    return None


def extend_search(state: dict[str, Any], body: Any) -> str | None:
    """Append one search page. Return the next URL, or None when the search is finished.

    `state["total"]` is remembered when the server sends it. A short page is an
    incomplete search only when that total is larger than the entries consumed
    and there is no next link. A caller that asked for `_count=1` should not
    use this; that page is the newest row, not the whole chart.
    """

    state.setdefault("resources", []).extend(bundle_resources(body))
    total = body.get("total") if isinstance(body, dict) else None
    if isinstance(total, int) and not isinstance(total, bool):
        state["total"] = total
    nxt = next_page_url(body)
    if nxt:
        return nxt
    reported = state.get("total")
    got = len(state["resources"])
    if isinstance(reported, int) and reported > got:
        raise IncompleteSearchError(
            f"search total {reported} exceeds {got} entries and there is no next page"
        )
    return None


def latest_quantity(
    resources: list[dict[str, Any]],
    within_hours: float | None = None,
    now: datetime = FROZEN_NOW,
) -> tuple[datetime | None, float | None, str | None]:
    """Most recent valueQuantity, optionally limited to the last `within_hours`."""

    last_time: datetime | None = None
    last_value: float | None = None
    last_raw: str | None = None
    cutoff = None if within_hours is None else now - timedelta(hours=within_hours)
    for resource in resources:
        raw = resource.get("effectiveDateTime")
        if not raw or "valueQuantity" not in resource:
            continue
        effective = datetime.fromisoformat(raw)
        if cutoff is not None and effective < cutoff:
            continue
        if last_time is None or effective > last_time:
            last_time = effective
            last_value = resource["valueQuantity"]["value"]
            last_raw = raw
    return last_time, last_value, last_raw


# TODO(decision): dose bands are hardcoded; every task5 and task9 row shares one context.
def magnesium_dose(last_value: float) -> tuple[int, int] | None:
    """IV magnesium (grams, hours), or None when the level is above 1.9."""

    if last_value > 1.9:
        return None
    if last_value < 1:
        return 4, 4
    if last_value < 1.5:
        return 2, 2
    return 1, 1


def potassium_dose_meq(last_value: float) -> float | None:
    """10 mEq for every 0.1 below 3.5. None when the level is not low."""

    if last_value >= 3.5:
        return None
    return round((3.5 - last_value) / 0.1 * 10, 1)


def resource_url_from_location(location: str, fhir_api_base: str) -> tuple[str, str]:
    """Strip `_history` and return (url, resource id)."""

    base = fhir_api_base.rstrip("/") + "/"
    url = location if location.startswith("http") else base + location.lstrip("/")
    url = re.sub(r"/_history/\d+$", "", url)
    resource_id = url.rstrip("/").rsplit("/", 1)[-1]
    return url, resource_id
