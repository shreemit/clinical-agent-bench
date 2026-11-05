from __future__ import annotations

from agent.fhir_rules import bundle_resources, next_page_url
from harness.http_client import HttpClient

_SEARCH_TYPES = ("Observation", "MedicationRequest", "ServiceRequest")
_ACCEPT = {"Accept": "application/fhir+json"}
_MAX_PAGES = 200


def collect_patient_state(http: HttpClient, fhir_api_base: str, patient_id: str) -> dict:
    """Return a collection Bundle of one patient's current chart."""

    base = fhir_api_base.rstrip("/") + "/"
    resources: list[dict] = []
    patient = _read_patient(http, f"{base}Patient/{patient_id}")
    if patient is not None:
        resources.append(patient)
    for resource_type in _SEARCH_TYPES:
        resources.extend(
            _search_all(http, f"{base}{resource_type}?patient={patient_id}&_count=200")
        )
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": resource} for resource in _dedupe(resources)],
    }


def _read_patient(http: HttpClient, url: str) -> dict | None:
    result = http.request("GET", url, headers=_ACCEPT)
    if result.status == 404:
        return None
    if result.status < 200 or result.status >= 300:
        raise RuntimeError(f"GET {url} returned status {result.status}")
    body = result.body
    if isinstance(body, dict) and body.get("resourceType") == "Patient":
        return body
    return None


def _search_all(http: HttpClient, url: str) -> list[dict]:
    found: list[dict] = []
    next_url: str | None = url
    pages = 0
    while next_url:
        pages += 1
        if pages > _MAX_PAGES:
            raise RuntimeError(f"search exceeded {_MAX_PAGES} pages: {url}")
        result = http.request("GET", next_url, headers=_ACCEPT)
        if result.status < 200 or result.status >= 300:
            raise RuntimeError(f"GET {next_url} returned status {result.status}")
        body = result.body if isinstance(result.body, dict) else {}
        found.extend(bundle_resources(body))
        next_url = next_page_url(body)
    return found


def _dedupe(resources: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for resource in resources:
        resource_id = resource.get("id")
        if resource_id is None or resource_id == "":
            unique.append(resource)
            continue
        key = (str(resource.get("resourceType") or ""), str(resource_id))
        if key in seen:
            continue
        seen.add(key)
        unique.append(resource)
    return unique
