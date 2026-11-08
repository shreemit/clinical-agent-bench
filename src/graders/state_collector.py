from __future__ import annotations

from agent.fhir_rules import extend_search, resource_url_from_location
from harness.http_client import HttpClient

_SEARCH_TYPES = ("Observation", "MedicationRequest", "ServiceRequest")
_HEADERS = {"Accept": "application/fhir+json", "Cache-Control": "no-cache"}
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
            _search_all(
                http,
                f"{base}{resource_type}?patient={patient_id}&_count=200&_sort=-_lastUpdated",
            )
        )
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": resource} for resource in _dedupe(resources)],
    }


def _read_patient(http: HttpClient, url: str) -> dict | None:
    result = http.request("GET", url, headers=_HEADERS)
    if result.status == 404:
        return None
    if result.status < 200 or result.status >= 300:
        raise RuntimeError(f"GET {url} returned status {result.status}")
    body = result.body
    if isinstance(body, dict) and body.get("resourceType") == "Patient":
        return body
    return None


def _search_all(http: HttpClient, url: str) -> list[dict]:
    state: dict = {"resources": []}
    next_url: str | None = url
    pages = 0
    while next_url:
        pages += 1
        if pages > _MAX_PAGES:
            raise RuntimeError(f"search exceeded {_MAX_PAGES} pages: {url}")
        result = http.request("GET", next_url, headers=_HEADERS)
        if result.status < 200 or result.status >= 300:
            raise RuntimeError(f"GET {next_url} returned status {result.status}")
        body = result.body if isinstance(result.body, dict) else {}
        next_url = extend_search(state, body)
    return list(state["resources"])


def read_locations(http: HttpClient, fhir_api_base: str, locations: list[str]) -> list[dict]:
    """Read the current resource for each write Location, ignoring history."""

    found: list[dict] = []
    for location in locations:
        if not location:
            continue
        url, _resource_id = resource_url_from_location(location, fhir_api_base)
        result = http.request("GET", url, headers=_HEADERS)
        if result.status < 200 or result.status >= 300 or not isinstance(result.body, dict):
            raise RuntimeError(f"GET {url} returned status {result.status}")
        if result.body.get("resourceType") in (None, "Bundle", "OperationOutcome"):
            raise RuntimeError(f"GET {url} did not return a resource")
        found.append(result.body)
    return found


def merge_resources(bundle: dict, resources: list[dict]) -> dict:
    """Add or replace bundle resources by type and id. Later copies win."""

    current = [entry["resource"] for entry in bundle.get("entry") or [] if "resource" in entry]
    order: list[tuple[str, str]] = []
    by_key: dict[tuple[str, str], dict] = {}
    anon: list[dict] = []
    for resource in [*current, *resources]:
        resource_id = resource.get("id")
        if resource_id is None or resource_id == "":
            anon.append(resource)
            continue
        key = (str(resource.get("resourceType") or ""), str(resource_id))
        if key not in by_key:
            order.append(key)
        by_key[key] = resource
    merged = [by_key[key] for key in order]
    merged.extend(anon)
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": resource} for resource in merged],
    }


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
