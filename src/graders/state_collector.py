from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import quote

from agent.fhir_rules import FROZEN_NOW, bundle_resources, extend_search, next_page_url, resource_url_from_location
from harness.http_client import HttpClient

_SEARCH_TYPES = ("Observation", "MedicationRequest", "ServiceRequest")
_HEADERS = {"Accept": "application/fhir+json", "Cache-Control": "no-cache"}
_MAX_PAGES = 200
_FULL_CHART_LIMIT = 400
_DELTA_PAGES = 5
_HISTORY_COUNT = 20
_MAX_DELETE_CHECKS = 50
_GOLD_LABS = (("MG", 24), ("K", None), ("A1C", None))


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
    return _as_bundle(_dedupe(resources))


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
    return _search_limited(http, url, _MAX_PAGES)


def _search_limited(http: HttpClient, url: str, max_pages: int) -> list[dict]:
    """Follow next links until the search ends or `max_pages` is exceeded."""

    state: dict = {"resources": []}
    next_url: str | None = url
    pages = 0
    while next_url:
        pages += 1
        if pages > max_pages:
            raise RuntimeError(f"search exceeded {max_pages} pages: {url}")
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
    return _as_bundle(merged)


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


@dataclass
class TypeChart:
    resource_type: str
    count: int | None
    complete: bool
    resources: list[dict]
    watermark: str | None = None


@dataclass
class ChartSnapshot:
    """Pre-trial chart. Large types keep a watermark instead of every resource."""

    bundle: dict
    patient_id: str
    types: tuple[TypeChart, ...] = field(default_factory=tuple)

    @property
    def large(self) -> bool:
        return any(not item.complete for item in self.types)


def snapshot_patient(http: HttpClient, fhir_api_base: str, patient_id: str) -> ChartSnapshot:
    """Read the chart, paging a type only when its count is small."""

    base = fhir_api_base.rstrip("/") + "/"
    patient = _read_patient(http, f"{base}Patient/{quote(str(patient_id), safe='')}")
    resources: list[dict] = []
    if patient is not None:
        resources.append(copy.deepcopy(patient))
    types: list[TypeChart] = []
    for resource_type in _SEARCH_TYPES:
        loaded, total, complete = _load_type(http, base, patient_id, resource_type)
        watermark = None
        if not complete and resource_type == "Observation":
            loaded.extend(_gold_labs(http, base, patient_id))
        if not complete:
            probe = _probe_newest(http, base, patient_id, resource_type)
            if probe is not None:
                loaded.append(probe)
                watermark = _last_updated(probe)
        loaded = [copy.deepcopy(item) for item in _dedupe(loaded)]
        types.append(
            TypeChart(
                resource_type=resource_type,
                count=total,
                complete=complete,
                resources=loaded,
                watermark=watermark,
            )
        )
        resources.extend(loaded)
    return ChartSnapshot(
        bundle=_as_bundle(_dedupe(resources)),
        patient_id=str(patient_id),
        types=tuple(types),
    )


def patient_bundles_for_grade(
    http: HttpClient,
    fhir_api_base: str,
    patient_id: str,
    snapshot: ChartSnapshot,
    locations: list[str] | None = None,
    created_keys: set[tuple[str, str]] | None = None,
    known_keys: set[tuple[str, str]] | None = None,
) -> tuple[dict, dict]:
    """Return pre and post bundles that show creates, updates, and deletes.

    Small types are read again in full. Large types keep the pre-trial resources
    and add rows with `_lastUpdated` after the watermark, plus harness Locations.
    """

    base = fhir_api_base.rstrip("/") + "/"
    created_keys = created_keys or set()
    known_keys = known_keys or set()
    pre_keys = {_resource_key(resource) for resource in bundle_resources(snapshot.bundle)}
    pre_keys.discard(None)
    post_resources: list[dict] = []
    patient = _read_patient(http, f"{base}Patient/{quote(str(patient_id), safe='')}")
    if patient is not None:
        post_resources.append(patient)
    for chart in snapshot.types:
        if chart.complete:
            reloaded, _total, _complete = _load_type(http, base, patient_id, chart.resource_type)
            post_resources.extend(reloaded)
            continue
        post_resources.extend(chart.resources)
        if chart.watermark:
            post_resources.extend(
                _changed_since(http, base, patient_id, chart.resource_type, chart.watermark)
            )
    written = read_locations(http, fhir_api_base, list(locations or []))
    post_resources.extend(written)
    pre_extra: list[dict] = []
    for resource in [*post_resources, *written]:
        key = _resource_key(resource)
        if key is None or key in pre_keys or key in created_keys:
            continue
        chart = _chart_for(snapshot, key[0])
        before = _before_image(http, base, key[0], key[1], None if chart is None else chart.watermark)
        if before is not None:
            pre_extra.append(before)
            pre_keys.add(key)
            continue
        if _version_id(resource) not in (None, "1", 1):
            pre_extra.append({"resourceType": key[0], "id": key[1]})
            pre_keys.add(key)
    post_bundle = merge_resources(_as_bundle(post_resources), written)
    post_bundle = _apply_deletes(
        http,
        base,
        snapshot,
        post_bundle,
        known_keys,
        pre_extra,
    )
    pre_bundle = merge_resources(snapshot.bundle, pre_extra)
    return pre_bundle, post_bundle


def created_keys_from_calls(calls: list) -> set[tuple[str, str]]:
    """Resource keys for POST writes the caller saw accepted."""

    keys: set[tuple[str, str]] = set()
    for call in calls:
        if getattr(call, "method", None) != "POST":
            continue
        status = getattr(call, "status", 0)
        if status < 200 or status >= 300:
            continue
        key = url_resource_key(getattr(call, "location", None) or "")
        if key is not None:
            keys.add(key)
    return keys


def known_keys_from_calls(calls: list) -> set[tuple[str, str]]:
    """Resource keys mentioned by harness calls, for delete checks."""

    keys: set[tuple[str, str]] = set()
    for call in calls:
        for url in (getattr(call, "url", None), getattr(call, "location", None)):
            key = url_resource_key(url or "")
            if key is not None:
                keys.add(key)
        keys.update(_keys_in_body(getattr(call, "response_body", None)))
    return keys


def url_resource_key(url: str) -> tuple[str, str] | None:
    """Return (type, id) for a resource URL, ignoring searches and history."""

    if not url:
        return None
    path = url.split("?", 1)[0]
    path = re.sub(r"/_history/\d+$", "", path).rstrip("/")
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return None
    resource_id = parts[-1]
    resource_type = parts[-2]
    if resource_type not in (*_SEARCH_TYPES, "Patient"):
        return None
    if resource_id in (*_SEARCH_TYPES, "Patient", "_history"):
        return None
    return resource_type, resource_id


def _load_type(
    http: HttpClient,
    base: str,
    patient_id: str,
    resource_type: str,
) -> tuple[list[dict], int | None, bool]:
    """Return (resources, total, complete). Complete lists are safe to diff in full."""

    body = _get_bundle(http, _count_url(base, patient_id, resource_type))
    total = _total(body)
    entries = bundle_resources(body)
    if _page_complete(body) and (total is None or total <= _FULL_CHART_LIMIT):
        return entries, total if total is not None else len(entries), True
    if total is not None and total > _FULL_CHART_LIMIT:
        return [], total, False
    if total is not None and total <= _FULL_CHART_LIMIT:
        found = _search_all(http, _full_search_url(base, patient_id, resource_type))
        return found, total, True
    if _page_complete(body):
        return entries, len(entries), True
    return [], total, False


def _count_url(base: str, patient_id: str, resource_type: str) -> str:
    return (
        f"{base}{resource_type}?patient={quote(str(patient_id), safe='')}"
        "&_summary=count"
    )


def _full_search_url(base: str, patient_id: str, resource_type: str) -> str:
    return (
        f"{base}{resource_type}?patient={quote(str(patient_id), safe='')}"
        "&_count=200&_sort=-_lastUpdated"
    )


def _changed_since(
    http: HttpClient,
    base: str,
    patient_id: str,
    resource_type: str,
    watermark: str,
) -> list[dict]:
    url = (
        f"{base}{resource_type}?patient={quote(str(patient_id), safe='')}"
        f"&_lastUpdated=gt{quote(watermark, safe='')}"
        "&_sort=-_lastUpdated&_count=200"
    )
    return _search_limited(http, url, _DELTA_PAGES)


def _probe_newest(http: HttpClient, base: str, patient_id: str, resource_type: str) -> dict | None:
    url = (
        f"{base}{resource_type}?patient={quote(str(patient_id), safe='')}"
        "&_sort=-_lastUpdated&_count=1"
    )
    body = _get_bundle(http, url)
    resources = bundle_resources(body)
    if not resources:
        return None
    return resources[0]


def _gold_labs(http: HttpClient, base: str, patient_id: str) -> list[dict]:
    """Labs the dose rules read, without paging the rest of the chart."""

    found: list[dict] = []
    for code, hours in _GOLD_LABS:
        params = (
            f"patient={quote(str(patient_id), safe='')}"
            f"&code={quote(code, safe='')}"
            "&_sort=-date&_count=20"
        )
        if hours is not None:
            cutoff = (FROZEN_NOW - timedelta(hours=hours)).isoformat()
            params += f"&date={quote('ge' + cutoff, safe='')}"
        found.extend(bundle_resources(_get_bundle(http, f"{base}Observation?{params}")))
    return found


def _before_image(
    http: HttpClient,
    base: str,
    resource_type: str,
    resource_id: str,
    watermark: str | None,
) -> dict | None:
    """Latest history version at or before the pre-trial watermark."""

    if not watermark:
        return None
    versions = _history(http, base, resource_type, resource_id)
    mark = _instant(watermark)
    if mark is None:
        return None
    prior: list[tuple[object, dict]] = []
    for version in versions:
        stamp = _last_updated(version)
        parsed = _instant(stamp) if stamp else None
        if parsed is not None and parsed <= mark:
            prior.append((parsed, version))
    if not prior:
        return None
    prior.sort(key=lambda item: item[0])
    return copy.deepcopy(prior[-1][1])


def _history(http: HttpClient, base: str, resource_type: str, resource_id: str) -> list[dict]:
    url = (
        f"{base}{resource_type}/{quote(str(resource_id), safe='')}"
        f"/_history?_count={_HISTORY_COUNT}&_sort=_lastUpdated"
    )
    result = http.request("GET", url, headers=_HEADERS)
    if result.status == 404 or result.status < 200 or result.status >= 300:
        return []
    if not isinstance(result.body, dict):
        return []
    return bundle_resources(result.body)


def _apply_deletes(
    http: HttpClient,
    base: str,
    snapshot: ChartSnapshot,
    post_bundle: dict,
    known_keys: set[tuple[str, str]],
    pre_extra: list[dict],
) -> dict:
    """Drop resources a large type no longer has, and record unresolved deletes."""

    missing: set[tuple[str, str]] = set()
    snapshot_keys = {
        key
        for resource in bundle_resources(snapshot.bundle)
        if (key := _resource_key(resource)) is not None
    }
    update_keys = {
        key for resource in pre_extra if (key := _resource_key(resource)) is not None
    }
    post_keys = {
        key
        for resource in bundle_resources(post_bundle)
        if (key := _resource_key(resource)) is not None
    }
    for chart in snapshot.types:
        if chart.complete or chart.count is None:
            continue
        new_count = _count_only(http, base, chart.resource_type, snapshot.patient_id)
        if new_count is None:
            continue
        created = sum(
            1
            for key in post_keys
            if key[0] == chart.resource_type and key not in snapshot_keys and key not in update_keys
        )
        n_delete = chart.count + created - new_count
        extra_create = new_count - chart.count - created
        if extra_create > 0:
            stubs = [
                {
                    "resourceType": chart.resource_type,
                    "id": f"unresolved-create-{chart.resource_type}-{index + 1}",
                }
                for index in range(extra_create)
            ]
            post_bundle = merge_resources(post_bundle, stubs)
        if n_delete <= 0:
            continue
        candidates = [
            key
            for key in [*snapshot_keys, *known_keys]
            if key is not None and key[0] == chart.resource_type
        ]
        found = 0
        checked = 0
        seen: set[tuple[str, str]] = set()
        for key in candidates:
            if key in seen or found >= n_delete or checked >= _MAX_DELETE_CHECKS:
                continue
            seen.add(key)
            checked += 1
            if _exists(http, base, key[0], key[1]):
                continue
            missing.add(key)
            found += 1
            if key not in snapshot_keys:
                pre_extra.append(
                    _before_image(http, base, key[0], key[1], chart.watermark)
                    or {"resourceType": key[0], "id": key[1]}
                )
        for index in range(n_delete - found):
            pre_extra.append(
                {
                    "resourceType": chart.resource_type,
                    "id": f"unresolved-delete-{chart.resource_type}-{index + 1}",
                }
            )
    if not missing:
        return post_bundle
    return _drop_keys(post_bundle, missing)


def _count_only(http: HttpClient, base: str, resource_type: str, patient_id: str | None) -> int | None:
    if not patient_id:
        return None
    body = _get_bundle(http, _count_url(base, patient_id, resource_type))
    return _total(body)


def _exists(http: HttpClient, base: str, resource_type: str, resource_id: str) -> bool:
    url = f"{base}{resource_type}/{quote(str(resource_id), safe='')}"
    result = http.request("GET", url, headers=_HEADERS)
    if result.status == 404:
        return False
    if result.status < 200 or result.status >= 300:
        raise RuntimeError(f"GET {url} returned status {result.status}")
    return True


def _drop_keys(bundle: dict, keys: set[tuple[str, str]]) -> dict:
    kept = []
    for resource in bundle_resources(bundle):
        key = _resource_key(resource)
        if key in keys:
            continue
        kept.append(resource)
    return _as_bundle(kept)


def _chart_for(snapshot: ChartSnapshot, resource_type: str) -> TypeChart | None:
    for chart in snapshot.types:
        if chart.resource_type == resource_type:
            return chart
    return None


def _get_bundle(http: HttpClient, url: str) -> dict:
    result = http.request("GET", url, headers=_HEADERS)
    if result.status < 200 or result.status >= 300:
        raise RuntimeError(f"GET {url} returned status {result.status}")
    if not isinstance(result.body, dict):
        raise RuntimeError(f"GET {url} did not return a Bundle")
    return result.body


def _page_complete(body: dict) -> bool:
    if next_page_url(body):
        return False
    total = _total(body)
    if total is None:
        return True
    return total <= len(bundle_resources(body))


def _total(body: dict) -> int | None:
    total = body.get("total") if isinstance(body, dict) else None
    if isinstance(total, bool) or not isinstance(total, int):
        return None
    return total


def _last_updated(resource: dict) -> str | None:
    meta = resource.get("meta")
    if not isinstance(meta, dict):
        return None
    stamp = meta.get("lastUpdated")
    if not stamp:
        return None
    return str(stamp)


def _instant(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _version_id(resource: dict):
    meta = resource.get("meta")
    if not isinstance(meta, dict):
        return None
    return meta.get("versionId")


def _resource_key(resource: dict) -> tuple[str, str] | None:
    resource_id = resource.get("id")
    if resource_id is None or resource_id == "":
        return None
    return (str(resource.get("resourceType") or ""), str(resource_id))


def _keys_in_body(body) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if isinstance(body, dict) and body.get("resourceType") == "Bundle":
        for resource in bundle_resources(body):
            key = _resource_key(resource)
            if key is not None:
                keys.add(key)
        return keys
    if isinstance(body, dict):
        key = _resource_key(body)
        if key is not None:
            keys.add(key)
    return keys


def _as_bundle(resources: list[dict]) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": resource} for resource in resources],
    }
