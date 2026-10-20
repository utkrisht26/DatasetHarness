"""PII policy enforcement (OUTLINE §2.4, ADR-0004).

The contract's ``pii_policy`` (``redact`` | ``drop`` | ``allow``, default
``redact``) is enforced at curation time behind a :class:`PIIRedactor` port:

* ``redact`` — detected spans are replaced with ``<ENTITY_TYPE>`` markers and
  the redaction is stamped into the record's provenance;
* ``drop`` — PII-bearing candidates are dead-lettered, naming the entity types
  that triggered them;
* ``allow`` — text ships unscanned (the spec author explicitly accepts it).

Detection is Presidio-class NER, never bare regex: the production adapter wraps
``presidio-analyzer``/``presidio-anonymizer`` as the lazy ``adh[pii]`` extra.
The posture is **fail-closed**: an *explicitly configured* drop/redact policy
with the extra missing raises :class:`ConfigurationError` rather than shipping
PII unenforced. Only the implicit contract default degrades loudly (warning +
run note) — a missing dependency may skip screening nobody asked to skip, but
never silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PII_ENTITIES",
    "ConfigurationError",
    "PIIEntity",
    "PIIRedactor",
    "PresidioRedactor",
    "resolve_redactor",
]

# Direct identifiers only: Presidio's pattern + checksum recognizers, which are
# high-precision on any text. Generic-NER types (PERSON, LOCATION, DATE_TIME, NRP)
# and URL are excluded — on technical corpora they misfire on crate/trait names
# ("Unpin" -> PERSON, "Tokio" -> LOCATION, "Rust" -> DATE_TIME) and redacting URLs
# destroys content. Deployments whose data is about people opt back in via
# ADH_PII_ENTITIES (e.g. "PERSON,EMAIL_ADDRESS,PHONE_NUMBER").
DEFAULT_PII_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "CRYPTO",
    "US_SSN",
    "US_ITIN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "US_BANK_NUMBER",
    "MEDICAL_LICENSE",
    "UK_NHS",
)


class ConfigurationError(RuntimeError):
    """A configured pii_policy cannot be enforced (the presidio extra is absent)."""


@dataclass(frozen=True)
class PIIEntity:
    """One detected span: its type plus character offsets into the analyzed text."""

    entity_type: str
    start: int
    end: int
    score: float


@runtime_checkable
class PIIRedactor(Protocol):
    """The detection/redaction port (ADR-0004).

    ``analyze`` returns spans into the exact text passed in; ``redact`` replaces
    those spans with ``<TYPE>`` markers (implementations sort descending by
    offset so edits stay valid). Implementations also carry a ``name: str``
    used to stamp redaction provenance — kept off the Protocol body because a
    data member would disable ``issubclass`` runtime checks.
    """

    def analyze(self, text: str) -> list[PIIEntity]: ...

    def redact(self, text: str, entities: list[PIIEntity]) -> str: ...


def _import_presidio() -> None:
    """Import the presidio extras, or raise ImportError naming how to get them."""
    try:
        import presidio_anonymizer  # noqa: F401
        import presidio_analyzer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pii_policy enforcement requires the presidio extra "
            "(pip install 'adh[pii]'): presidio-analyzer + presidio-anonymizer"
        ) from exc


def resolve_redactor(
    spec,
    *,
    injected: PIIRedactor | None = None,
    notes: list[str] | None = None,
) -> PIIRedactor | None:
    """Pick the redactor for ``spec.pii_policy`` — or ``None`` under ``allow``.

    ``None`` means the implicit contract default (``"redact"``): the spec author
    never configured PII handling. Resolution order: ``allow`` disables scanning
    outright; an injected redactor (tests / custom deployments) bypasses the
    presidio requirement; otherwise the presidio extra is probed. Fail-closed:
    an *explicit* ``drop``/``redact`` policy without presidio raises
    :class:`ConfigurationError`. The implicit default degrades loudly instead —
    a WARNING plus a run note — so the skipped screening is visible in the
    summary. Explicitness is read from the value (``None`` vs a policy string),
    not ``model_fields_set``, because persisted specs round-trip through a
    complete ``model_dump()`` that would mark every defaulted field as set.
    """
    policy = getattr(spec, "pii_policy", None)
    if policy == "allow":
        return None
    if injected is not None:
        return injected
    try:
        _import_presidio()
    except ImportError as exc:
        if policy is not None:
            raise ConfigurationError(
                f"pii_policy={policy!r} requires the presidio extra "
                "(pip install 'adh[pii]'); refusing to ship PII unenforced (fail-closed)"
            ) from exc
        message = (
            f"pii policy '{policy}' (contract default) NOT enforced: the presidio "
            "extra is not installed (adh[pii]); proceeding without PII screening"
        )
        logger.warning(message)
        if notes is not None:
            notes.append(message)
        return None
    from adh.settings import get_settings

    return PresidioRedactor(entities=get_settings().pii_entities or None)


class PresidioRedactor:
    """:class:`PIIRedactor` over presidio-analyzer NER + marker redaction.

    ``entities`` defaults to :data:`DEFAULT_PII_ENTITIES`; names the analyzer does
    not support raise :class:`ConfigurationError` up front rather than mid-build.
    """

    name = "presidio"

    def __init__(
        self, entities: list[str] | tuple[str, ...] | None = None, score_threshold: float = 0.5
    ) -> None:
        _import_presidio()
        from presidio_analyzer import AnalyzerEngine

        self._analyzer = AnalyzerEngine()
        self._entities = tuple(entities) if entities else DEFAULT_PII_ENTITIES
        unknown = sorted(set(self._entities) - set(self._analyzer.get_supported_entities()))
        if unknown:
            raise ConfigurationError(f"unsupported PII entity type(s): {', '.join(unknown)}")
        self._threshold = score_threshold

    @property
    def entities(self) -> tuple[str, ...]:
        """The entity types this redactor screens for."""
        return self._entities

    def analyze(self, text: str) -> list[PIIEntity]:
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=list(self._entities),
            score_threshold=self._threshold,
        )
        return [PIIEntity(r.entity_type, r.start, r.end, r.score) for r in results]

    def redact(self, text: str, entities: list[PIIEntity]) -> str:
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            text = text[: entity.start] + f"<{entity.entity_type}>" + text[entity.end :]
        return text
