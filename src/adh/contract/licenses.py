"""License normalization and output/source compatibility checks.

The contract requires ``output_license`` to be compatible with every license that
is actually retained (PHASE0-CONTRACT.md §0.3 cross-field invariants) and, per the
open decision §0.8.3 (recommendation adopted here), ``unknown``-license sources are
*quarantined*, never shipped into a split.

This is a deliberately small, auditable compatibility table — not a legal engine.
It encodes the conservative SPDX relationships needed by Phase 1:

  * Public-domain / permissive inputs (CC0, MIT, Apache-2.0, BSD, public-domain)
    can be redistributed under any permissive or copyleft output license.
  * CC-BY-* requires attribution downstream; compatible with CC-BY-* and
    more-restrictive CC licenses, not with CC0 (which drops the attribution term).
  * Stock-media licenses with no downstream obligations (Pexels) are permissive.
  * CC-BY-SA flows only into CC-BY-SA output (ShareAlike); NC/ND variants fall
    through the same-license-only tail — adding aliases never widens this table.
  * ``unknown`` is compatible with nothing — it must be quarantined.
"""

from __future__ import annotations

# Canonical aliases -> SPDX-ish key used internally.
_ALIASES: dict[str, str] = {
    "cc0": "CC0-1.0",
    "cc0-1.0": "CC0-1.0",
    "public-domain": "public-domain",
    "publicdomain": "public-domain",
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache2": "Apache-2.0",
    "bsd": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    # Creative Commons family, versioned and Openverse's bare short forms
    # ("by", "by-sa", …) — providers compose f"{license}-{version}" and it lands here.
    "cc": "CC-BY-4.0",
    "by": "CC-BY-4.0",
    "by-sa": "CC-BY-SA-4.0",
    "by-nc": "CC-BY-NC-4.0",
    "by-nd": "CC-BY-ND-4.0",
    "by-nc-sa": "CC-BY-NC-SA-4.0",
    "by-nc-nd": "CC-BY-NC-ND-4.0",
    "cc-by": "CC-BY-4.0",
    "cc-by-sa": "CC-BY-SA-4.0",
    "cc-by-nc": "CC-BY-NC-4.0",
    "cc-by-nd": "CC-BY-ND-4.0",
    "cc-by-nc-sa": "CC-BY-NC-SA-4.0",
    "cc-by-nc-nd": "CC-BY-NC-ND-4.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-nd-4.0": "CC-BY-ND-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc-by-nc-nd-4.0": "CC-BY-NC-ND-4.0",
    "pdm": "public-domain",
    "cc-pdm": "public-domain",
    "pexels": "Pexels-License",
    "pexels-license": "Pexels-License",
    "unknown": "unknown",
}

# Licenses with no downstream obligations beyond (at most) notice retention.
_PERMISSIVE = {"CC0-1.0", "public-domain", "MIT", "Apache-2.0", "BSD-3-Clause", "Pexels-License"}


def normalize_license(value: str) -> str:
    """Normalize a license string to a canonical internal key.

    Unknown strings are passed through unchanged (callers treat unrecognized
    licenses conservatively rather than guessing).
    """
    key = (value or "").strip().lower()
    return _ALIASES.get(key, value.strip() if value else "unknown")


def is_compatible(source_license: str, output_license: str) -> bool:
    """Can content under ``source_license`` be redistributed under ``output_license``?

    Conservative and one-directional. ``unknown`` is never compatible.
    """
    src = normalize_license(source_license)
    out = normalize_license(output_license)

    if src == "unknown" or out == "unknown":
        return False
    if src == out:
        return True
    # Permissive sources flow into anything.
    if src in _PERMISSIVE:
        return True
    # CC-BY-4.0 source: needs attribution preserved downstream. Allowed into
    # CC-BY-* / share-alike / non-commercial CC, not into CC0 (drops attribution).
    if src == "CC-BY-4.0":
        return out.startswith("CC-BY")
    if src == "CC-BY-SA-4.0":
        return out == "CC-BY-SA-4.0"
    if src == "CC-BY-NC-4.0":
        return out in {"CC-BY-NC-4.0"}
    # Anything else: only same-license redistribution is provably safe.
    return False


def incompatible_licenses(allowed: list[str], output_license: str) -> list[str]:
    """Return the subset of ``allowed`` that cannot be redistributed as ``output_license``."""
    return [lic for lic in allowed if not is_compatible(lic, output_license)]
