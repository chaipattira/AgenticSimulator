"""Plain-text parser/writer for MP-Gadget's `Key = Value  # comment` paramfile format.

Not a general INI library — MP-Gadget paramfiles are simple enough that reading them
as a dict of override-able lines, replacing specific keys, and rewriting is sufficient.
Overridden lines are rewritten as bare `Key = Value` (comment stripped); every other
line — including multi-value fields like `WindModel = ofjt10,isotropic` that this
wrapper never touches — passes through byte-for-byte.
"""


def override_paramfile(lines: list[str], overrides: dict[str, str]) -> list[str]:
    seen = set()
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            out.append(f"{key} = {overrides[key]}")
            seen.add(key)
        else:
            out.append(line)
    missing = set(overrides) - seen
    if missing:
        raise ValueError(f"paramfile keys not found to override: {missing}")
    return out
