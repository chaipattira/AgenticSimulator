import pytest

from simulator.mpgadget_paramfile import override_paramfile


def test_override_replaces_matching_keys():
    lines = [
        "OutputDir = output # Directory for output",
        "Ngrid = 32 # comment",
        "BoxSize = 4000",
        "",
        "Omega0 = 0.2814",
    ]
    out = override_paramfile(lines, {"Ngrid": "48", "Omega0": "0.3"})
    assert "Ngrid = 48" in out
    assert "Omega0 = 0.3" in out
    # untouched lines pass through byte-for-byte
    assert "OutputDir = output # Directory for output" in out
    assert "BoxSize = 4000" in out
    assert "" in out


def test_override_strips_comment_on_overridden_line():
    lines = ["Sigma8 = 0.810      # power spectrum normalization"]
    out = override_paramfile(lines, {"Sigma8": "0.75"})
    assert out == ["Sigma8 = 0.75"]


def test_override_preserves_multivalue_fields_untouched():
    lines = ["WindModel = ofjt10,isotropic", "BlackHoleFeedbackMethod = spline | mass"]
    out = override_paramfile(lines, {})
    assert out == lines


def test_override_raises_on_missing_key():
    lines = ["Omega0 = 0.2814"]
    with pytest.raises(ValueError, match="NotAKey"):
        override_paramfile(lines, {"NotAKey": "1"})
