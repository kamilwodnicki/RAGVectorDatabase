import pytest

from src.extractor.cleaner import clean

pytestmark = pytest.mark.unit


def test_empty_string_returns_empty():
    assert clean("") == ""


def test_whitespace_only_returns_empty():
    assert clean("   \n\t  ") == ""


def test_removes_cid_artifacts():
    assert clean("test(cid:3095)abc") == "testabc"


def test_removes_multiple_cid_artifacts():
    assert clean("(cid:1)foo(cid:22)bar(cid:333)baz") == "foobarbaz"


def test_fixes_hyphenated_line_break():
    assert clean("wspomi-\nnamy") == "wspominamy"


def test_preserves_regular_hyphens():
    assert clean("polsko-angielski") == "polsko-angielski"


def test_does_not_merge_across_paragraph_break():
    out = clean("wspomi-\n\nnamy")
    assert "wspomi-" in out or "wspominamy" not in out


def test_nfkc_decomposes_polish_dz_ligature():
    assert clean("\u01f3is") == "dzis"


def test_nfkc_decomposes_uppercase_dz_ligature():
    assert clean("\u01f1rzewo") == "DZrzewo"


def test_collapses_three_or_more_newlines():
    assert clean("a\n\n\n\nb") == "a\n\nb"


def test_preserves_double_newline():
    assert clean("a\n\nb") == "a\n\nb"


def test_collapses_multiple_spaces():
    assert clean("word1    word2") == "word1 word2"


def test_collapses_tabs():
    assert clean("word1\t\t\tword2") == "word1 word2"


def test_strips_leading_and_trailing_whitespace():
    assert clean("   hello world   ") == "hello world"


def test_combined_transforms():
    raw = "  Słowo(cid:12)  z  przerwa-\nną   linią  "
    assert clean(raw) == "Słowo z przerwaną linią"
