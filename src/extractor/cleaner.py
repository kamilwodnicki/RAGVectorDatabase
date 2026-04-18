import re
import unicodedata

_FOOTNOTE_RE = re.compile(
    r'\['
    r'(?:przypis|przyp\.?)\s+\w+\.?'
    r'|(?:red|tłum|aut|tł)\.?'
    r'|\d+'
    r'\]',
    re.IGNORECASE,
)

_STANDALONE_NUMBER_RE = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
_HYPHEN_BREAK_RE = re.compile(r'(\w)-\n(\w)')
_EXCESSIVE_NEWLINES_RE = re.compile(r'\n{3,}')
_EXCESSIVE_SPACES_RE = re.compile(r'[ \t]{2,}')


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean(text: str) -> str:
    text = _normalize_unicode(text)
    text = _FOOTNOTE_RE.sub('', text)
    text = _STANDALONE_NUMBER_RE.sub('', text)
    text = _HYPHEN_BREAK_RE.sub(r'\1\2', text)
    text = _EXCESSIVE_NEWLINES_RE.sub('\n\n', text)
    text = _EXCESSIVE_SPACES_RE.sub(' ', text)
    return text.strip()
