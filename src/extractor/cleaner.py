import re
import unicodedata

_CID_RE = re.compile(r'\(cid:\d+\)')
_HYPHEN_BREAK_RE = re.compile(r'(\w)-\n(\w)')
_EXCESSIVE_NEWLINES_RE = re.compile(r'\n{3,}')
_EXCESSIVE_SPACES_RE = re.compile(r'[ \t]{2,}')


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CID_RE.sub('', text)
    text = _HYPHEN_BREAK_RE.sub(r'\1\2', text)
    text = _EXCESSIVE_NEWLINES_RE.sub('\n\n', text)
    text = _EXCESSIVE_SPACES_RE.sub(' ', text)
    return text.strip()
