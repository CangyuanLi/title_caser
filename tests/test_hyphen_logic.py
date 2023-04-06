import pytest

from title_caser import ChicagoStyler

TITLES = [
    "Under-the-Counter Transactions and Out-of-Fashion Initiatives",
    "Bed-and-Breakfast Options in Upstate New York",
    "Record-Breaking Borrowings from Medium-Sized Libraries",
    "Cross-Stitching for Beginners",
    "A History of the Chicago Lying-In Hospital",
    "The E-flat Concerto",
    "Self-Sustaining Reactions",
    "Anti-intellectual Pursuits",
    "Does E-mail Alter Thinking Patterns?",
    "A Two-Thirds Majority of Non-English-Speaking Representatives",
    "Ninety-Fifth Avenue Blues",
    "Atari's Twenty-First-Century Adherents",
    "Are We All a Little Stalker-ish?",
    "Organic Tomato-y Pasta with Vegetables",
]


@pytest.mark.parametrize("input, expected", [(title, title) for title in TITLES])
def test_chicago(input: str, expected: str):
    assert ChicagoStyler(input).title_case() == expected
