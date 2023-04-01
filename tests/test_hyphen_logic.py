import pytest

from title_caser import ChicagoStyler

TITLE1 = "Under-the-Counter Transactions and Out-of-Fashion Initiatives"
TITLE2 = "Bed-and-Breakfast Options in Upstate New York"
TITLE3 = "Record-Breaking Borrowings from Medium-Sized Libraries"
TITLE4 = "Cross-Stitching for Beginners"
TITLE5 = "A History of the Chicago Lying-In Hospital"
TITLE6 = "The E-flat Concerto"
TITLE7 = "Self-Sustaining Reactions"
TITLE8 = "Anti-intellectual Pursuits"
TITLE9 = "Does E-mail Alter Thinking Patterns?"
TITLE10 = "A Two-Thirds Majority of Non-English-Speaking Representatives"
TITLE11 = "Ninety-Fifth Avenue Blues"
TITLE12 = "Atari's Twenty-First-Century Adherents"

TITLES = [
    TITLE1,
    TITLE2,
    TITLE3,
    TITLE4,
    TITLE5,
    TITLE6,
    TITLE6,
    TITLE8,
    TITLE9,
    TITLE10,
    TITLE11,
    TITLE12,
]


@pytest.mark.parametrize("input, expected", [(title, title) for title in TITLES])
def test_chicago(input: str, expected: str):
    assert ChicagoStyler(input).title_case() == expected
