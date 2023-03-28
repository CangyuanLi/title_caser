from title_caser import Styler

TITLE1 = "Corporate distress diagnosis: Comparisons using linear discriminant analysis and neural networks (the Italian experience)"
TITLE2 = "How Much do Bank Shocks Affect Investment? Evidence from Matched Bank-Firm Loan Data"


def test_chicago():
    assert (
        Styler(TITLE1).chicago_case()
        == "Corporate Distress Diagnosis: Comparisons Using Linear Discriminant Analysis and Neural Networks (The Italian Experience)"
    )
