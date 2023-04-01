from title_caser import ChicagoStyler

TITLE1 = (
    "Corporate distress diagnosis: Comparisons using linear discriminant analysis and"
    " neural networks (the Italian experience)"
)

TITLE2 = (
    "How Much do Bank Shocks AfFect Investment? evidence from Matched Bank-Firm Loan"
    " Data"
)


def test_chicago():
    assert (
        ChicagoStyler(TITLE1).title_case()
        == "Corporate Distress Diagnosis: Comparisons Using Linear Discriminant"
        " Analysis and Neural Networks (The Italian Experience)"
    )

    assert ChicagoStyler("twenty-first f-sharp").title_case() == "Twenty-First F-sharp"
