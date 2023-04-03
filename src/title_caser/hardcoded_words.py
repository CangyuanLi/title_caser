ARTICLES = {"a", "an", "the"}

CONJUNCTIONS = {"and", "but", "for", "nor", "or", "so", "yet"}

PREPOSITIONS = {
    "abaft",
    "aboard",
    "about",
    "above",
    "absent",
    "across",
    "afore",
    "after",
    "against",
    "along",
    "alongside",
    "amid",
    "amidst",
    "among",
    "amongst",
    "apropos",
    "apud",
    "around",
    "as",
    "aside",
    "astride",
    "at",
    "athwart",
    "atop",
    "barring",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "besides",
    "between",
    "beyond",
    "by",
    "circa",
    "concerning",
    "despite",
    "down",
    "during",
    "except",
    "excluding",
    "failing",
    "following",
    "for",
    "from",
    "given",
    "in",
    "including",
    "inside",
    "into",
    "lest",
    "like",
    "mid",
    "midst",
    "minus",
    "modulo",
    "near",
    "next",
    "notwithstanding",
    "of",
    "off",
    "on",
    "onto",
    "opposite",
    "out",
    "outside",
    "over",
    "pace",
    "past",
    "per",
    "plus",
    "pro",
    "qua",
    "regarding",
    "round",
    "sans",
    "save",
    "since",
    "than",
    "through",
    "thru",
    "throughout",
    "thruout",
    "till",
    "times",
    "to",
    "toward",
    "towards",
    "under",
    "underneath",
    "unlike",
    "until",
    "unto",
    "up",
    "upon",
    "versus",
    "vs.",
    "vs",
    "v.",
    "v",
    "via",
    "vice",
    "with",
    "within",
    "without",
    "worth",
}

# Note that two letter words not in the valid list are already considered acronyms
ACRONYMS = {
    "aaa",
    "aamc",
    "aamr",
    "aarp",
    "aasa",
    "aau",
    "aauap",
    "abba",
    "ada",
    "adhd",
    "adsa",
    "adt",
    "adtv",
    "aer",
    "afch",
    "afdc",
    "afk",
    "ahec",
    "ahic",
    "ahima",
    "ahrq",
    "ama",
    "amex",
    "apa",
    "apr",
    "arm",
    "asap",
    "asl",
    "avm",
    "awol",
    "bafta",
    "bbb",
    "bbl",
    "bea",
    "cagr",
    "cao",
    "capex",
    "capm",
    "captcha",
    "cati",
    "cba",
    "cea",
    "ceo",
    "cer",
    "cessi",
    "cfa",
    "cfm",
    "cfo",
    "cia",
    "cia",
    "cisa",
    "cma",
    "cmo",
    "cmp",
    "cob",
    "coo",
    "covid",
    "cpa",
    "cpi",
    "cpp",
    "cpr",
    "crspr",
    "cso",
    "cto",
    "dhca",
    "djia",
    "dmv",
    "ebit",
    "ebitda",
    "eft",
    "eps",
    "esg",
    "etf",
    "ets",
    "fafsa",
    "faq",
    "fasb",
    "fbi",
    "fbr",
    "fca",
    "fda",
    "fdb",
    "fdic",
    "fema",
    "ffs",
    "fomo",
    "forex",
    "fpl",
    "frb",
    "gaap",
    "gdp",
    "gif",
    "gis",
    "gmp",
    "gnp",
    "gpa",
    "gps",
    "hud",
    "icbm",
    "icp",
    "icu",
    "idk",
    "imax",
    "ipo",
    "ira",
    "irl",
    "jel",
    "jfe",
    "jme",
    "jmp",
    "jpe",
    "jpeg",
    "json",
    "jstor",
    "kfc",
    "lbo",
    "llc",
    "loi",
    "lol",
    "mit",
    "mlb",
    "mmkt",
    "mtd",
    "nasa",
    "nasdaq",
    "nato",
    "nav",
    "nba",
    "nba",
    "nber",
    "ncnd",
    "nda",
    "neer",
    "nfl",
    "nfl",
    "nkfc",
    "nsa",
    "nyse",
    "nyu",
    "opex",
    "osha",
    "pdf",
    "pe",
    "pep",
    "pfd",
    "png",
    "pos",
    "potus",
    "ppp",
    "ppplf",
    "psp",
    "qje",
    "qje",
    "qtd",
    "qte",
    "rbi",
    "reit",
    "rfs",
    "risc",
    "risp",
    "roa",
    "roce",
    "roe",
    "roi",
    "roic",
    "rona",
    "ros",
    "sars",
    "sba",
    "sbo",
    "scotus",
    "ssh",
    "smqt",
    "smwt",
    "sec",
    "sif",
    "siv",
    "ssrn",
    "ssa",
    "ssl",
    "ssn",
    "ssp",
    "tcp",
    "tlc",
    "tirr",
    "thca",
    "tcu",
    "tcm",
    "tbd",
    "sncc",
    "swppa",
    "tfp",
    "tnh",
    "toefl",
    "trt",
    "tsa",
    "tsr",
    "ucla",
    "unicef",
    "url",
    "usa",
    "usada",
    "usc",
    "usda",
    "uss",
    "ussr",
    "uti",
    "wearc",
    "wia",
    "wic",
    "wic",
    "yolo",
    "ytd",
    "ytm",
}

VALID_TWO_LETTER_WORDS = {
    "am",
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "he",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "ox",
    "pi",
    "so",
    "to",
    "up",
    "we",
}  # no "us" because in title it is much more likely to refer to "US"

# These are special words that will simply be replaced according to a match
_special = {"iPhone", "iPad", "iPod", "E-mail", "WiFi"}
SPECIAL = {word.lower(): word for word in _special}

# List of prefixes- these cannot stand on their own
PREFIXES = {
    "ante",
    "anti",
    "contra",
    "dis",
    "ex",
    "hemi",
    "hypo",
    "infra",
    "inter",
    "intra",
    "non",
    "peri",
    "post",
    "pre",
    "pro",
    "re",
    "semi",
    "sub",
    "syn",
    "trans",
}
