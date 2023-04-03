# Imports

import dataclasses
import enum
import string

import cutils
import nltk
import spacy

from .hardcoded_words import (
    ACRONYMS,
    ARTICLES,
    PREFIXES,
    PREPOSITIONS,
    SPECIAL,
    VALID_TWO_LETTER_WORDS,
)

# Globals

nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)

# Types


class SpacyModel(enum.StrEnum):
    LG = "en_core_web_lg"
    SM = "en_core_web_sm"
    MD = "en_core_web_md"
    TRF = "en_core_web_trf"


DEFAULT_SPACY_MODEL = spacy.load(SpacyModel.LG)


class SpacyModelLoader:
    def __init__(self) -> None:
        # Load the default model
        self._models: dict[SpacyModel, spacy.Language] = {
            SpacyModel.LG: DEFAULT_SPACY_MODEL
        }

    def load(self, model: SpacyModel):
        if model not in self._models:
            nlp = spacy.load(model)
            self._models[model] = nlp

            return nlp

        return self._models[model]


LOADER = SpacyModelLoader()


@dataclasses.dataclass
class WordInfo:
    word: str = ""
    tag: str = ""
    is_acronym: bool = False
    is_after_puncutation: bool = False
    is_article: bool = False
    is_coordinating_conjuction: bool = False
    is_first_word: bool = False
    is_first_word_of_paranthetical: bool = False
    is_hyphenated: bool = False
    is_last_word: bool = False
    is_plural_acronym: bool = False
    is_prefix: bool = False
    is_preposition: bool = False
    is_proper: bool = False
    is_subordinating_conjuction: bool = False


class WhitespaceTokenizer(object):
    """By default, spacy splits on things other than the whitespace, including dashes,
    and so on. We want to split ONLY on the whitespace.
    """

    def __init__(self, vocab: spacy.vocab.Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str):
        words = text.split(" ")
        # All tokens "own" a subsequent space character in this tokenizer
        spaces = [True] * len(words)

        return spacy.tokens.Doc(self.vocab, words=words, spaces=spaces)


class Styler:
    def __init__(
        self,
        title: str,
        acronyms: set[str] = ACRONYMS,
        special: dict[str, str] = SPECIAL,
        model: SpacyModel = SpacyModel.LG,
    ):
        """Title is required to be passed in. Acronyms may be passed in since it is
        desireable for the user to be able to define a custom list of acronyms, e.g. for
        a specific field.

        Args:
            title (str): The title
            acronyms (set[str], optional): A set of acronyms. Defaults to ACRONYMS.
        """
        self._nlp = LOADER.load(model)
        self._nlp.tokenizer = WhitespaceTokenizer(self._nlp.vocab)

        self._title = title
        self._acronyms = acronyms
        self._special = special
        self._words = self.clean_title()
        self._tagged_words = self.tag_words(self._words)

    def clean_title(self) -> str:
        title = self._title
        title = title.strip()  # strip whitespace off ends
        title = " ".join(title.split())  # normalize whitespace to one
        title = title.lower()

        return title

    @staticmethod
    def is_article(word) -> bool:
        return word in ARTICLES

    @staticmethod
    def between_parantheses(word: str) -> bool:
        return word[0] == "(" and word[-1] == ")"

    @staticmethod
    def has_no_vowels(word: str) -> bool:
        vowels = "aeiouy"  # Consider "y" a vowel, don't want, e.g. spy to be an acronym

        return all(char not in vowels for char in word)

    def is_acronym(self, word: str) -> bool:
        """There is no good way of determining if a a word is an acronym. Therefore,
        several heuristics are used.

        1. Word is in a pre-defined list of acronyms
        2. Word contains "&" or "/", e.g. K&R, A/B. It is unlikely that a non-acronym
        word would contain these characters.
        3. Word is between parantheses AND word is four letters or less. In a title
        specifically, parantheses likely do not denote a note (like so), rather, they
        more likely contain an acronym, e.g. County Business Patterns (CBP)
        4. Word is two letters (and word is not in valid two letter words, a manual list
        of all two letter words). Note that this list does not contain "us", since it is
        much more likely that "us" refers to "US" in a title.

        Args:
            word (str): _description_

        Returns:
            _type_: _description_
        """
        words_with_no_vowels = {"crwth", "crwths", "cwm", "cwms"}
        word_no_punc = word.translate(word.maketrans("", "", string.punctuation))
        word_no_punc_len = len(word_no_punc)
        cond = (
            word_no_punc in self._acronyms
            or (
                self.has_no_vowels(word_no_punc)
                and word_no_punc not in words_with_no_vowels
            )
            or cutils.contains(word, {"&", "/"})
            or (self.between_parantheses(word) and word_no_punc_len <= 4)
            or word_no_punc_len == 2
            and word_no_punc not in VALID_TWO_LETTER_WORDS
        )

        return cond

    def is_plural_acronym(self, word: str) -> bool:
        word_no_trailing_punc = word.rstrip(string.punctuation)
        if word_no_trailing_punc[-1] == "s":
            s_pos = cutils.find_last_index(word, "s")
            word_no_s = word[:s_pos] + word[s_pos + 1 :]

            return self.is_acronym(word_no_s)

        return False

    @staticmethod
    def is_coordinating_conjunction(tag: str) -> bool:
        return tag == "CC"

    @staticmethod
    def is_subordinating_conjuction(word: str, tag: str) -> bool:
        return word not in PREPOSITIONS and tag == "IN"

    @staticmethod
    def is_proper(tag: str) -> bool:
        return tag in {"NNP", "NNPS"}

    @staticmethod
    def is_prefix(word: str) -> bool:
        return word in PREFIXES

    @staticmethod
    def is_preposition(word: str) -> bool:
        return word in PREPOSITIONS

    @staticmethod
    def is_hyphenated(word: str) -> bool:
        return "-" in word and word[-1] != "-"

    @staticmethod
    def lowercase_after_dash(word: str) -> str:
        dash_pos = word.index("-")
        before_dash = word[: dash_pos + 1].title()
        after_dash = word[dash_pos + 2 :]

        return before_dash + word[dash_pos + 1].lower() + after_dash

    @staticmethod
    def is_after_punctuation(previous_word: str) -> bool:
        """Tests if the word comes after a word that ends in punctuation, e.g.
            "Empirical Investment Equations: An Integrative Framework"

        Args:
            previous_word (str): The previous word in the title

        Returns:
            bool:
        """
        puncs = {":", "?", "!", ".", "--", "-"}

        return previous_word[-1] in puncs

    @staticmethod
    def is_first_word_of_paranthetical(word: str) -> bool:
        return cutils.contains(word[0], ("(", "{"))

    @staticmethod
    def uppercase_plural_acronyms(word: str) -> str:
        last_s = cutils.find_last_index(word, "s")
        correct_word = word[:last_s].upper() + "s" + word[last_s + 1 :].upper()

        return correct_word

    @staticmethod
    def capitalize(word: str) -> str:
        """Capitalize but ignore punctuation. This is needed because the builtin
        .capitalize() method will return "(the" from "(the" instead of "(The".

        Args:
            word (str): word

        Returns:
            str: capitalized word
        """
        word_lst = []
        first = True
        for c in word:
            if c not in string.punctuation and first:
                c = c.upper()
                first = False

            word_lst.append(c)

        return "".join(word_lst)

    def replace_special(self, word: str) -> str:
        key = word.lower()
        if key in self._special:
            corrected_word = self._special[key]
        else:
            corrected_word = word

        return corrected_word

    def tag_words(self, words: str) -> list[WordInfo]:
        doc = self._nlp(words)
        model_tags = [(token.text, token.tag_) for token in doc]
        tagged_words = []
        for idx, word_tag in enumerate(model_tags):
            word, tag = word_tag

            if idx != 0:
                previous_word, _ = model_tags[idx - 1]
            else:
                # If this is the first word, idx - 1 is -1, and is therefore the last
                # word in the list. Since we just use the previous word to test if it
                # comes after punctuation setting the previous word to something w/o
                # punctuation makes is_after_punctuation return False.
                previous_word = "SENTINEL"

            tagged_word = WordInfo(
                word=word,
                tag=tag,
                is_acronym=self.is_acronym(word),
                is_after_puncutation=self.is_after_punctuation(previous_word),
                is_article=self.is_article(word),
                is_coordinating_conjuction=self.is_coordinating_conjunction(tag),
                is_first_word=(idx == 0),
                is_first_word_of_paranthetical=self.is_first_word_of_paranthetical(
                    word
                ),
                is_hyphenated=self.is_hyphenated(word),
                is_last_word=(idx == len(model_tags) - 1),
                is_plural_acronym=self.is_plural_acronym(word),
                is_prefix=self.is_prefix(word),
                is_preposition=self.is_preposition(word),
                is_proper=self.is_proper(tag),
                is_subordinating_conjuction=self.is_subordinating_conjuction(word, tag),
            )
            tagged_words.append(tagged_word)

        return tagged_words


class ChicagoStyler(Styler):
    def __init__(
        self,
        title: str,
        acronyms: set[str] = ACRONYMS,
        special: dict[str, str] = SPECIAL,
    ) -> None:
        super().__init__(title, acronyms, special)

    def _correct_hyphenated_word(self, word: str) -> str:
        """
        As per the Chicago style manual:
        1. Always capitalize the first element.
        2. Capitalize any subsequent elements unless they are articles, prepositions,
        coordinating conjunctions (and, but, for, or, nor), or such modifiers as flat or
        sharp following musical key symbols.
        3. If the first element is merely a prefix or combining form that could not
        stand by itself as a word (anti, pre, etc.), do not capitalize the second
        element unless it is a proper noun or proper adjective.
        4. Capitalize the second element in a hyphenated spelled-out number
        (twenty-first, etc.) or hyphenated simple fraction
        (two-thirds in two-thirds majority).

        The 7 musical notes are A, B, C, D, E, F, G
        """
        tagged_words = self.tag_words(" ".join(word.split("-")))
        musical_notes = {"a", "b", "c", "d", "e", "f", "g"}
        musical_modifiers = {"sharp", "flat"}

        corrected = []
        for idx, word_info in enumerate(tagged_words):
            w = word_info.word

            if idx != 0:
                prev = tagged_words[idx - 1]
            else:
                prev = WordInfo()

            prev_w = prev.word

            # Since first part is always capitalized, short-circuit
            if idx == 0:
                corrected.append(w.capitalize())
                continue

            cw = w.capitalize()

            if w in musical_modifiers and prev_w in musical_notes:
                cw = w

            if (
                word_info.is_coordinating_conjuction
                or word_info.is_article
                or word_info.is_preposition
            ):
                cw = w

            if prev.is_prefix:
                cw = w

            if word_info.is_proper:
                cw = w.capitalize()

            corrected.append(cw)

        return "-".join(corrected)

    def title_case(self) -> str:
        corrected = []
        for word_info in self._tagged_words:
            word = word_info.word
            if (
                word_info.is_article
                or word_info.is_coordinating_conjuction
                or word_info.is_preposition
            ):
                correct_word = word.lower()
            else:
                correct_word = word.capitalize()

            if (
                word_info.is_first_word
                or word_info.is_last_word
                or word_info.is_after_puncutation
            ):
                correct_word = word.capitalize()

            if word_info.is_first_word_of_paranthetical:
                correct_word = self.capitalize(word)

            if word_info.is_acronym:
                correct_word = word.upper()

            if word_info.is_plural_acronym:
                correct_word = self.uppercase_plural_acronyms(word)

            if word_info.is_hyphenated:
                correct_word = self._correct_hyphenated_word(word)

            correct_word = self.replace_special(correct_word)
            corrected.append(correct_word)

        return " ".join(corrected)

    # Capitalize the first word of the title/heading and of any subtitle/subheading
    # Capitalize all major words (nouns, verbs including phrasal verbs such as “play with”, adjectives, adverbs, and pronouns) in the title/heading, including the second part of hyphenated major words (e.g., Self-Report not Self-report)
    # Capitalize all words of four letters or more.
    # Lowercase the second word after a hyphenated prefix (e.g., Mid-, Anti-, Super-, etc.) in compound modifiers (e.g., Mid-year, Anti-hero, etc.).
