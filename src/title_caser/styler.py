# Imports

import dataclasses
import string

import cutils
import nltk

from hardcoded_words import ACRONYMS, ARTICLES, PREPOSITIONS, VALID_TWO_LETTER_WORDS

# Globals

nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")


@dataclasses.dataclass
class WordInfo:
    word: str
    is_acronym: bool
    is_article: bool
    is_coordinating_conjuction: bool
    is_first_word: bool
    is_hyphenated: bool
    is_last_word: bool
    is_plural_acronym: bool
    is_preposition: bool
    is_subordinating_conjuction: bool


class Styler:
    def __init__(self, title: str, acronyms: set[str] = ACRONYMS):
        """Title is required to be passed in. Acronyms may be passed in since it is desireable
        for the user to be able to define a custom list of acronyms, e.g. for a specific field.

        Args:
            title (str): The title
            acronyms (set[str], optional): A set of acronyms. Defaults to ACRONYMS.
        """
        self._title = title
        self._acronyms = acronyms
        self._words = self.clean_and_split_title()
        self._tagged_words = self.tag_words()

    def clean_and_split_title(self) -> list[str]:
        title = self._title
        title = title.strip()  # strip whitespace off ends
        title = " ".join(title.split())  # normalize whitespace to one
        title = title.lower()

        return title.split()

    @staticmethod
    def is_article(word) -> bool:
        return word in ARTICLES

    @staticmethod
    def between_parantheses(word: str) -> bool:
        return word[0] == "(" and word[-1] == ")"

    def is_acronym(self, word: str) -> bool:
        """There is no good way of determining if a a word is an acronym. Therefore, several
        heuristics are used.

        1. Word is in a pre-defined list of acronyms
        2. Word contains "&" or "/", e.g. K&R, A/B. It is unlikely that a non-acronym word would
        contain these characters.
        3. Word is between parantheses AND word is four letters or less. In a title specifically,
        parantheses likely do not denote a note (like so), rather, they more likely contain an
        acronym, e.g. County Business Patterns (CBP)
        4. Word is two letters (and word is not in valid two letter words, a manual list of all
        two letter words). Note that this list does not contain "us", since it is much more likely
        that "us" refers to "US" in a title.

        Args:
            word (str): _description_

        Returns:
            _type_: _description_
        """
        word_no_punc = word.translate(word.maketrans("", "", string.punctuation))
        word_no_punc_len = len(word_no_punc)
        cond = (
            word_no_punc in self._acronyms
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
            assert s_pos is not None  # get mypy off my back

            # fmt: off
            word_no_s = word[:s_pos] + word[s_pos + 1:]

            return self.is_acronym(word_no_s)

        return False

    @staticmethod
    def is_coordinating_conjunction(tag: str) -> bool:
        return tag == "CC"

    @staticmethod
    def is_subordinating_conjuction(word: str, tag: str) -> bool:
        return word not in PREPOSITIONS and tag == "IN"

    @staticmethod
    def is_preposition(word: str) -> bool:
        return word in PREPOSITIONS

    @staticmethod
    def is_hyphenated(word: str) -> bool:
        return "-" in word and word[-1] != "-"

    def tag_words(self):
        nltk_tags = nltk.pos_tag(self._words)
        tagged_words = []
        for idx, word_tag in enumerate(nltk_tags):
            word, tag = word_tag
            tagged_word = WordInfo(
                word=word,
                is_acronym=self.is_acronym(word),
                is_plural_acronym=self.is_plural_acronym(word),
                is_article=self.is_article(word),
                is_coordinating_conjuction=self.is_coordinating_conjunction(tag),
                is_preposition=self.is_preposition(word),
                is_first_word=(idx == 0),
                is_last_word=(idx == len(nltk_tags) - 1),
                is_hyphenated=self.is_hyphenated(word),
                is_subordinating_conjuction=self.is_subordinating_conjuction(word, tag),
            )
            tagged_words.append(tagged_word)

        return tagged_words

    # Capitalize the first and the last word.
    # Capitalize nouns, pronouns, adjectives, verbs (including phrasal verbs such as “play with”), adverbs, and subordinate conjunctions.
    # Lowercase articles (a, an, the), coordinating conjunctions, and prepositions (regardless of length).
    # Lowercase the second word after a hyphenated prefix (e.g., Mid-, Anti-, Super-, etc.) in compound modifiers (e.g., Mid-year, Anti-hero, etc.).
    # Lowercase the ‘to’ in an infinitive (e.g., I Want to Play Guitar).


def main():
    title = "something here blah-dku"
    Styler(title).tag_words()


if __name__ == "__main__":
    main()