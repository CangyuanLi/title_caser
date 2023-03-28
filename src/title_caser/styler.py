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
    is_after_puncutation: bool
    is_article: bool
    is_coordinating_conjuction: bool
    is_first_word: bool
    is_first_word_of_paranthetical: bool
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
        self._title: str = title
        self._acronyms: set[str] = acronyms
        self._words: list[str] = self.clean_and_split_title()
        self._tagged_words: list[WordInfo] = self.tag_words()

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
    def is_preposition(word: str) -> bool:
        return word in PREPOSITIONS

    @staticmethod
    def is_hyphenated(word: str) -> bool:
        return "-" in word and word[-1] != "-"

    @staticmethod
    def lowercase_after_dash(word: str, dash_pos: int) -> str:
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
    def capitalize(word: str) -> str:
        """Capitalize but ignore punctuation. This is needed because the builtin .capitalize()
        method will return "(the" from "(the" instead of "(The".

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

    def tag_words(self):
        nltk_tags = nltk.pos_tag(self._words)
        tagged_words = []
        for idx, word_tag in enumerate(nltk_tags):
            word, tag = word_tag

            if idx != 0:
                previous_word, _ = nltk_tags[idx - 1]
            else:
                # If this is the first word, idx - 1 is -1, and is therefore the last word in the
                # list. Since we just use the previous word to test if it comes after punctuation
                # setting the previous word to something w/o punctuation makes is_after_punctuation
                # return False.
                previous_word = "SENTINEL"

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
                is_after_puncutation=self.is_after_punctuation(previous_word),
                is_first_word_of_paranthetical=self.is_first_word_of_paranthetical(
                    word
                ),
            )
            tagged_words.append(tagged_word)

        return tagged_words

    def chicago_case(self) -> str:
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
                last_s = cutils.find_last_index(word, "s")
                correct_word = word[:last_s].upper() + "s" + word[last_s + 1 :].upper()

            if word_info.is_hyphenated:
                dash_pos = word.index("-")
                correct_word = self.lowercase_after_dash(word, dash_pos)

            corrected.append(correct_word)

        return " ".join(corrected)


def main():
    title = "Corporate distress diagnosis: Comparisons using linear discriminant analysis and neural networks (the Italian experience)"
    title = Styler(title).chicago_case()
    print(title)


if __name__ == "__main__":
    main()
