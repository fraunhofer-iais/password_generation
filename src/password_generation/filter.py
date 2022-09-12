import math
import re
from dataclasses import dataclass
from typing import List, Optional

from password_generation.utils.valid_characters import (
    ascii_characters,
    ascii_punctuation,
    digits,
    letter_lower,
    letter_upper,
    password_punctuation,
)


class Filter:
    """
    Default behaviour:
    Remove passwords containing non-ascii characters, remove empty passwords, nothing else.

    Optional:
    Remove passwords with:
        - Unlikely password punctuation. Keep only .!?*$_#&/+
        - Any punctuation
        - Any digits
        - Length not in character range
        - Uppercase letters
        - Lowercase letters
        - Any letters
    """

    def __init__(
        self,
        keep_letters: str = "all",  # 'all', 'lower', 'upper', 'none'
        keep_digits: str = "all",  # 'all', 'none'
        keep_punctuation: str = "password",  # 'all', 'password', 'none'
        min_length: Optional[int] = 4,
        max_length: Optional[int] = 12,
    ):
        self.min_length = min_length or 0
        self.max_length = max_length or math.inf
        assert 0 <= self.min_length <= self.max_length
        self.keep_letters = keep_letters
        self.keep_digits = keep_digits
        self.keep_punctuation = keep_punctuation

        # Each regex searches for characters that are NOT allowed to appear in the password string.
        self.ascii_regex = re.compile(f"[^{re.escape(ascii_characters)}]")

        if keep_letters == "all":
            self.letters_regex = None
        elif keep_letters == "lower":
            self.letters_regex = re.compile(f"[{re.escape(letter_upper)}]")
        elif keep_letters == "upper":
            self.letters_regex = re.compile(f"[{re.escape(letter_lower)}]")
        elif keep_letters == "none":
            self.letters_regex = re.compile(f"[{re.escape(letter_lower + letter_upper)}]")
        else:
            raise NotImplementedError

        if keep_digits == "all":
            self.digits_regex = None
        elif keep_digits == "none":
            self.digits_regex = re.compile(f"[{re.escape(digits)}]")
        else:
            raise NotImplementedError

        if keep_punctuation == "all":
            self.punctuation_regex = None
        elif keep_punctuation == "password":
            non_password_punctuation = "".join([c for c in ascii_punctuation if c not in password_punctuation])
            self.punctuation_regex = re.compile(f"[{re.escape(non_password_punctuation)}]")
        elif keep_punctuation == "none":
            self.punctuation_regex = re.compile(f"[{re.escape(ascii_punctuation)}]")
        else:
            raise NotImplementedError

    def __call__(self, pwd: str) -> bool:
        return self.filter_ok(pwd)

    def filter_ok(self, pwd: str) -> bool:
        if not self.min_length <= len(pwd) <= self.max_length:
            return False
        if self.letters_regex is not None:
            if self.letters_regex.search(pwd) is not None:
                return False
        if self.digits_regex is not None:
            if self.digits_regex.search(pwd) is not None:
                return False
        if self.punctuation_regex is not None:
            if self.punctuation_regex.search(pwd) is not None:
                return False
        return True

    @property
    def config(self):
        return {
            "min_length": self.min_length,
            "max_length": self.max_length,
            "keep_letters": self.keep_letters,
            "keep_digits": self.keep_digits,
            "keep_punctuation": self.keep_punctuation,
        }


if __name__ == "__main__":
    pass
