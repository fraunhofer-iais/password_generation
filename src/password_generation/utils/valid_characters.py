import re
import string
from typing import Dict, List, Union

Vocab = Union[str, List[str], Dict[str, int], List[int]]

letter_upper = string.ascii_uppercase
letter_lower = string.ascii_lowercase
digits = string.digits
password_punctuation = ".!?*$_#&/+ "  # most common punctuation mentioned in the password_generation report
ascii_punctuation = string.punctuation + " "  # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ and whitespace

ascii_characters = letter_upper + letter_lower + digits + ascii_punctuation
password_characters = letter_upper + letter_lower + digits + password_punctuation

ascii_mapping = {char: i for i, char in enumerate(ascii_characters)}
password_mapping = {char: i for i, char in enumerate(password_characters)}

_valid_characters_regex = re.compile(f"[^{re.escape(password_characters)}]")


def is_valid_password(pwd: str) -> bool:
    if not pwd:
        return False
    return _valid_characters_regex.search(pwd) is None


if __name__ == "__main__":
    print(is_valid_password("abc23!!."))  # -> True
    print(is_valid_password("abcde§§"))  # -> False
    print(is_valid_password("ab cde"))  # -> False
    print(is_valid_password("ab-cde"))  # -> False
    print(is_valid_password("ab\\cde"))  # -> False
    print(is_valid_password(""))  # -> True
