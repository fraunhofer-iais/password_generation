"""
Script downloads the following datasets and extracts to specified download directory:
Single breaches:
    - Rockyou: 133.44 Mb (DEFAULT)
        https://download.weakpass.com/wordlists/90/rockyou.txt.gz
    - MySpace: 561.51 Kb (DEFAULT)
        https://download.weakpass.com/wordlists/22/myspace.txt.gz
    - Yahoo: 4.35 Mb (DEFAULT)
        https://download.weakpass.com/wordlists/44/yahoo.txt.gz
    - LinkedIn: 52.69 Mb (DEFAULT)
        https://weakpass.com/wordlist/567
        https://download.weakpass.com/wordlists/567/linkedin.dic.gz

Compilations:
    - SecList: Maintained list of breaches, 8.13 Mb (DEFAULT)
        https://github.com/danielmiessler/SecLists/tree/master/Passwords
        https://weakpass.com/wordlist/50
        https://download.weakpass.com/wordlists/50/10_million_password_list_top_1000000.txt.gz
    - SkullSecurity: Maintained list of breaches: 69.13 Mb (DEFAULT)
        https://blog.skullsecurity.org/
        https://weakpass.com/wordlist/671
        https://download.weakpass.com/wordlists/671/SkullSecurityComp.gz
    - weakpass 2a: Complete list of everything, 85.44 Gb (!)
        https://weakpass.com/wordlist/1919
        https://download.weakpass.com/wordlists/1919/weakpass_2a.gz
    - breachcompilation: Training and verification, 8.98 Gb (!)
        https://weakpass.com/wordlist/1849
        https://download.weakpass.com/wordlists/1849/breachcompilation.txt.gz
    - hashesorg2019: Training and verification, 12.79 Gb (!)
        https://hashes.org
        https://weakpass.com/wordlist/1851
        https://download.weakpass.com/wordlists/1851/hashesorg2019.gz
"""

import argparse
import os

from password_generation.utils.file_operations import download_file

download_mapping = {
    "rockyou": {
        "url": "https://download.weakpass.com/wordlists/90/rockyou.txt.gz",
        "filetype": ".txt.gz",
        "filename": "rockyou.txt",
    },
    "rockyou_small": {
        "url": "https://download.weakpass.com/wordlists/90/rockyou.txt.gz",
        "filetype": ".txt.gz",
        "filename": "rockyou.txt",
    },
    "myspace": {
        "url": "https://download.weakpass.com/wordlists/22/myspace.txt.gz",
        "filetype": ".txt.gz",
        "filename": "myspace.txt",
    },
    "yahoo": {
        "url": "https://download.weakpass.com/wordlists/44/yahoo.txt.gz",
        "filetype": ".txt.gz",
        "filename": "yahoo.txt",
    },
    "linkedin": {
        "url": "https://download.weakpass.com/wordlists/567/linkedin.dic.gz",
        "filetype": ".gz",
        "filename": "linkedin.txt",
    },
    "seclist": {
        "url": "https://download.weakpass.com/wordlists/50/10_million_password_list_top_1000000.txt.gz",
        "filetype": ".txt.gz",
        "filename": "seclist.txt",
    },
    "skullsecurity": {
        "url": "https://download.weakpass.com/wordlists/671/SkullSecurityComp.gz",
        "filetype": ".txt.gz",
        "filename": "skullsecurity.txt",
    },
    "2a": {
        "url": "https://download.weakpass.com/wordlists/1919/weakpass_2a.gz",
        "filetype": ".txt.gz",
        "filename": "2a.txt",
    },
    "breachcompilation": {
        "url": "https://download.weakpass.com/wordlists/1849/breachcompilation.txt.gz",
        "filetype": ".txt.gz",
        "filename": "breachcompilation.txt",
    },
    "hashesorg2019": {
        "url": "https://download.weakpass.com/wordlists/1851/hashesorg2019.gz",
        "filetype": ".gz",
        "filename": "hashesorg2019.txt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=list(download_mapping.keys()),
        default=["rockyou", "myspace", "yahoo", "linkedin", "seclist", "skullsecurity"],
        help='List of datasets to download and extract. \
                        Warning: "2a", "hashesorg2019" and "breachcompilation" are very large!',
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory. Will be created if it does not exist.",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.all:
        args.datasets = list(download_mapping.keys())

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    else:
        assert os.path.isdir(args.output)

    for dataset in args.datasets:
        print(f"Downloading dataset {dataset}.")
        download_file(
            url=download_mapping[dataset]["url"],
            output_file=os.path.join(args.output, download_mapping[dataset]["filename"]),
            filetype=download_mapping[dataset]["filetype"],
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
