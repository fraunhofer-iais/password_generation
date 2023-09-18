import argparse
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generated", type=str, required=True, help="Path to generated.txt passwords file.")
    parser.add_argument("-r", "--reference", type=str, required=True, help="Path to test_passwords.txt test set file.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    generated = set(open(args.generated, "r").read().splitlines())
    reference = set(open(args.reference, "r").read().splitlines())
    print(f"Generated: {len(generated)}")
    print(f"Reference: {len(reference)}")
    intersection = generated.intersection(reference)
    print(f"Intersection: {len(intersection)}")
    print(f"Percentage: {len(intersection) / len(reference) * 100:.2f}%")


if __name__ == "__main__":
    main()
