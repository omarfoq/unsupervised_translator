import argparse
from translator.translator import Translator


parser = argparse.ArgumentParser()
parser.add_argument("-src", "--source", help="Source language")
parser.add_argument("-trgt", "--target", help="Target language")
parser.add_argument('words', metavar='N', type=str, nargs='+',
                    help='words for the accumulator')


args = parser.parse_args()
if __name__ == "__main__":
    translator = Translator(args.source, args.target, src_lang_size=10000, target_lang_size=10000)
    words_fr = args.words
    for word in words_fr:
        translator.nearest_neighbours_translation(word, k=1)



