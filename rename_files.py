import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--old', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('--replace', action='store_true', default=False)
    args = parser.parse_args()

    for filename in os.listdir(args.folder):
        new_filename = filename.replace(args.old, args.new)
        if filename != new_filename:
            print(filename, '->', new_filename)
            src = f'{args.folder}/{filename}'
            dst = f'{args.folder}/{new_filename}'
            if args.replace:
                os.rename(src, dst)
    if args.replace:
        print('Renaming done!')

main()