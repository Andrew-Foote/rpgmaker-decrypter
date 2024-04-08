import argparse
from pathlib import Path
import numpy as np
from rpgmaker_decrypter_main import main

if __name__ == '__main__':
    np.seterr(over='ignore')

    arg_parser = argparse.ArgumentParser(
        prog='rpgmaker_decrypter',
        description='Decrypts RPG Maker XP game data files.'
    )

    arg_parser.add_argument('input_file', type=Path)
    arg_parser.add_argument('output_dir', type=Path)
    arg_parser.add_argument('-o', '--overwrite', action='store_true')
    arg_parser.add_argument('-p', '--profile', action='store_true')
    arg_parser.add_argument('-v', '--verbose', action='store_true')
    parsed_args = arg_parser.parse_args()
    
    main(
        parsed_args.input_file, parsed_args.output_dir,
        overwrite=parsed_args.overwrite,
        profile=parsed_args.profile,
        verbose=parsed_args.verbose
    )