import sys
import os
import argparse
from wesa_app.main import process_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="WeSA Command Line Interface",
        description="This script allows you to run WeSA via command line without visual interface"
    )
    parser.add_argument("-i", "--input", help="Path to input file containing protein pairs", required=True)
    parser.add_argument("-d", "--database",
                        help="Database to use for protein-protein interaction data.     ",
                        choices=['biogrid', 'bioplex', 'intact', 'intact_biogrid', 'intact_bioplex', 'biogrid_bioplex', 'all'],
                        default='biogrid')
    parser.add_argument("-o", "--output",
                        help="Path to output folder",
                        default="output/")
    args = parser.parse_args()
    print(args)

    print("Running WeSA CLI")

    # Read input file
    with open(args.input, "r") as f:
        job_input = f.read()

    n_pairs = len(set(job_input.splitlines()))
    print(f"Using input file: '{args.input}' containing {n_pairs} protein pairs")
    if n_pairs == 0:
        print("No protein pairs found in input file. Aborting")
        sys.exit()

    print(f"Using database: {args.database}")

    # Create folder if it doesn't exist
    if not os.path.exists(args.output):
        try:
            os.mkdir(args.output)
        except OSError:
            print(f"Creation of output directory {args.output} failed. Aborting")
            sys.exit()
    print(f"Output will be saved to: {args.output}")

    process_data(job_input, args.database, args.output)

    print("WeSA CLI finished")
