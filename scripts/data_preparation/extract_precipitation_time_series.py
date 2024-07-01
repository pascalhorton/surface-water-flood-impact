"""
Extracts the precipitation time series from the raw data and saves it as a netCDF files.
"""
import argparse
import numpy as np

from swafi.config import Config
from swafi.precip_combiprecip import CombiPrecip

config = Config()

START_DATE = '2005-01-01'
END_DATE = '2022-12-31'


def main(args):
    # Load CombiPrecip files
    precip = CombiPrecip()
    precip.load_data(config.get('DIR_PRECIP'))

    cids = np.unique(precip.domain.cids['ids_map'])
    cids = cids[cids != 0]

    if args.start:
        cids = cids[cids >= int(args.start)]
    if args.end:
        cids = cids[cids <= int(args.end)]

    for cid in cids:
        print(f'Processing CID {cid}')
        precip.save_nc_file_per_cid(cid, START_DATE, END_DATE)


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", help="Starting CID")
    parser.add_argument("-e", "--end", help="Ending CID")

    # Read arguments from command line
    args = parser.parse_args()

    main(args)
