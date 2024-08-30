"""
Extracts the precipitation time series from the raw data and saves it as a netCDF files.
"""
import argparse
import numpy as np

from swafi.config import Config
from swafi.precip_combiprecip import CombiPrecip

config = Config()

year_start = 2005
year_end = 2022


def main(args):
    # Load CombiPrecip files
    precip = CombiPrecip(year_start=year_start, year_end=year_end)
    precip.prepare_data(config.get('DIR_PRECIP'))

    cids = np.unique(precip.domain.cids['ids_map'])
    cids = cids[cids != 0]

    if args.start:
        cids = cids[cids >= int(args.start)]
    if args.end:
        cids = cids[cids <= int(args.end)]

    if args.reverse:
        cids = cids[::-1]

    for cid in cids:
        print(f'Processing CID {cid}')
        precip.save_nc_file_per_cid(cid, f'{year_start}-01-01', f'{year_end}-12-31')


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", help="Starting CID")
    parser.add_argument("-e", "--end", help="Ending CID")
    parser.add_argument('--reverse', action='store_true',
                        help='Process in reverse order')

    # Read arguments from command line
    args = parser.parse_args()

    main(args)
