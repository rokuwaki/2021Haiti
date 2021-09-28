'''
This is a Python script to download SAC files used for finite-fault inversion.
  Download the SAC files, append P-picked arrivals as `amarker`, and save them
  at the disired directory (e.g., '__sac/').
usage: python fetchSAC.py directory_to_strore_SAC_files
author: Ryo Okuwaki (rokuwaki@geol.tsukuba.ac.jp) 2021-09-27
'''

import os
import sys
import obspy
import numpy as np
import pandas as pd
from glob import glob
from obspy.clients.fdsn import Client

def main():

    args = sys.argv
    outdir = args[1]
    os.makedirs(outdir, exist_ok=True) # directory to store SAC files

    # station list with the picked P arrival as `amarker`
    df = pd.read_csv('../materials/ffm/stationlist.csv', dtype=str)

    # if data location is 'nan' (not, e.g., 00 or 10), fill it with a blank ('')
    df = df.fillna('')

    for j in range(len(df)):

        if df.network[j] == 'GE':
            client = Client('GFZ')
        else:
            client = Client('IRIS')

        rawsac = client.get_waveforms(df.network[j], df.station[j], df.location[j],
                                      df.channel[j], obspy.UTCDateTime(df.starttime[j]), obspy.UTCDateTime(df.endtime[j]))
        rawsactrace = rawsac[0].copy()
        rawsactrace.stats.sac = obspy.core.AttribDict()
        rawsactrace.stats.sac.a = df.amarker[j] # picked P arrival time

        rawsacname = df.network[j]+'.'+df.station[j]+'.'+df.location[j]+'.'+\
                     df.channel[j]+'.'+ df.starttime[j]+'.'+df.endtime[j]+'.SAC'

        print(str(j+1).rjust(len(str(len(df))))+'/'+str(len(df)).rjust(len(str(len(df))))+': '+df.station[j].ljust(6), rawsacname)
        rawsactrace.write(os.path.join(outdir, rawsacname), format='SAC')

if __name__ == '__main__':
    main()
