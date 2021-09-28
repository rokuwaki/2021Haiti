'''
This is a Python script to download publicly available materials required for generating figures.
usage: python fetch_public_data.py
author: Ryo Okuwaki (rokuwaki@geol.tsukuba.ac.jp) 2021-09-25
'''

import requests
import sys
import os
import pandas as pd
import io
import git

def main():

    # directory to store data
    outdir = '../materials/work'
    os.makedirs(outdir, exist_ok=True)

    # GEM Global Active Faults Database (GEM GAF-DB)
    try:
        git.Git(outdir).clone('https://github.com/GEMScienceTools/gem-global-active-faults.git')
    except git.GitCommandError as e:
        pass

    # USGS seismicity
    url = 'https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime=1900-08-14%2013:29:08&endtime=2022-12-11%2013:27:35&maxlatitude=20&minlatitude=17&maxlongitude=-71&minlongitude=-75&minmagnitude=0&orderby=time'
    urlData = requests.get(url).content
    df = pd.read_csv(io.StringIO(urlData.decode('utf-8')), skipinitialspace=True)
    df.to_csv(os.path.join(outdir, 'USGSseismicity.csv'))

    # SPUD GCMT
    URL = 'http://ds.iris.edu/spudservice/momenttensor/bundle/quakeml?evtminlat=17&evtmaxlat=20&evtminlon=-75&evtmaxlon=-71&evtmindepth=0.0&evtmaxdepth=700.0&evtminmag=0.0&evtmaxmag=10.0&evtstartdate=1950-01-01T00:00:00&evtenddate=2022-05-29T21:04:00&minstrike=0&maxstrike=360'
    response = requests.get(URL)
    with open(os.path.join(outdir, 'SPUD_QUAKEML_bundle.xml'), 'wb') as file:
        file.write(response.content)

    # SRCMOD slip model
    URL = 'http://equake-rc.info/media/srcmod/_fsp_files/s2010HAITIx01HAYE.fsp'
    response = requests.get(URL)
    with open(os.path.join(outdir, 's2010HAITIx01HAYE.fsp'), 'wb') as file:
        file.write(response.content)

    # Plate boundaries; Bird, 2003, doi:10.1029/2001GC000252
    try:
        git.Git(outdir).clone('https://github.com/fraxen/tectonicplates.git')
    except git.GitCommandError as e:
        pass

if __name__ == '__main__':
    main()
