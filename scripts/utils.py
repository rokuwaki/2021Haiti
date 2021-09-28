import numpy as np
from netCDF4 import Dataset
from cmcrameri import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.patheffects as path_effects
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import patches
import matplotlib.colors as mcolors
import obspy
from pyrocko.plot import beachball
from pyrocko import moment_tensor as pmt
from pyrocko.model import load_events
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84
import pandas as pd
import shapefile

from scipy.interpolate import griddata
import scipy.io as io

import warnings
warnings.filterwarnings("ignore")

import os

# some default font setting using Open Sans https://fonts.google.com/specimen/Open+Sans
initfontsize = 10
mpl.rc('axes', labelsize=initfontsize, titlesize=initfontsize)
mpl.rc('xtick', labelsize=initfontsize)
mpl.rc('ytick', labelsize=initfontsize)
mpl.rc('legend', fontsize=initfontsize, edgecolor='none')
mpl.rc('savefig', dpi=600, transparent=False, facecolor='w')
mpl.rc('font', size=initfontsize)

'''
mpl.rcParams['font.weight'] = 400
mpl.rcParams['font.family'] = 'Open Sans'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Open Sans'
mpl.rcParams['mathtext.it'] = 'Open Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Open Sans:bold'
'''
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'

# default colour cycle ('C0', 'C1', ..., 'C7') using `Set2`
cmap = plt.get_cmap('Set2', 8)
from cycler import cycler
custom_color_cycle=[]
for i in range(cmap.N):
    rgb = cmap(i)[:3]
    custom_color_cycle.append(str(mpl.colors.rgb2hex(rgb)))
plt.rc('axes', prop_cycle=(cycler(color=custom_color_cycle)))

# connvert coordinates of moment tensor from USE to NED
def convertUSEtoNED(focmec):
    '''
    convert basis-moment tensors
        from mrr, mtt, mpp, mrt, mrp, mtp
          to mnn, mee, mdd, mne, mnd, med
    https://pyrocko.org/docs/current/library/reference/moment_tensor.html#pyrocko.moment_tensor.MomentTensor
    usage:
    focmec = convertUSEtoNED(focmec[0],focmec[1],focmec[2],focmec[3],focmec[4],focmec[5])
    '''
    mrr = focmec[0]
    mtt = focmec[1]
    mpp = focmec[2]
    mrt = focmec[3]
    mrp = focmec[4]
    mtp = focmec[5]
    #
    return [mtt, mpp, mrr, -mtp, mrt, -mrp]


# load model parameters from `fort.40` and store the values as pandas data frame
def load_fort40(infile):
    col_names = [ 'c{0:02d}'.format(i) for i in range(10)]
    df = pd.read_table(infile, names=col_names, header=None, delimiter='\t')
    tmp1 = [ float(x) for x in df['c00'][1].replace(' ', ',').split(',') if x ]
    tmp3 = [ float(x) for x in df['c00'][3].replace(' ', ',').split(',') if x ]
    tmp5 = [ float(x) for x in df['c00'][5].replace(' ', ',').split(',') if x ]
    tmp7 = [ float(x) for x in df['c00'][7].replace(' ', ',').split(',') if x ]
    df = pd.DataFrame({ 'moment'   : tmp1[0],
                        'mw'       : tmp1[1],
                        'rigidity' : tmp1[2],
                        'lat'      : tmp1[3],
                        'lon'      : tmp1[4],
                        'depth'    : tmp1[5],
                        'vr'       : tmp1[6],
                        'nsurface' : tmp1[7],

                        'strike'   : tmp3[0],
                        'dip'      : tmp3[1],

                        'xx'       : tmp5[0],
                        'yy'       : tmp5[1],
                        'mn'       : int(tmp5[2]),
                        'nn'       : int(tmp5[3]),
                        'm0'       : int(tmp5[4]),
                        'n0'       : int(tmp5[5]),
                        'tr'       : tmp5[6],
                        'jtn'      : int(tmp5[7]),
                        'icmn'     : int(tmp5[8]),

                        'variance' : tmp7[0],

                      }, index=[0])
    return df


# mpl-ish tick style for Basemap
def mapTicksBasemap(fig,m,ax,xtickint,ytickint,lonmin,lonmax,latmin,latmax, dpoint):
    axp=ax.get_position()
    ax2 = fig.add_axes([axp.x0, axp.y0, axp.width, axp.height])

    #xticklabels = np.arange(int(lonmin - lonmin%xtickint), int(lonmax - lonmax%xtickint) + xtickint, xtickint)
    #yticklabels = np.arange(int(latmin - latmin%ytickint), int(latmax - latmax%ytickint) + ytickint, ytickint)

    xticklabels = np.arange(lonmin - lonmin%xtickint, lonmax - lonmax%xtickint + xtickint, xtickint)
    yticklabels = np.arange(latmin - latmin%ytickint, latmax - latmax%ytickint + ytickint, ytickint)

    tmp = [m(i, latmin) for i in xticklabels]
    xticks = [tmp[i][0] for i in range(len(xticklabels))]
    ax2.set_xticks(xticks)

    tmp = [m(lonmin, i) for i in yticklabels]
    yticks = [tmp[i][1] for i in range(len(yticklabels))]
    ax2.set_yticks(yticks)

    #print(xticklabels)
    #tmp = [ str(xticklabels[i])+'$\degree$' for i in range(len(xticklabels)) ]
    if dpoint == 0:
        tmp = [ str('{:.0f}'.format(xticklabels[i]))+'$\degree$E' if xticklabels[i] > 0 and xticklabels[i] <= 180 else \
               str('{:.0f}'.format(abs(xticklabels[i])))+'$\degree$W' if  xticklabels[i] < 0 else \
               str('{:.0f}'.format(abs(360-xticklabels[i])))+'$\degree$W' if  xticklabels[i] > 180 else \
               str('{:.0f}'.format(xticklabels[i]))+'$\degree$' for i in range(len(xticklabels)) ]
        ax2.set_xticklabels(tmp)

        #tmp = [ str(yticklabels[i])+'$\degree$' for i in range(len(yticklabels)) ]
        tmp = [ str('{:.0f}'.format(yticklabels[i]))+'$\degree$N' if yticklabels[i] > 0 else \
               str('{:.0f}'.format(abs(yticklabels[i])))+'$\degree$S' if  yticklabels[i] < 0 else \
               str('{:.0f}'.format(yticklabels[i]))+'$\degree$' for i in range(len(yticklabels)) ]
        ax2.set_yticklabels(tmp)

    elif dpoint == 1:
        tmp = [ str('{:.1f}'.format(xticklabels[i]))+'$\degree$E' if xticklabels[i] > 0 and xticklabels[i] <= 180 else \
               str('{:.1f}'.format(abs(xticklabels[i])))+'$\degree$W' if  xticklabels[i] < 0 else \
               str('{:.1f}'.format(abs(360-xticklabels[i])))+'$\degree$W' if  xticklabels[i] > 180 else \
               str('{:.1f}'.format(xticklabels[i]))+'$\degree$' for i in range(len(xticklabels)) ]
        ax2.set_xticklabels(tmp)

        #tmp = [ str(yticklabels[i])+'$\degree$' for i in range(len(yticklabels)) ]
        tmp = [ str('{:.1f}'.format(yticklabels[i]))+'$\degree$N' if yticklabels[i] > 0 else \
               str('{:.1f}'.format(abs(yticklabels[i])))+'$\degree$S' if  yticklabels[i] < 0 else \
               str('{:.1f}'.format(yticklabels[i]))+'$\degree$' for i in range(len(yticklabels)) ]
        ax2.set_yticklabels(tmp)

    elif dpoint == 2:
        tmp = [ str('{:.2f}'.format(xticklabels[i]))+'$\degree$E' if xticklabels[i] > 0 and xticklabels[i] <= 180 else \
               str('{:.2f}'.format(abs(xticklabels[i])))+'$\degree$W' if  xticklabels[i] < 0 else \
               str('{:.2f}'.format(abs(360-xticklabels[i])))+'$\degree$W' if  xticklabels[i] > 180 else \
               str('{:.2f}'.format(xticklabels[i]))+'$\degree$' for i in range(len(xticklabels)) ]
        ax2.set_xticklabels(tmp)

        #tmp = [ str(yticklabels[i])+'$\degree$' for i in range(len(yticklabels)) ]
        tmp = [ str('{:.2f}'.format(yticklabels[i]))+'$\degree$N' if yticklabels[i] > 0 else \
               str('{:.2f}'.format(abs(yticklabels[i])))+'$\degree$S' if  yticklabels[i] < 0 else \
               str('{:.2f}'.format(yticklabels[i]))+'$\degree$' for i in range(len(yticklabels)) ]
        ax2.set_yticklabels(tmp)

    elif dpoint == 3:
        tmp = [ str('{:.3f}'.format(xticklabels[i]))+'$\degree$E' if xticklabels[i] > 0 and xticklabels[i] <= 180 else \
               str('{:.3f}'.format(abs(xticklabels[i])))+'$\degree$W' if  xticklabels[i] < 0 else \
               str('{:.3f}'.format(abs(360-xticklabels[i])))+'$\degree$W' if  xticklabels[i] > 180 else \
               str('{:.3f}'.format(xticklabels[i]))+'$\degree$' for i in range(len(xticklabels)) ]
        ax2.set_xticklabels(tmp)

        #tmp = [ str(yticklabels[i])+'$\degree$' for i in range(len(yticklabels)) ]
        tmp = [ str('{:.3f}'.format(yticklabels[i]))+'$\degree$N' if yticklabels[i] > 0 else \
               str('{:.3f}'.format(abs(yticklabels[i])))+'$\degree$S' if  yticklabels[i] < 0 else \
               str('{:.3f}'.format(yticklabels[i]))+'$\degree$' for i in range(len(yticklabels)) ]
        ax2.set_yticklabels(tmp)

    xlimmin, ylimmin = m(lonmin, latmin)
    xlimmax, ylimmax = m(lonmax, latmax)
    ax2.set_xlim(xlimmin, xlimmax)
    ax2.set_ylim(ylimmin, ylimmax)
    minZorder=min([_.zorder for _ in ax.get_children()])
    ax2.set_zorder(minZorder-1)
    ax2.set_facecolor('none')
    #ax.set_xticks([]); ax.set_yticks([])

    return ax2


# back-projection
def drawBP(OP2, bplat, bplon, bpcolumn, bprow, ax, m):
    x, y = m(bplon, bplat)
    winnum1 = 5
    Fs = 100
    ts = 5
    t1 = 10
    iwinstep = ts*Fs
    iwinlen = t1*Fs
    levels = np.arange(90, 100+2, 2)
    colors = cm.batlow(np.linspace(0,1,winnum1+1))
    power = OP2['BP'][0][0][0]['BAKM0'][0]
    for iwin in range(winnum1):
        wstp = iwin * iwinstep
        wedp = wstp + iwinlen
        rms = np.sqrt(np.mean(power[:,wstp:wedp]**2, axis=1))
        rms = rms / np.max(rms) * 100
        rms = rms.reshape(bpcolumn, bprow)
        sc = ax.contour(x, y, rms, levels=levels, colors=colors[iwin].reshape(-1,4), linewidths=1.2, alpha=1, zorder=int(10-iwin))

        
def drawlegend(fig, ax, m):
    winnum1 = 5
    ts = 5
    t1 = 10
    colors = cm.batlow(np.linspace(0,1,winnum1+1))
    axp = ax.get_position()
    axlegend = fig.add_axes([axp.x1+0.005, axp.y0, axp.height*0.1, axp.height])
    angle = 1
    tmpt = np.arange(0, winnum1, 1) * ts + t1/2
    for i in range(winnum1):
        #x, y = 0.25, 0.33-i*0.05 # range 7 version
        x, y = 0.25, 0.56-i*0.04
        for width, height in zip([0.08, 0.05], [0.03, 0.02]):
            e1 = patches.Ellipse((x, y), width*5, height, angle=angle, linewidth=0.75, fill=False, edgecolor=colors[i])
            axlegend.add_patch(e1)

        tmpt_s = i * ts
        tmpt_e = tmpt_s + t1
        axlegend.text(x+0.28, y, str(tmpt_s)+'–'+str(tmpt_e)+' s', va='center', size=8)
        axlegend.axis('off')

    axlegend.text(0, 0.6, 'HF radiation', va='bottom', ha='left', size=8)

    axlegend.text(0, 0.25, 'Seismicity', va='bottom', ha='left', size=8)

    reflon = -71.9
    for x, y, mw in zip([reflon, reflon, reflon], [18, 17.98, 17.97], [7, 5, 3]):
        sc = ax.scatter(x, y+0.1, s=np.exp(mw/2), marker='o', facecolor='C7',
                edgecolor='k', alpha=1, zorder=11, lw=0.75, label='USGS', clip_on=False)
    for x, y, mw in zip([reflon, reflon, reflon], [18.04, 18, 17.96], [7, 5, 3]):
        ax.text(x+0.08, y+0.1, 'M'+str(mw), va='center', ha='left', size=6)

    focmecs = [3.640 ,-4.200,0.560,-2.640, -2.500, 4.150]
    focmecs = convertUSEtoNED(focmecs)
    x = reflon
    for y, mag in zip([17.9, 17.9-0.04, 17.9-0.06],[7, 5, 3]):
        size = mag * 2
        tmp = beachball.plot_beachball_mpl(focmecs, ax, size=size, position=(x, y+0.05),
                                 beachball_type='deviatoric', edgecolor='none', color_t='C7',
                                 color_p='w', linewidth=0.75, alpha=1, zorder=int(100-mag*10))
        tmp.set_clip_on(False)
        tmp = beachball.plot_beachball_mpl(focmecs, ax, size=size, position=(x, y+0.05),
                                 beachball_type='dc', edgecolor='k', color_t='none',
                                 color_p='none', linewidth=0.75, alpha=1, zorder=int(100-mag*10))
        tmp.set_clip_on(False)

    for x, y, mw in zip([reflon, reflon, reflon], [18.04, 18, 17.96], [7, 5, 3]):
        ax.text(x+0.08, y-0.07, 'M'+str(mw), va='center', ha='left', size=6)

        
def drawseismicity(ax, m, elon, elat):
    df = pd.read_csv('../materials/work/USGSseismicity.csv').sort_values(by='mag', ascending=False)
    tmpdf = df[(df['time'] < obspy.UTCDateTime('2021-08-14')) & (df['mag'] >= 0) & (df['id'] != 'usp000h60h')]
    x, y = m(tmpdf.longitude.values, tmpdf.latitude.values)
    sc = ax.scatter(x, y, s=np.exp(tmpdf.mag.values/2), marker='o', facecolor='C7',
            edgecolor='k', alpha=1, zorder=11, lw=0.75, label='USGS')

    tmpdf = df[(df['time'] >= obspy.UTCDateTime('2021-08-14')) & (df['time'] <= obspy.UTCDateTime('2021-09-14')) & (df['mag'] >= 0) & (df['id'] != 'us6000f65h')]
    x, y = m(tmpdf.longitude.values, tmpdf.latitude.values)
    sc = ax.scatter(x, y, s=np.exp(tmpdf.mag.values/2), marker='o', facecolor='C5',
            edgecolor='k', alpha=1, zorder=11, lw=0.75, label='USGS')

    tmpdf = df[(df['id'] == 'usp000h60h')]
    x, y = m(tmpdf.longitude.values, tmpdf.latitude.values)
    sc=ax.scatter(x, y, s=600*(7/7.2), marker='*', facecolor='none', edgecolor='k', alpha=1, lw=1.5, zorder=100, 
                  path_effects=[path_effects.Stroke(linewidth=2.5, foreground='w', alpha=1), path_effects.Normal()])

    x, y = m(elon, elat)
    sc=ax.scatter(x, y, s=600, marker='*', facecolor='none', edgecolor='k', alpha=1, lw=1.5, zorder=100, 
                  path_effects=[path_effects.Stroke(linewidth=2.5, foreground='w', alpha=1), path_effects.Normal()])

    
    # Bakun+2012BSSA, Table 1
    for y, x, mi, year in zip([18.42,18.36,18.54,18.50,18.55],[-72.65,-70.84,-72.32,-72.86,-73.17],
                        [6.6,7.4,6.6,7.5,6.3],[1701,1751,1751,1770,1860]):
        x, y = m(x, y)
        sc=ax.scatter(x, y, s=50*7.2/mi, marker='*', facecolor='none', edgecolor='ivory', alpha=1, lw=0.75, zorder=1000, 
                      path_effects=[path_effects.Stroke(linewidth=2, foreground='k', alpha=1), path_effects.Normal()])
        if year == 1701:
            va='bottom'
            y = y+0.04
        else:
            va='top'
            y = y-0.04
        text=ax.text(x, y, str(year), va=va, ha='center', size=8, color='k', zorder=1001, clip_on=True,
                    path_effects=[path_effects.Stroke(linewidth=1.5, foreground='w', alpha=1), path_effects.Normal()])

        
def drawbeachball(ax, m, data):
    cat_gcmt = obspy.read_events(data)
    for event in cat_gcmt.events:
        mag = event.magnitudes[0].mag
        lon, lat, dep = event.origins[1].longitude, event.origins[1].latitude, event.origins[1].depth
        #if lon > lonmin+0.01 and lon < lonmax-0.01 and lat > latmin+0.01 and lat < latmax-0.01 and event.origins[0].time <= obspy.UTCDateTime('2021-11-14'):
        if event.origins[0].time <= obspy.UTCDateTime('2021-09-14'):
            focmecs=[event.focal_mechanisms[0].moment_tensor.tensor.m_rr,
                     event.focal_mechanisms[0].moment_tensor.tensor.m_tt,
                     event.focal_mechanisms[0].moment_tensor.tensor.m_pp,
                     event.focal_mechanisms[0].moment_tensor.tensor.m_rt,
                     event.focal_mechanisms[0].moment_tensor.tensor.m_rp,
                     event.focal_mechanisms[0].moment_tensor.tensor.m_tp]
            focmecs = convertUSEtoNED(focmecs)

            size = mag * 2
            if event.origins[0].time <= obspy.UTCDateTime('2021-08-14'):
                color='C7'
            else:
                color='C5'
            
            x, y = m(lon, lat)
            if event.resource_id == 'smi:service.iris.edu/fdsnws/event/1/query?eventid=11456117': # mainshock
                tmp = beachball.plot_beachball_mpl(focmecs, ax, size=size, position=(x, y),
                                         beachball_type='deviatoric', edgecolor='none', color_t='C5',
                                         color_p='w', linewidth=0.75, alpha=1, zorder=int(82-mag*10))
                tmp = beachball.plot_beachball_mpl(focmecs, ax, size=size, position=(x, y),
                                         beachball_type='dc', edgecolor='k', color_t='none',
                                         color_p='none', linewidth=0.75, alpha=1, zorder=int(82-mag*10))
            else:
                tmp = beachball.plot_beachball_mpl(focmecs, ax, size=size, position=(x, y),
                                         beachball_type='deviatoric', edgecolor='none', color_t=color,
                                         color_p='w', linewidth=0.75, alpha=1, zorder=int(82-mag*10))
                tmp = beachball.plot_beachball_mpl(focmecs, ax, size=size, position=(x, y),
                                         beachball_type='dc', edgecolor='k', color_t='none',
                                         color_p='none', linewidth=0.75, alpha=1, zorder=int(82-mag*10))

                
def drawactivefault(ax, m, zorder, data):
    sf = shapefile.Reader(data, encoding='ISO8859-1')
    fields = sf.fields[1:] 
    field_names = [field[0] for field in fields] 
    for r in sf.shapeRecords():  
        atr = dict(zip(field_names, r.record))
        if atr['catalog_na'] == 'GEM_Central_Am_Carib':# or 'Bird 2003':
            x = [i[0] for i in r.shape.points[:]]
            y = [i[1] for i in r.shape.points[:]]
            x, y = m(x, y)
            #if atr['slip_type'] == 'Reverse':
            rev = ax.plot(x, y, color='k', lw=1.2, zorder=zorder, alpha=1, linestyle='-', 
                         path_effects=[path_effects.Stroke(linewidth=2.5, foreground='w', alpha=1), path_effects.Normal()])
            
def drawactivefaultInset(ax, m, data):
    sf = shapefile.Reader(data, encoding='ISO8859-1')
    fields = sf.fields[1:] 
    field_names = [field[0] for field in fields] 
    for r in sf.shapeRecords():  
        atr = dict(zip(field_names, r.record))
        if atr['catalog_na'] == 'GEM_Central_Am_Carib':# or 'Bird 2003':
            x = [i[0] for i in r.shape.points[:]]
            y = [i[1] for i in r.shape.points[:]]
            x, y = m(x, y)
            #if atr['slip_type'] == 'Reverse':
            rev = ax.plot(x, y, color='k', lw=0.75, zorder=1, alpha=1, linestyle='-')

            
def drawinsetmap(fig, ax, m, elon, elat, lonminb, lonmaxb, latminb, latmaxb, data):
    axp = ax.get_position()
    tickintx, tickinty, tickformat = 10, 10, 0
    lonmin, lonmax, latmin, latmax = elon-10, elon+10, elat-4, elat+6
    m=Basemap(llcrnrlon=lonmin,llcrnrlat=latmin,urcrnrlon=lonmax,urcrnrlat=latmax,rsphere=(6378137.00,6356752.3142),resolution='i',projection='cyl')
    x, y=m([lonmin, lonmax], [latmin, latmax])
    aspect=abs(max(x)-min(x))/abs(max(y)-min(y))
    axpwidth = 0.35
    axpheight=axpwidth/aspect
    axpxloc, axpyloc = axp.x0+0.005, axp.y1-axpheight-0.005
    ax=fig.add_axes([axpxloc, axpyloc, axpwidth, axpheight])
    axp = axpA = ax.get_position()
    m.fillcontinents(color='k', zorder=0, alpha=0)
    m.drawcoastlines(color='k', linewidth=0.3, zorder=1)
    ax2 = mapTicksBasemap(fig,m,ax,tickintx,tickinty,lonmin,lonmax,latmin,latmax,tickformat)
    ax2.tick_params(axis='x',direction='in', pad=-10)
    ax2.tick_params(axis='y',direction='in', pad=-22)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)        
    drawactivefaultInset(ax, m, data)
    
    x, y = m(elon, elat)
    sc=ax.scatter(x, y, s=100, marker='*', facecolor='none', edgecolor='k', alpha=1, lw=1, zorder=100, 
                  path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=1), path_effects.Normal()])

    for x, y, text in zip([-75,-78,-70.9], [17.5,19.3,22.6], ['CA','GO','NA']):
        ax.text(x, y, text, fontsize=8, zorder=100, va='top',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=1), path_effects.Normal()])

    # MOVEL2010, CA(NA)
    #MORVEL 2010	16° N 16.000000°	73° 28' 30" W -73.475000°	19.92	73.52°	5.65	19.10	CA(NA)
    #MORVEL 2010	21° 30' N 21.500000°	71° W -71.000000°	19.29	252.77°	-5.71	-18.42	NA(CA)	
    plon, plat, azi, vel = [-73.475, -71], [16, 21.5], [73.52, 252.77], [19.92, 19.29]
    for i in range(len(plat)):
        g = geod.Direct(plat[i], plon[i], azi[i], vel[i]*12.5*1e3)
        spoint = m(plon[i], plat[i])
        epoint = m(g['lon2'], g['lat2'])
        an = ax.annotate('', xy=(epoint[0], epoint[1]), xytext=(spoint[0], spoint[1]),
                         arrowprops=dict(arrowstyle="simple,head_length=0.45,head_width=0.45,tail_width=0.15", edgecolor='w', facecolor='k', lw=0.2))
        if plat[i] == 16:
            va, ha='top', 'right'
            x, y = m(g['lon1'], g['lat1']+0.5)
        else:
            va, ha='top', 'left'        
            x, y = m(g['lon1'], g['lat1'])
        text=ax.text(x, y, '{:.0f}'.format(vel[i])+' mm/yr', va=va, ha=ha, size=8, color='k',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=0.75), path_effects.Normal()])


    x, y = m([lonminb, lonmaxb, lonmaxb, lonminb, lonminb],[latminb, latminb, latmaxb, latmaxb, latminb])
    #ax.plot(x, y, linewidth=1.5, color='C5', solid_joinstyle='miter', path_effects=[path_effects.Stroke(linewidth=2, foreground='k', alpha=1), path_effects.Normal()])
    ax.plot(x, y, linewidth=1.5, color='C5', solid_joinstyle='miter')
    ax.set_facecolor('w')
    
    
def drawtexts(ax, m, elat, elon, lonmin, lonmax, latmin, latmax):
    
    elat2010, elon2010 = 18.443, -72.571
    
    for x, y, text in zip([elon-0.3,elon2010+0.1,elon], [18.7,elat2010-0.1,elat-0.12],
                          ['2021-08-14 Mw 7.2 Haiti earthquake','2010-01-12 Mw 7.0 Haiti', 'Enriquillo-Plantain Garden Fault']):
        if y == 18.7:
            linewidth = 3
            fontsize = 10
        else:
            linewidth = 3
            fontsize = 8
        ax.text(x, y, text, fontsize=fontsize, zorder=5000, va='top', ha='center',
                path_effects=[path_effects.Stroke(linewidth=linewidth, foreground='w', alpha=1), path_effects.Normal()])

    x, y = m(-73.75, 18.2)
    ax.scatter(x, y, marker='s', facecolor='k', edgecolor='w', linewidth=1, s=25)
    ax.text(x, y-0.03, 'Les Cayes', fontsize=8, zorder=5000, va='top', ha='left',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=1), path_effects.Normal()])

    x, y = m(-72.33333, 18.53333)
    ax.scatter(x, y, marker='s', facecolor='k', edgecolor='w', linewidth=1, s=25, zorder=10)
    ax.text(x, y+0.03, 'Port-au-Prince', fontsize=8, zorder=1, va='bottom', ha='center',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=1), path_effects.Normal()])

    tmp = geod.Direct(latmin+0.15, lonmin+0.15, 90, 50*1e3)
    x0, y0 = m(tmp['lon1'], tmp['lat1'])
    x1, y1 = m(tmp['lon2'], tmp['lat2'])
    ax.plot([x0, x1], [y0, y1], lw=1, color='k',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=1), path_effects.Normal()])
    tmp = geod.Direct(latmin+0.15, lonmin+0.15, 90, 25*1e3)
    x0, y0 = m(tmp['lon2'], tmp['lat2'])
    ax.text(x0, y0-0.05, '50 km', fontsize=8, zorder=100, va='top', ha='center',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=1), path_effects.Normal()])


    tmp = geod.Direct(18.32, -74.8, 80, 20*1e3)
    x0, y0 = m(tmp['lon1'], tmp['lat1'])
    x1, y1 = m(tmp['lon2'], tmp['lat2'])
    tmp = geod.Direct(tmp['lat1'], tmp['lon1'], 80-20, 10*1e3)
    x2, y2 = m(tmp['lon2'], tmp['lat2'])
    ax.plot([x1, x0, x2], [y1, y0, y2], lw=1, color='k',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=1), path_effects.Normal()])

    tmp = geod.Direct(18.24, -74.8, 80, 20*1e3)
    x0, y0 = m(tmp['lon1'], tmp['lat1'])
    x1, y1 = m(tmp['lon2'], tmp['lat2'])
    tmp = geod.Direct(tmp['lat2'], tmp['lon2'], 80+180-20, 10*1e3)
    x2, y2 = m(tmp['lon2'], tmp['lat2'])
    ax.plot([x0, x1, x2], [y0, y1, y2], lw=1, color='k',
            path_effects=[path_effects.Stroke(linewidth=2, foreground='w', alpha=1), path_effects.Normal()])

    
def getFFMslipcell(slat, slon, dl, dk, stk, dip, m):
    shiftk = dk/2 * np.cos(np.deg2rad(dip))
    shiftl = dl/2

    tmp0 = geod.Direct(slat, slon, stk-90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], stk, shiftl*1e3)
    x, y = m(tmp1['lon2'], tmp1['lat2'])
    rbx, rby = x, y

    tmp0 = geod.Direct(slat, slon, stk-90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], stk-180, shiftl*1e3)
    x, y = m(tmp1['lon2'], tmp1['lat2'])
    rtx, rty = x, y

    tmp0 = geod.Direct(slat, slon, stk+90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], stk-180, shiftl*1e3)
    x, y = m(tmp1['lon2'], tmp1['lat2'])
    ltx, lty = x, y

    tmp0 = geod.Direct(slat, slon, stk+90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], stk, shiftl*1e3)
    x, y = m(tmp1['lon2'], tmp1['lat2'])
    lbx, lby = x, y

    xlist = [lbx, rbx, rtx, ltx, lbx]
    ylist = [lby, rby, rty, lty, lby]
    return xlist, ylist


def drawFFMmodeltop(dx, dy, nx, ny, x0, y0, model_dip, model_stk, elat, elon, m, ax, zorder):
    shiftk = ((ny-y0)*dy+dy/2) * np.cos(np.deg2rad(model_dip))
    tmp0 = geod.Direct(elat, elon, model_stk-90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], model_stk, ((nx-x0)*dx+dx/2)*1e3)
    x1, y1 = m(tmp1['lon1'], tmp1['lat1'])
    x2, y2 = m(tmp1['lon2'], tmp1['lat2'])
    ax.plot([x1, x2], [y1, y2], lw=1, color='k', solid_capstyle='projecting', zorder=zorder)

    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], model_stk, ((-x0)*dx+dx/2)*1e3)
    x1, y1 = m(tmp1['lon1'], tmp1['lat1'])
    x2, y2 = m(tmp1['lon2'], tmp1['lat2'])
    ax.plot([x1, x2], [y1, y2], lw=1, color='k', solid_capstyle='projecting', zorder=zorder)

    
def drawsliponmap(fig, ax, m, model, model_para, elon, elat):
    cmap = cm.bilbao
    dx, dy = model_para.xx[0], model_para.yy[0]
    model_stk, model_dip = model_para.strike[0], model_para.dip[0]
    nx, ny, x0, y0 = model_para.mn[0], model_para.nn[0], model_para.m0[0], model_para.n0[0]
    lon, lat, dep, slip = np.loadtxt('model_'+model+'/FFM_DCall.txt', unpack=True, usecols=(1,2,3,4), skiprows=1)
    tmp = np.argsort(slip)
    lon = [ lon[i] for i in tmp ]
    lat = [ lat[i] for i in tmp ]
    slip = [ slip[i] for i in tmp ]
    dep = [ dep[i] for i in tmp ]
    for i in range(len(lat)):
        xslipcell, yslipcell = getFFMslipcell(lat[i], lon[i], dx, dy, model_stk, model_dip, m)
        ax.fill(xslipcell, yslipcell, facecolor=cmap(slip[i]/max(slip)), zorder=1, edgecolor='C7', lw=0.5)
        
    drawFFMmodeltop(dx, dy, nx, ny, x0, y0, model_dip, model_stk, elat, elon, m, ax, 1)
    axp = ax.get_position()
    cax=fig.add_axes([axp.x1+0.005, axp.y1-axp.height*0.25, 0.01, axp.height*0.25])
    norm=mpl.colors.Normalize(vmin=0, vmax=max(slip))
    cb=mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, 
                                 ticks=np.linspace(0, max(slip), 3), format='%.2f')
    cb.ax.tick_params(labelsize=8)
    cb.set_label(label='Slip (m)', fontsize=8)
    
    
    
def drawslipbeachball(ax, m, model, model_para, elat, elon, scaleflag=0, fig=None):
    cmap = cm.bilbao
    dx, dy = model_para.xx[0], model_para.yy[0]
    model_stk, model_dip = model_para.strike[0], model_para.dip[0] # <-- this is just for the visualization purpose
    nx, ny, x0, y0 = model_para.mn[0], model_para.nn[0], model_para.m0[0], model_para.n0[0]
    lon, lat, dep, slip = np.loadtxt('../materials/ffm/model_'+model+'/FFM_DCall.txt', unpack=True, usecols=(1,2,3,4), skiprows=1)
    tmp = np.argsort(slip)
    lon = [ lon[i] for i in tmp ]
    lat = [ lat[i] for i in tmp ]
    slip = [ slip[i] for i in tmp ]
    dep = [ dep[i] for i in tmp ]
    for i in range(len(lat)):
        xslipcell, yslipcell = getSlipCell(lat[i], lon[i], dx, dy, model_stk, model_dip, m)
        ax.fill(xslipcell, yslipcell, facecolor='none', zorder=1, edgecolor='gray', lw=0.5)

    plotModelTop(dx, dy, nx, ny, x0, y0, model_dip, model_stk, elat, elon, m, ax, 10)    

    data = np.loadtxt(os.path.join('../materials/ffm/model_'+model, 'FFM_DCall.txt'), skiprows=1)
    lon,lat,dep,slip = data[:,1],data[:,2],data[:,3],data[:,4]
    strike0,dip0,rake0,strike1,dip1,rake1 = data[:,5],data[:,6],data[:,7],data[:,8],data[:,9],data[:,10]
    xloc,yloc = data[:,11],data[:,12]

    data=np.loadtxt(os.path.join('../materials/ffm/model_'+model, 'FFM_MT.txt'))
    m1,m2,m3,m4,m5,m6 = data[:,4],data[:,5],data[:,6],data[:,7],data[:,8],data[:,9]

    x, y = m(lon, lat)
    for i in range(len(slip)):

        focmec = [m1[i],m2[i],m3[i],m4[i],m5[i],m6[i]]
        focmec = convertUSEtoNED(focmec)
        tmp = beachball.plot_beachball_mpl(focmec, ax, size=dx*0.75, position=(x[i], y[i]),
                                 beachball_type='deviatoric', edgecolor='none', color_t=cmap(slip[i]/max(slip)),
                                 color_p='w', linewidth=0.75, alpha=1, zorder=11+int(slip[i]/max(slip)*100), view='top')
        tmp = beachball.plot_beachball_mpl(focmec, ax, size=dx*0.75, position=(x[i], y[i]),
                                 beachball_type='dc', edgecolor='k', color_t='none',
                                 color_p='none', linewidth=0.75, alpha=1, zorder=11+int(slip[i]/max(slip)*100), view='top')
        
     
    if scaleflag == 1:
        axp = ax.get_position()
        cax=fig.add_axes([axp.x1+0.005, axp.y0, 0.01, axp.height*0.5])
        norm=mpl.colors.Normalize(vmin=0, vmax=max(slip))
        cb=mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, 
                                     ticks=np.linspace(0, max(slip), 3), format='%.2f')
        cb.set_label(label='Slip (m)')
    
        

        
def getSlipCell(slat, slon, dl, dk, stk, dip, m):
    shiftk = dk/2 * np.cos(np.deg2rad(dip))
    shiftl = dl/2

    tmp0 = geod.Direct(slat, slon, stk-90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], stk, shiftl*1e3)
    x, y = m(tmp1['lon2'], tmp1['lat2'])
    rbx, rby = x, y

    tmp0 = geod.Direct(slat, slon, stk-90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], stk-180, shiftl*1e3)
    x, y = m(tmp1['lon2'], tmp1['lat2'])
    rtx, rty = x, y

    tmp0 = geod.Direct(slat, slon, stk+90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], stk-180, shiftl*1e3)
    x, y = m(tmp1['lon2'], tmp1['lat2'])
    ltx, lty = x, y

    tmp0 = geod.Direct(slat, slon, stk+90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], stk, shiftl*1e3)
    x, y = m(tmp1['lon2'], tmp1['lat2'])
    lbx, lby = x, y

    xlist = [lbx, rbx, rtx, ltx, lbx]
    ylist = [lby, rby, rty, lty, lby]
    return xlist, ylist


def plotModelTop(dx, dy, nx, ny, x0, y0, model_dip, model_stk, elat, elon, m, ax, zorder):
    shiftk = ((ny-y0)*dy+dy/2) * np.cos(np.deg2rad(model_dip))
    tmp0 = geod.Direct(elat, elon, model_stk-90, shiftk*1e3)
    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], model_stk, ((nx-x0)*dx+dx/2)*1e3)
    x1, y1 = m(tmp1['lon1'], tmp1['lat1'])
    x2, y2 = m(tmp1['lon2'], tmp1['lat2'])
    ax.plot([x1, x2], [y1, y2], lw=1, color='k', solid_capstyle='butt', zorder=zorder+1)

    tmp1 = geod.Direct(tmp0['lat2'], tmp0['lon2'], model_stk, ((-x0)*dx+dx/2)*1e3)
    x1, y1 = m(tmp1['lon1'], tmp1['lat1'])
    x2, y2 = m(tmp1['lon2'], tmp1['lat2'])
    ax.plot([x1, x2], [y1, y2], lw=1, color='k', solid_capstyle='butt', zorder=zorder+1)

    
class TwoInnerPointsNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
        self.low = low
        self.up = up
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.25, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
def drawtopobathy(fig, ax, m, data, scaleflag=0, elevflag='gebco'):
    
    colors1 = cm.oslo(np.linspace(0, 1, 128))
    colors2 = cm.grayC(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)    
    
    if elevflag == 'srtm':
        fh = Dataset(data, mode='r')
        lons = fh.variables['lon'][:]; lats = fh.variables['lat'][:]; tmax = fh.variables['z'][:]
        fh.close()
    else:
        fh = Dataset(data, mode='r')
        lons = fh.variables['lon'][:]; lats = fh.variables['lat'][:]; tmax = fh.variables['elevation'][:]
        fh.close()
        
    ls = LightSource(azdeg=135, altdeg=15)
    
    norm = TwoInnerPointsNormalize(vmin=-6000, vmax=3000, low=-3000, up=0)    
    rgb = ls.shade(tmax, cmap=mymap, norm=norm)
    im = ax.imshow(rgb, origin='lower',alpha=1, zorder=-1, interpolation='gaussian')
    x0 = m(np.min(lons), np.min(lats))
    x1 = m(np.max(lons), np.max(lats))
    im.set_extent([x0[0], x1[0], x0[1], x1[1]])

    cf = ax.contourf(lons, lats, tmax, levels=np.linspace(-100000, 0, 21), cmap=cm.grayC_r, alpha=0.95, antialiased=True, zorder=0)
    
def drawdetailtopography(fig, ax, m, data):
    
    fh = Dataset(data, mode='r')
    lons = fh.variables['lon'][:]; lats = fh.variables['lat'][:]; tmax = fh.variables['z'][:]
    #lons = fh.variables['lon'][:]; lats = fh.variables['lat'][:]; tmax = fh.variables['elevation'][:]
    fh.close()
    vmin, vmax = 0, 3000
    ls = LightSource(azdeg=135, altdeg=15)
    rgb = ls.shade(tmax, cmap=cm.grayC, vmin=vmin, vmax=vmax)
    im = ax.imshow(rgb, origin='lower',alpha=1, zorder=-1, interpolation='gaussian')
    x0 = m(np.min(lons), np.min(lats))
    x1 = m(np.max(lons), np.max(lats))
    im.set_extent([x0[0], x1[0], x0[1], x1[1]])
    
    axp = ax.get_position()
    cax=fig.add_axes([axp.x1+0.005, axp.y1-axp.height*0.4, 0.01, axp.height*0.4])
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb=mpl.colorbar.ColorbarBase(cax, cmap=cm.grayC, norm=norm, ticks=np.linspace(vmin, vmax, 3), format='%.0f')
    cb.set_label(label='Topography (m)', fontsize=8)
    cb.ax.tick_params(labelsize=8)
    
    
    
# select preferred fault plane based on the model plane geometry
def selectplane(modelstk, modeldip, stk0, dip0, rake0, stk1, dip1, rake1):
    vecmodelplane = faultnormalvec(modelstk, modeldip)
    vecplane0 = faultnormalvec(stk0, dip0)
    vecplane1 = faultnormalvec(stk1, dip1)
    tmp0 = np.inner(vecmodelplane, vecplane0)
    tmp1 = np.inner(vecmodelplane, vecplane1)
    if abs(tmp0) > abs(tmp1):
        stk_s = stk0
        dip_s = dip0
        rake_s = rake0
    elif abs(tmp0) < abs(tmp1):
        stk_s = stk1
        dip_s = dip1
        rake_s = rake1
    else:
        stk_s = stk0
        dip_s = dip0
        rake_s = rake0
    return stk_s, dip_s, rake_s

def faultnormalvec(stk, dip):
    nn = -np.sin(np.deg2rad(stk)) * np.sin(np.deg2rad(dip))
    ne =  np.cos(np.deg2rad(stk)) * np.sin(np.deg2rad(dip))
    nd = -np.cos(np.deg2rad(dip))
    return np.array([ne, nn, nd])


def plotTimeEvo(fig, modelid, xloc0, yloc0, axpw, axph):
    data=np.loadtxt('../materials/ffm/model_'+str(modelid)+'/slip-rate-time_along_strike.txt', skiprows=1)
    t, x, amp = data[:,0], data[:,1], data[:,3]
    ys=np.linspace(min(x), max(x), 1000)
    xs=np.linspace(min(t), max(t), 1000)
    X, Y = np.meshgrid(xs, ys)
    Z = griddata((t, x), amp, (X, Y),'linear')
    levels = np.linspace(0, max(amp), 21)
    ax=fig.add_axes([xloc0, yloc0, axpw, axph])
    axp=ax.get_position()
    
    cmap = cm.bilbao
    sc = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, vmin=0, vmax=max(amp))
    
    for pm in [-1, 1]:
        for vr in [3, 4, 5]:
            for xloc in [np.max(t), np.min(t)]:
                x0, x1 = 0, xloc
                y0, y1 = 0, abs((x1-x0)*vr) * pm
                ax.plot([x0, x1], [y0, y1], color='k', lw=0.3, alpha=1, linestyle='--', zorder=1)
            xloc = 10
            if vr == 3:
                text=ax.text(xloc, xloc*vr*pm, str(vr)+' km/s', alpha=1, size=8, 
                             color='k', ha='left', va='center', zorder=1)     
            else:
                text=ax.text(xloc, xloc*vr*pm, str(vr), alpha=1, size=8, 
                             color='k', ha='left', va='center', zorder=1)     
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='w', alpha=0.75), path_effects.Normal()])

    ax.set_ylabel('Distance (km)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim(min(x), max(x))
    ax.set_xlim(0, max(t))
    axp=ax.get_position()

    cax = fig.add_axes([axp.x1-0.12, axp.y0+0.05, 0.01, 0.1])
    norm=mpl.colors.Normalize(vmin=0, vmax=max(amp))
    cb=mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, label='Slip rate (m/s)', 
                                 ticks=np.linspace(0, max(amp), 3), format='%.2f')
    
    return ax


def plotTimeEvoDip(fig, modelid, xloc0, yloc0, axpw, axph):
    data=np.loadtxt('../materials/ffm/model_'+str(modelid)+'/slip-rate-time_along_dip.txt', skiprows=1)
    t, x, amp = data[:,0], data[:,1], data[:,3]

    model_para = load_fort40(os.path.join('../materials/ffm', 'model_'+str(modelid), 'fort.40'))
    x = model_para.depth[0]-np.sin(np.deg2rad(model_para.dip[0]))*x
    
    ys=np.linspace(min(x), max(x), 1000)
    xs=np.linspace(min(t), max(t), 1000)
    X, Y = np.meshgrid(xs, ys)
    Z = griddata((t, x), amp, (X, Y),'linear')
    levels = np.linspace(0, max(amp), 21)
    ax=fig.add_axes([xloc0, yloc0, axpw, axph])
    axp=ax.get_position()
    
    cmap = cm.bilbao
    sc = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, vmin=0, vmax=max(amp))
    
    ax.set_ylabel('Depth (km)', labelpad=9.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylim(min(x), max(x))
    ax.set_xlim(0, max(t))
    axp=ax.get_position()
    ax.invert_yaxis()
    
    return ax


def project_on_ref_fault(elat, elon, slat, slon, ref_azi):
    tmp = geod.Inverse(elat, elon, slat, slon)
    theta = ref_azi-tmp['azi1']
    actual_dis = tmp['s12']*1e-3
    proj_dis = tmp['s12']*1e-3 * np.cos(np.deg2rad(theta))
    return actual_dis, proj_dis, theta


def strikeevo(model, ax, refazi):
    model_para = load_fort40(os.path.join('../materials/ffm', 'model_'+model, 'fort.40'))
    maxsliprateall = np.loadtxt(os.path.join('../materials/ffm', 'model_'+model, 'snap_1sec_meca_201.txt'), usecols=8, skiprows=1)[0]

    data = np.loadtxt(os.path.join('../materials/ffm', 'model_'+model, 'tw_mec.dat'))
    m1, m2, m3, m4, m5, m6 = data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8]
    snap_t, strike0, dip0, rake0 = data[:,0], data[:,9], data[:,11], data[:,13]
    strike1, dip1, rake1 = data[:,10], data[:,12], data[:,14]
    snap_t = snap_t-0.5 # center of time window

    strike, dip, rake = [],[],[]
    for i in range(len(strike0)):
        if snap_t[i] <= 10:
            tmpstr, tmpdip, tmprake = selectplane(model_para.strike[0],model_para.dip[0],strike0[i],dip0[i],rake0[i],strike1[i],dip1[i],rake1[i])
        else:
            tmpstr, tmpdip, tmprake = selectplane(223,90,strike0[i],dip0[i],rake0[i],strike1[i],dip1[i],rake1[i])

        if tmpstr < refazi - 180:
            tmpstr = (tmpstr + 180)%360
        if tmpstr > refazi - 180 and tmpstr <= refazi - 90:
            tmpstr = tmpstr + 180            

        strike.append(tmpstr)
        dip.append(tmpdip)
        rake.append(tmprake)


    maxsliprate_timewindow = []
    for snap in np.arange(0, 32, 1):
        data=np.loadtxt(os.path.join('../materials/ffm', 'model_'+model, 'snap_1sec_meca_'+str(200+snap+1)+'.txt'), skiprows=1)
        xloc, yloc, dep, avesliprate, maxsliprate=data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
        maxsliprate_timewindow.append(max(avesliprate))
    maxsliprate_timewindow = np.array(maxsliprate_timewindow)


    for pm in [-45, 0, 45]:
        ax.axhline(refazi+pm, linestyle='--', lw=0.5, zorder=0, color='k')

    cmap = cm.bilbao
    vmin=0
    vmax=maxsliprateall
    for i in range(len(snap_t)):
        #size = np.exp(maxsliprate_timewindow[i]*20)*10
        size = maxsliprate_timewindow[i]*250
        ax.scatter(snap_t[i], strike[i], s=size,
                   facecolor=cmap(maxsliprate_timewindow[i]/maxsliprateall), edgecolor='none', lw=0.75, zorder=int(100*maxsliprate_timewindow[i]/maxsliprateall))
        if maxsliprate_timewindow[i] > 0.5*maxsliprateall:
            ax.scatter(snap_t[i], strike[i], s=size,
                       facecolor='none', edgecolor='k', lw=0.75, zorder=int(100*maxsliprate_timewindow[i]/maxsliprateall))
    ax.set(ylim=[268-90, 268+90], xlim=[0, 31], yticks=[268-45, 268, 268+45], xlabel='Time (s)', ylabel='Strike ($\degree$)')
