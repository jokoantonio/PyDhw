# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:49:48 2023

@author: Joko Prihantono
        Research Center for Conservation of Marine and Inland Water Resources
        National Research and Innovation Agency, Indonesia
"""
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from matplotlib.ticker import MultipleLocator
import pandas as pd
import ee

ee.Initialize()

def ee_array_to_df(arr, list_of_bands):
    """--------------------------------------------------------------
    Transforms client-side ee.Image.getRegion array to pandas.
    DataFrame.
    
    input: -arr: list contain longitude, latitude, timestamp, sst band
           -list_of_band: sst band from the gee dataset
    output:-time
           -datetime
           -sst band
    -----------------------------------------------------------------"""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df


def avg_sst_day(dfile,list_of_bands):
    """-------------------------------------------------------------
    Calculate hourly sst to daily max sst. 
    
    Input: - Pandas dataframe: datetime and sst

    output: - date
            - sst
    ----------------------------------------------------------------"""
    dfile['year']  = pd.DatetimeIndex(dfile['datetime']).year
    dfile['month'] = pd.DatetimeIndex(dfile['datetime']).month
    dfile['day']   = pd.DatetimeIndex(dfile['datetime']).day

    ##grouping
    df  = dfile.groupby(['year','month','day'], as_index=False)[list_of_bands].max()

    sst = df[list_of_bands].to_numpy()   
    tgl = np.array([date(df.year[x],df.month[x],df.day[x]) 
                        for x in range(len(df.year))])
    
    return tgl,sst


def dhw_calc(tgl,sst):
    """--------------------------------------------------------------
    Calculate Maximum of the Monthly Mean (MMM) SST and Hot Spot (HS).
    
    Input : - Date in the format datetime
            - SST data
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW)
    -----------------------------------------------------------------"""
    bulan = np.array([tgl[i].month for i in range(len(tgl))])
    
    month = []
    MM    = []
    for i in range(12):
        # print(i+1)
        month.append(i+1)
        MM.append(np.mean(sst[bulan==i+1]))
    
    month = np.array(month)
    MM    = np.array(MM)
    
    ## Plot MM
    # plt.plot(month,MM)
    
    ##===== Max Monthly Mean =========
    MMM = max(MM)
    
    ##========= Hot Spot ===========
    HS = sst.data-MMM
    HS[HS<0]=0
        
    ##========= DHW ==============
    DHW = []
    HSm = HS-1
    for i in range(len(HS)):
        imin = np.where((i-84)<=0,0,(i-84))   
        a = np.where((HSm[imin:i+1])<0,0,HSm[imin:i+1])
        DHW.append(a.sum()/7)
        # a    = np.sum(HSm[imin:i+1])/7
        # DHW.append(a)
        
        # print(DHW)
    DHW = np.array(DHW)
    
    return MM,MMM,HS,DHW

def plot_dhw_warning(df):
    """---------------------------------------------------------------
    Plot SST, MM, MMM, HS, DHW and the warning. It is similar with graphic
    from NOAA.
    
    input : -date, SST, MM, MMM, HS, DWH
    Output: - Graph
    -----------------------------------------------------------------"""
    fig,ax1 = plt.subplots(figsize=(20,8))

    ax1.plot(df['tgl'],df['sst'],'k-', label='SST')
    grp2 = ax1.plot(df['tgl'],df['MMM']+1,'b-', label='Bleaching Threshold SST')
    grp3 = ax1.plot(df['tgl'],df['MMM'],'b--', label='Max Monthly Mean SST')
    grp4 = ax1.plot(df['tgl'][pd.DatetimeIndex(df['tgl']).day==15],
              df['MM'][pd.DatetimeIndex(df['tgl']).day==15],
              'b+',markersize=10, label='Monthly Mean Climatology')

    ax1.set_ylabel('SST ($^\circ$C)',fontsize=16) 
    ax1.set_xlim([min(df['tgl'])],max(df['tgl']))
    ax1.set_ylim([15,35])
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.tick_params(axis='y',labelsize=16, which='both', width=2) ## Tick font size
    ax1.tick_params(axis='x',labelsize=16, #labelrotation=30,
                    width=2, length=8) ## Tick font size and rotation
    ax1.tick_params(axis='y',which='major',length=8)
    ax1.tick_params(axis='y',which='minor',length=4)
    # date_format = mdates.DateFormatter('%d-%m-%Y') ## Format date x axis
    # ax1.xaxis.set_major_formatter(date_format)     ## Format date x axis

    trsh1 = np.ones(len(df['sst']))*4 ## line for treshold 4
    trsh2 = np.ones(len(df['sst']))*8 ## line for treshold 8  

    ax2 = ax1.twinx()
    ax2.plot(df['tgl'],df['DHW'],color='orangered')
    ax2.plot(df['tgl'],trsh1,'--',color='orangered')
    grp5 = ax2.plot(df['tgl'],trsh2,'--',color='orangered', label='4, 8 DHWs')

    ax2.set_ylabel('DHW ($^\circ$C)',color='orangered',fontsize=16)
    ax2.set_ylim([0,25])
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.tick_params(axis='y',labelcolor='orangered',labelsize=16, 
                    which='both', width=2)
    ax2.tick_params(axis='y', which='major', length=8,color='orangered')
    ax2.tick_params(axis='y', which='minor', length=4,color='orangered')

    grph = grp2+grp3+grp4+grp5
    grphc = [gr.get_label() for gr in grph]
    ax1.legend(grph, grphc, loc='upper center', ncol=5, bbox_to_anchor=(0.5,1.1),
                fontsize=16)
    # ax1.legend()

    ###==================================================================
    ###====================== fill between HS ===========================
    ###==================================================================
    ghs0 = ax2.fill_between(df['tgl'],df['DHW'], where=df['HS']<=0, 
                      facecolor='lightskyblue', edgecolor='black', 
                      label='No Stress')
    ghs1 = ax2.fill_between(df['tgl'],df['DHW'], where=(df['HS']>0) & (df['HS']<1), 
                      facecolor='yellow', edgecolor='black', 
                      label='Bleaching Watch')
    ghs2 = ax2.fill_between(df['tgl'],df['DHW'], where=(df['HS']>=1) & (df['DHW']>0) & (df['DHW']<4), 
                      facecolor='orange', edgecolor='black', 
                      label='Bleaching Warning')
    ghs3 = ax2.fill_between(df['tgl'],df['DHW'], where=(df['HS']>=1) & (df['DHW']>=4) & (df['DHW']<8), 
                      facecolor='red', edgecolor='black', 
                      label='Bleaching Alert Level 1')
    ghs4 = ax2.fill_between(df['tgl'],df['DHW'], where=(df['HS']>=1) & (df['DHW']>=8), 
                      facecolor='darkred', edgecolor='black', 
                      label='Bleaching Alert Level 2')

    ghs = [ghs0,ghs1,ghs2,ghs3,ghs4]
    grphc = [gr.get_label() for gr in ghs]
    ax2.legend(ghs, grphc, loc='lower center', ncol=5, bbox_to_anchor=(0.5,-0.15),
                fontsize=16) 

    ###==================================================================
    ###====================== Plot Box ==================================
    ###==================================================================
    def dhw_warning_box_zone(ihs,fcolor):
        if ihs.size != 0:
            idk  = np.where(np.diff(ihs)>1)
            ihss = np.append(ihs[0],ihs[idk]+np.diff(ihs)[idk])
            ihse = np.append(ihs[idk],ihs[-1])+1
            ihse0= np.where(ihse[-1]>len(df['tgl'])-1,
                              np.append(ihs[idk]+1,len(df['tgl'])-1),
                              ihse)
            
            for idhs in range(len(ihse0)):
                ax2.axvspan(df['tgl'][ihss[idhs]],df['tgl'][ihse0[idhs]],ymin=0,ymax=1/25,
                            facecolor=fcolor, edgecolor='black')
        

    ###======================== No Stress =========================
    ihs0  =df[df['HS']<=0].index
    dhw_warning_box_zone(ihs0,'lightskyblue')

    # ##==================== Bleaching Watch =========================
    ihs1  = df[(df['HS']>0) & (df['HS']<1)].index
    dhw_warning_box_zone(ihs1,'yellow')

    # ##==================== Bleaching Warning =========================
    ihs2  = df[(df['HS']>=1) & (df['DHW']>0) & (df['DHW']<4)].index
    dhw_warning_box_zone(ihs2,'orange')

    # ##==================== Bleacihing Alert Level 1 =========================
    ihs3  = df[(df['HS']>=1) & (df['DHW']>=4) & (df['DHW']<8)].index
    dhw_warning_box_zone(ihs3,'red')

    # ##==================== Bleacihing Alert Level 2 =========================
    ihs4  = df[(df['HS']>=1) & (df['DHW']>=8)].index
    dhw_warning_box_zone(ihs4,'darkred')
    

def dhw_hycom_gee(lat,lon,sdate,edate,scale):
    """---------------------------------------------------------------
    Load and extract SST data from HYCOM GEE dataset in specific latitude,
    longitude, radius, and time range.
    
    input : -latitude, longitude
            -start time, end time
            -scale
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW) 
    -----------------------------------------------------------------"""    
    # Import the HYCOM SST collection.
    dataset = ee.ImageCollection('HYCOM/sea_temp_salinity')

    # Selection of appropriate bands and dataset for SST
    dataset = dataset.select('water_temp_0').filterDate(sdate,edate)

    # Define the location of interst as point at the sea
    poi = ee.Geometry.Point(lon,lat)

    # Get the data for the pixel intersecting the point.
    sst_poi = dataset.getRegion(poi, scale).getInfo()
    
    df_sst = ee_array_to_df(sst_poi,['water_temp_0'])
    
    # convert unit of sst hycom to deg C.
    df_sst['water_temp_0'] = df_sst['water_temp_0']*0.001+20
    
    tgl,sst = avg_sst_day(df_sst,['water_temp_0'])
    
    MM,MMM,HS,DHW    = dhw_calc(tgl,sst) # call MM, MMM, HS, DHW
    
    ## save data to csv
    MMMsv  = MMM*np.ones(len(DHW))
    latsv  = lat*np.ones(len(DHW))
    lonsv  = lon*np.ones(len(DHW))
    df_asc = pd.DataFrame({'tgl':np.ravel(tgl),'lat':latsv,'lon':lonsv,
                           'sst':np.ravel(sst),'MMM':MMMsv,'HS':np.ravel(HS),
                           'DHW':np.ravel(DHW)})
    df_asc['MM']=df_asc['tgl'].apply(lambda x: MM[x.month-1])
    df_asc['blch_trshd']=df_asc['MMM']+1
    
    conditions = [
        (df_asc['HS']<=0),
        (df_asc['HS']>0)  & (df_asc['HS']<1),
        (df_asc['HS']>=1) & (df_asc['DHW']>0) & (df_asc['DHW']<4),
        (df_asc['HS']>=1) & (df_asc['DHW']>=4) & (df_asc['DHW']<8),
        (df_asc['HS']>=1) & (df_asc['DHW']>=8)
        ]

    results = [
        'No Stress',
        'Bleaching Watch',
        'Bleaching Warning',
        'Bleaching Alert Level 1',
        'Bleaching Alert Level 2'
        ]    

    df_asc['Alert'] = np.select(conditions, results)
    df_asc.to_csv('Hycom_dhw.csv',index=False)
    
    plot_dhw_warning(df_asc) #Plot DHW
    print("DHW data has been saved as: 'Hycom_dhw.csv'")

def dhw_noaa_oisst_gee(lat,lon,sdate,edate,scale):
    """---------------------------------------------------------------
    Load and extract SST data from NOAA OISST GEE dataset in specific latitude,
    longitude, radius, and time range.
    
    input : -latitude, longitude
            -start time, end time
            -scale
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW) 
    -----------------------------------------------------------------"""    
    # Import the NOAA OISST collection.
    dataset = ee.ImageCollection('NOAA/CDR/OISST/V2_1')

    # Selection of appropriate bands and dataset for SST
    dataset = dataset.select('sst').filterDate(sdate,edate)

    # Define the location of interst as point at the sea
    poi = ee.Geometry.Point(lon,lat)

    # Get the data for the pixel intersecting the point.
    sst_poi = dataset.getRegion(poi, scale).getInfo()
    
    df_sst = ee_array_to_df(sst_poi,['sst'])
    
    # convert unit of sst hycom to deg C.
    df_sst['sst'] = df_sst['sst']*0.01
    
    tgl,sst = avg_sst_day(df_sst,['sst'])
    
    MM,MMM,HS,DHW    = dhw_calc(tgl,sst) 
    
    ## save data to csv
    MMMsv  = MMM*np.ones(len(DHW))
    latsv  = lat*np.ones(len(DHW))
    lonsv  = lon*np.ones(len(DHW))
    df_asc = pd.DataFrame({'tgl':np.ravel(tgl),'lat':latsv,'lon':lonsv,
                           'sst':np.ravel(sst),'MMM':MMMsv,'HS':np.ravel(HS),
                           'DHW':np.ravel(DHW)})
    df_asc['MM']=df_asc['tgl'].apply(lambda x: MM[x.month-1])
    df_asc['blch_trshd']=df_asc['MMM']+1
    
    conditions = [
        (df_asc['HS']<=0),
        (df_asc['HS']>0)  & (df_asc['HS']<1),
        (df_asc['HS']>=1) & (df_asc['DHW']>0) & (df_asc['DHW']<4),
        (df_asc['HS']>=1) & (df_asc['DHW']>=4) & (df_asc['DHW']<8),
        (df_asc['HS']>=1) & (df_asc['DHW']>=8)
        ]

    results = [
        'No Stress',
        'Bleaching Watch',
        'Bleaching Warning',
        'Bleaching Alert Level 1',
        'Bleaching Alert Level 2'
        ]    

    df_asc['Alert'] = np.select(conditions, results)
    df_asc.to_csv('Noaa_oisst_dhw.csv',index=False)
    
    plot_dhw_warning(df_asc) #Plot DHW
    print("DHW data has been saved as: 'Noaa_oisst_dhw.csv'")
    

def dhw_modis_gee(lat,lon,sdate,edate,scale):
    """---------------------------------------------------------------
    Load and extract SST data from MODIS GEE dataset in specific latitude,
    longitude, radius, and time range.
    
    input : -latitude, longitude
            -start time, end time
            -scale
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW) 
    -----------------------------------------------------------------"""    
    # Import the MODIS SST collection.
    dataset = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI')

    # Selection of appropriate bands and dataset for SST
    dataset = dataset.select('sst').filterDate(sdate,edate)

    # Define the location of interst as point at the sea
    poi = ee.Geometry.Point(lon,lat)

    # Get the data for the pixel intersecting the point.
    sst_poi = dataset.getRegion(poi, scale).getInfo()
    
    df_sst = ee_array_to_df(sst_poi,['sst'])
    
    # convert unit of sst hycom to deg C.
    # df_sst['sst'] = df_sst['sst']
    
    tgl,sst = avg_sst_day(df_sst,['sst'])
    
    MM,MMM,HS,DHW    = dhw_calc(tgl,sst) 
    
    ## save data to csv
    MMMsv  = MMM*np.ones(len(DHW))
    latsv  = lat*np.ones(len(DHW))
    lonsv  = lon*np.ones(len(DHW))
    df_asc = pd.DataFrame({'tgl':np.ravel(tgl),'lat':latsv,'lon':lonsv,
                           'sst':np.ravel(sst),'MMM':MMMsv,'HS':np.ravel(HS),
                           'DHW':np.ravel(DHW)})
    df_asc['MM']=df_asc['tgl'].apply(lambda x: MM[x.month-1])
    df_asc['blch_trshd']=df_asc['MMM']+1
    
    conditions = [
        (df_asc['HS']<=0),
        (df_asc['HS']>0)  & (df_asc['HS']<1),
        (df_asc['HS']>=1) & (df_asc['DHW']>0) & (df_asc['DHW']<4),
        (df_asc['HS']>=1) & (df_asc['DHW']>=4) & (df_asc['DHW']<8),
        (df_asc['HS']>=1) & (df_asc['DHW']>=8)
        ]

    results = [
        'No Stress',
        'Bleaching Watch',
        'Bleaching Warning',
        'Bleaching Alert Level 1',
        'Bleaching Alert Level 2'
        ]    

    df_asc['Alert'] = np.select(conditions, results)
    df_asc.to_csv('Modis_dhw.csv',index=False)
    
    plot_dhw_warning(df_asc) #Plot DHW
    print("DHW data has been saved as: 'Modis_dhw.csv'")


def dhw_noaa_pathfinder_gee(lat,lon,sdate,edate,scale):
    """---------------------------------------------------------------
    Load and extract SST data from NOAA PATHFINDER GEE dataset in specific latitude,
    longitude, radius, and time range.
    
    input : -latitude, longitude
            -start time, end time
            -scale
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW) 
    -----------------------------------------------------------------"""    
    # Import the NOAA PATHFINDER SST collection.
    dataset = ee.ImageCollection('NOAA/CDR/SST_PATHFINDER/V53')

    # Selection of appropriate bands and dataset for SST
    dataset = dataset.select('sea_surface_temperature').filterDate(sdate,edate)

    # Define the location of interst as point at the sea
    poi = ee.Geometry.Point(lon,lat)

    # Get the data for the pixel intersecting the point.
    sst_poi = dataset.getRegion(poi, scale).getInfo()
    
    df_sst = ee_array_to_df(sst_poi,['sea_surface_temperature'])
    
    # convert unit of sst hycom to deg C.
    df_sst['sea_surface_temperature'] = df_sst['sea_surface_temperature']*0.01
    
    tgl,sst = avg_sst_day(df_sst,['sea_surface_temperature'])
    
    #remove data <=0
    a=np.where(sst<=0)
    sst = np.delete(sst,a)
    tgl = np.delete(tgl,a)
    
    MM,MMM,HS,DHW    = dhw_calc(tgl,sst) 
    
    ## save data to csv
    MMMsv  = MMM*np.ones(len(DHW))
    latsv  = lat*np.ones(len(DHW))
    lonsv  = lon*np.ones(len(DHW))
    df_asc = pd.DataFrame({'tgl':np.ravel(tgl),'lat':latsv,'lon':lonsv,
                           'sst':np.ravel(sst),'MMM':MMMsv,'HS':np.ravel(HS),
                           'DHW':np.ravel(DHW)})
    df_asc['MM']=df_asc['tgl'].apply(lambda x: MM[x.month-1])
    df_asc['blch_trshd']=df_asc['MMM']+1
    
    conditions = [
        (df_asc['HS']<=0),
        (df_asc['HS']>0)  & (df_asc['HS']<1),
        (df_asc['HS']>=1) & (df_asc['DHW']>0) & (df_asc['DHW']<4),
        (df_asc['HS']>=1) & (df_asc['DHW']>=4) & (df_asc['DHW']<8),
        (df_asc['HS']>=1) & (df_asc['DHW']>=8)
        ]

    results = [
        'No Stress',
        'Bleaching Watch',
        'Bleaching Warning',
        'Bleaching Alert Level 1',
        'Bleaching Alert Level 2'
        ]    

    df_asc['Alert'] = np.select(conditions, results)
    df_asc.to_csv('Noaa_pf_dhw.csv',index=False)
    
    plot_dhw_warning(df_asc) #Plot DHW
    print("DHW data has been saved as: 'Noaa_pf_dhw.csv'")
    

###===========================================================###
sdate = '2010-01-01' # Initial date of interest (inclusive).
edate = '2020-12-31' # Final date of interest (exclusive).
lon   = 119.670127  # longitude
lat   =  -8.491431   # lattitude
scale = 1  # scale in meters

# dhw_hycom_gee(lat,lon,sdate,edate,scale)
# dhw_noaa_oisst_gee(lat,lon,sdate,edate,scale)
# dhw_noaa_pathfinder_gee(lat,lon,sdate,edate,scale)
dhw_modis_gee(lat,lon,sdate,edate,scale)
