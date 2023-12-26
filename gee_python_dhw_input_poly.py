# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:23:54 2023

@author: jprih
"""
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import ee

ee.Initialize()


"""================================================================== 
Please find the input of the calculation at the bottom of this script
=================================================================="""

def List2SSTDay(lista):
    """--------------------------------------------------------------
    Transforms list of the average SST inside the polygon to pandas 
    DataFrame and Calculate hourly sst to daily max sst. 
    
    input: -time
           -SST from the GEE dataset
    output:-date
           -dily max SST (unconverted to degC)  
    -----------------------------------------------------------------"""
    df = pd.DataFrame(lista,columns=['time','SST'])
    # Remove rows without data inside.
    df = df[['time', 'SST']].dropna()

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'])

    # Keep the columns of interest.
    df = df[['time','datetime', 'SST']]

    df['year']  = pd.DatetimeIndex(df['datetime']).year
    df['month'] = pd.DatetimeIndex(df['datetime']).month
    df['day']   = pd.DatetimeIndex(df['datetime']).day

    #grouping
    df  = df.groupby(['year','month','day'], as_index=False)['SST'].max()

    sst = df['SST'].to_numpy()   
    tgl = np.array([date(df.year[x],df.month[x],df.day[x]) 
                        for x in range(len(df.year))])
    
    return sst, tgl

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
    HS = sst-MMM
    HS[HS<0]=0 ## change number < 0 to 0
        
    ##========= DHW ==============
    DHW = []
    HSm = HS-1   ##HS greater than MMM
    HSm[HSm<0]=0 ## change number < 0 to 0

    for i in range(len(HSm)):
        if i<84:
            a=np.sum(HSm[0:i+1])/7
        else:
            a=np.sum(HSm[i-84:i+1])/7
        
        DHW.append(a)
    
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


def dhw_hycom_gee(sdate,edate,geometry):
    """---------------------------------------------------------------------------------
    Load and extract SST data from HYCOM GEE dataset 
    (https://developers.google.com/earth-engine/datasets/catalog/HYCOM_sea_temp_salinity)
    in specific region (polygon) and time range. Output SST is the average SST inside 
    the polygon.
    
    input : -polygon of interested area
            -start time, end time
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW) 
    -------------------------------------------------------------------------------------"""
    
    def calcMean(img):
        # gets the mean NDVI for the area in this img
        mean = img.reduceRegion(ee.Reducer.mean(), geometry, 8905.6).get('water_temp_0') ##Hycom
        
        # sets the date and the mean NDVI as a property of the image
        return img.set('date', img.date().format()).set('mean', mean)

    dataset = ee.ImageCollection('HYCOM/sea_temp_salinity')
    col = dataset.select('water_temp_0').filterDate(sdate,edate).map(calcMean)

    # Reduces the images properties to a list of lists
    values = col.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)
    
    # Type casts the result into a List
    lista = ee.List(values).getInfo()
    
    sst, tgl = List2SSTDay(lista)
    
    # convert unit of sst hycom to deg C.
    sst = sst*0.001+20 ##hycom
    
    MM,MMM,HS,DHW    = dhw_calc(tgl,sst) # call MM, MMM, HS, DHW
    
    ## save data to csv
    MMMsv  = MMM*np.ones(len(DHW))
    df_asc = pd.DataFrame({'tgl':np.ravel(tgl),'sst':np.ravel(sst),
                           'MMM':MMMsv,'HS':np.ravel(HS),'DHW':np.ravel(DHW)})
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

def dhw_noaa_oisst_gee(sdate,edate,geometry):
    """------------------------------------------------------------------------------
    Load and extract SST data from NOAA OISST GEE dataset
    (https://developers.google.com/earth-engine/datasets/catalog/NOAA_CDR_OISST_V2_1)
    in specific region (polygon) and time range. Output SST is the average SST inside 
    the polygon.
    
    input : -polygon of interested area
            -start time, end time
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW) 
    ---------------------------------------------------------------------------------"""
    
    def calcMean(img):
        # gets the mean NDVI for the area in this img
        mean = img.reduceRegion(ee.Reducer.mean(), geometry, 27830).get('sst') 
        
        # sets the date and the mean NDVI as a property of the image
        return img.set('date', img.date().format()).set('mean', mean)

    dataset = ee.ImageCollection('NOAA/CDR/OISST/V2_1')
    col = dataset.select('sst').filterDate(sdate,edate).map(calcMean)

    # Reduces the images properties to a list of lists
    values = col.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)
    
    # Type casts the result into a List
    lista = ee.List(values).getInfo()
    
    sst, tgl = List2SSTDay(lista)
    
    # convert unit of sst hycom to deg C.
    sst = sst*0.01
    
    MM,MMM,HS,DHW    = dhw_calc(tgl,sst) # call MM, MMM, HS, DHW
    
    ## save data to csv
    MMMsv  = MMM*np.ones(len(DHW))
    df_asc = pd.DataFrame({'tgl':np.ravel(tgl),'sst':np.ravel(sst),
                           'MMM':MMMsv,'HS':np.ravel(HS),'DHW':np.ravel(DHW)})
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
    
def dhw_modis_gee(sdate,edate,geometry):#####Belum Selesai#####
    """------------------------------------------------------------------------------
    Load and extract SST data from MODIS-Aqua GEE dataset
    (https://developers.google.com/earth-engine/datasets/catalog/NASA_OCEANDATA_MODIS-Aqua_L3SMI)
    in specific region (polygon) and time range. Output SST is the average SST inside 
    the polygon.
    
    input : -polygon of interested area
            -start time, end time
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW) 
    ---------------------------------------------------------------------------------"""
    
    def calcMean(img):
        # gets the mean NDVI for the area in this img
        mean = img.reduceRegion(ee.Reducer.mean(), geometry, 4616).get('sst') 
        
        # sets the date and the mean NDVI as a property of the image
        return img.set('date', img.date().format()).set('mean', mean)

    dataset = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI')
    col = dataset.select('sst').filterDate(sdate,edate).map(calcMean)

    # Reduces the images properties to a list of lists
    values = col.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)
    
    # Type casts the result into a List
    lista = ee.List(values).getInfo()
    
    sst, tgl = List2SSTDay(lista)
    
    MM,MMM,HS,DHW    = dhw_calc(tgl,sst) # call MM, MMM, HS, DHW
    
    ## save data to csv
    MMMsv  = MMM*np.ones(len(DHW))
    df_asc = pd.DataFrame({'tgl':np.ravel(tgl),'sst':np.ravel(sst),
                           'MMM':MMMsv,'HS':np.ravel(HS),'DHW':np.ravel(DHW)})
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

def dhw_noaa_pathfinder_gee(sdate,edate,geometry):#####Belum Selesai#####
    """------------------------------------------------------------------------------
    Load and extract SST data from NOAA Pathfinder v5.3 GEE dataset
    (https://developers.google.com/earth-engine/datasets/catalog/NOAA_CDR_SST_PATHFINDER_V53)
    in specific region (polygon) and time range. Output SST is the average SST inside 
    the polygon.
    
    input : -polygon of interested area
            -start time, end time
    Output: - Maximum of the Monthly Mean (MMM)
            - Hot Spot (HS)
            - Degree Heating Weeks (DHW) 
    ---------------------------------------------------------------------------------"""
    
    def calcMean(img):
        # gets the mean NDVI for the area in this img
        mean = img.reduceRegion(ee.Reducer.mean(), geometry, 4000).get('sea_surface_temperature') 
        
        # sets the date and the mean NDVI as a property of the image
        return img.set('date', img.date().format()).set('mean', mean)

    dataset = ee.ImageCollection('NOAA/CDR/SST_PATHFINDER/V53')
    col = dataset.select('sea_surface_temperature').filterDate(sdate,edate).map(calcMean)

    # Reduces the images properties to a list of lists
    values = col.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)
    
    # Type casts the result into a List
    lista = ee.List(values).getInfo()
    
    sst, tgl = List2SSTDay(lista)
    
    # convert unit of sst hycom to deg C.
    sst = sst*0.01
    
    #remove data <=0
    a=np.where(sst<=0)
    sst = np.delete(sst,a)
    tgl = np.delete(tgl,a)
    
    MM,MMM,HS,DHW    = dhw_calc(tgl,sst) # call MM, MMM, HS, DHW
    
    ## save data to csv
    MMMsv  = MMM*np.ones(len(DHW))
    df_asc = pd.DataFrame({'tgl':np.ravel(tgl),'sst':np.ravel(sst),
                           'MMM':MMMsv,'HS':np.ravel(HS),'DHW':np.ravel(DHW)})
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
    

"""============================================================
Change the coordinate of the polygon as you desire.
Change the start date (sdate) and end date (edate) as You desire
==============================================================="""
geometry = ee.Geometry.Polygon(
  [[[119.7541369953,-8.4997906243],
    [119.5015440462,-8.5014967056],
    [119.4999374502,-8.2500048977],
    [119.7523686501,-8.2483499727],
    [119.7541369953,-8.4997906243]]])

sdate = '2010-01-01' # Initial date of interest (inclusive).
edate = '2020-12-31' # Final date of interest (exclusive)

"""==============================================================
Call the function and calculate DHW.
you can run all at once by uncomment the script.
or you can run one by one by comment and uncomment the data you desire.
=============================================================="""
# dhw_hycom_gee(sdate,edate,geometry)  ## calculate DHW using Hycom SST Data
# dhw_noaa_oisst_gee(sdate,edate,geometry) ## calculate DHW using OISST Data
# dhw_modis_gee(sdate,edate,geometry) ## calculate DHW using MODIS SST Data
dhw_noaa_pathfinder_gee(sdate,edate,geometry) ## calculate DHW using Pathfinder SST Data





