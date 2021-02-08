# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:21:21 2021

@author: Christopher Grießhaber

Programming environment: Spyder 3.8
"""

import numpy as np  # useful for scientific computing in Python
import pandas as pd # primary data structure library
import datetime
import math
import seaborn as sns
from matplotlib import pyplot as plt
import os

#pip install gdown
import gdown

eval_no = 0
pd.set_option('display.max_columns', None)


## ----------------------------------------------------------------------
## Data Understanding and Preparation -----------------------------------
## ----------------------------------------------------------------------


# Original data can be found on the official webpage of msci:
# https://www.msci.com/end-of-day-data-search
# The loaded csv is a collection of the indices "Developed Markets Standard (Large+Mid Cap)," 
# "Developed Markets (Small Cap)" and "Emerging Markets Standard (Large+Mid Cap)"

# For each index three variants of data were loaded: "Net","Gross" and "Price"
# Depending of the index the data reaches back to between 31.12.1969 and 29.12.2000:
    # Starting dates:
    # 31.12.1969: Developed Markets Standard (Large+Mid Cap) Net, Gross and Price
    # 31.12.1987: Emerging Markets Standard (Large+Mid Cap) Gross and Price
    # 31.12.1992: Developed Markets (Small Cap) Price
    # 29.12.2000: Emerging Markets Standard (Large+Mid Cap) Net
    #             Developed Markets (Small Cap) Net and Gross
# All indices contain data until 31.12.2020
# The data is recorded in monthly intervals.




# url = 'https://drive.google.com/file/d/1tGC-zZPJZVu68QPApb1Jl6lf_NL4d4Gm/view?usp=sharing'
# urllib.request.urlretrieve(url, "test.xlsx")

# Define the folder, in which the graphs should be saved.
# Be aware, that the folder must exist.
strFolderPath_Graphs = 'C:/Users/Chris2015/Desktop/SpyderPython/Graphs/'

# if gdown isn't installed, you have to install it to download the file



# Performance History Data gdrive path (data in gdrive, so that it can
# be deleted after accomplishing the capstone project)
url = 'https://drive.google.com/uc?id=1tGC-zZPJZVu68QPApb1Jl6lf_NL4d4Gm'

# Temporary Filename
strFilenamePerformanceData = 'MSCI_PerformanceDataTemporary.xlsx'

# downloading the file
gdown.download(url, strFilenamePerformanceData, quiet=False) 


# Read csv data to a dataframe, skip the footer
df_msci = pd.read_excel(strFilenamePerformanceData,
                       sheet_name='Data',
                       skiprows=range(2),
                       skipfooter=18,
                       header=0
                      )

# delete the temporary file again (using os)
os.remove(strFilenamePerformanceData)

print('Data downloaded and read into a dataframe!')
print('______________________________________________________________________')
print(' ')

print('Show first 2 rows of df_msci:')
print(df_msci.head(2))
print(' ')
print('Show shape of df_msci:')
print(df_msci.shape)
print('______________________________________________________________________')
print(' ')

# Ensure, that columns are of the type String and Uppercase
df_msci.columns = list(map(str, df_msci.columns))
df_msci.rename(columns = lambda x: x.upper(),inplace=True)

# Convert the date column to pandas datetime format
df_msci['DATE'] = pd.to_datetime(df_msci['DATE'])


# ----------------------------------------------------------------------------
# define function to calculate the return rate
def calculateReturnRate(y2,y1,x2 = 1,x1 = 1):
    ReturnRate = ((y2/y1)**(1/((x2+1)-x1)))-1
    return ReturnRate

# ----------------------------------------------------------------------------
# set up total expanse ratio for all index types
# in general you'll can get the MSCI world and EM index for a TER, which is 
# lower than 0.2% (nowadays) and the small cap index for approx. TER=0.35% 

dict_TER = {'WORLD_LMC_NET': 0.2 / 100,
            'WORLD_LMC_GROSS': 0.2 / 100,
            'WORLD_LMC_PRICE': 0.2 / 100,
            'EM_LMC_NET': 0.2 / 100,
            'EM_LMC_GROSS': 0.2 / 100,
            'EM_LMC_PRICE': 0.2 / 100,
            'WORLD_SC_NET': 0.35 / 100,
            'WORLD_SC_GROSS': 0.35 / 100,
            'WORLD_SC_PRICE': 0.35 / 100
            }

# define function to reduce index by total expense ratio (TER) of anually 0.2%
# standard value: TER = 0.2 / 100
# print('TER: ' + str(TER*100) + "%")

def calculateIndexCorrection(y1, fcn_returnRate):
    ReturnRate = y1 * (fcn_returnRate + 1)
    return ReturnRate

# ----------------------------------------------------------------------------

print('Return Monthly:')
print(' ')

# create copy of df_msci and remove first row to be able to calculate the return rate
df_ReturnMonth = df_msci.copy()
df_ReturnMonth = df_ReturnMonth.drop([0],axis=0)
df_ReturnMonth.reset_index

# create 2nd copy to recalculate df_msci (consider TER)
df_msciCorrected_Month = df_msci.copy()

# for loop to perform calculations for all 9 indices (columns):
for i_column in range(df_msci.shape[1]):
    # don't consider DATE column for calculation
    if i_column > 0:
        # get TER for index
        TER = dict_TER[df_msci.columns[i_column]]

        # create new empty dictionary
        new_items = {}        
        # calculate return rate (with function calculateReturnRate for each 
        # time interval based on current and previous row and save it to the dictionary
        new_items = {i_row: calculateReturnRate(df_msci.iloc[i_row,i_column],df_msci.iloc[i_row-1,i_column]) for i_row in range(1,df_msci.shape[0])}
        new_items[0] = df_msci.iloc[0,i_column]
        # save dictionary to dataframe (after conversion to Pandas Series)
        df_ReturnMonth.iloc[:,i_column]= (pd.Series(new_items))
        # subtract TER from return rate 
        df_ReturnMonth.iloc[:,i_column]= df_ReturnMonth.iloc[:,i_column]-(1-((1-TER)**(1/12)))

# ----------------------------------------------------------------------------
# for loop to correct original index values (subtract TER)
for i_column in range(df_msci.shape[1]):
    # don't consider DATE column for calculation
    if i_column > 0:        
        # create 2nd empty dictionary
        new_items_Corrected = {}
        
        # calculate corrected index value (with function calculateIndexCorrection for each 
        # row and save it to the dictionary 
        new_items_Corrected[0] = df_msci.iloc[0,i_column]
        for i_row in range(1,df_msci.shape[0]):
            new_items_Corrected[i_row] = calculateIndexCorrection(new_items_Corrected[i_row-1],df_ReturnMonth.iloc[i_row-1,i_column])
            if math.isnan(new_items_Corrected[i_row]):
                new_items_Corrected[i_row] = df_msciCorrected_Month.iloc[i_row,i_column]
        
        # save dictionary to dataframe (after conversion to Pandas Series)
        df_msciCorrected_Month.iloc[:,i_column]= (pd.Series(new_items_Corrected)) 
        
    
print('Show first and last 2 rows of df_ReturnMonth:')
print(df_ReturnMonth.head(2))
print(df_ReturnMonth.tail(2))
print(' ')
print('Show shape of df_ReturnMonth:')
print(df_ReturnMonth.shape)
print(' ')
print('______________________________________________________________________')
print(' ')  
print('Show first and last 2 rows of df_msciCorrected_Month:')
print(df_msciCorrected_Month.head(2))
print(df_msciCorrected_Month.tail(2))
print(' ')
print('______________________________________________________________________')
print(' ')     


# add return rate to original df_msciCorrected_Month
df_msciCorrected_Month['WORLD_LMC_NET_PERFORMANCE'] = df_ReturnMonth['WORLD_LMC_NET'].copy()
df_msciCorrected_Month['EM_LMC_NET_PERFORMANCE'] = df_ReturnMonth['EM_LMC_NET'].copy()
df_msciCorrected_Month['WORLD_SC_NET_PERFORMANCE'] = df_ReturnMonth['WORLD_SC_NET'].copy()

# add month and year column for easier analyzing
df_msciCorrected_Month['MONTH'] = df_msciCorrected_Month['DATE'].apply(lambda x: x.month)
df_msciCorrected_Month['YEAR'] = df_msciCorrected_Month['DATE'].apply(lambda x: x.year)

#df_ReturnMonth['MONTH'] = df_ReturnMonth['DATE'].apply(lambda x: x.month)
#df_ReturnMonth['YEAR'] = df_ReturnMonth['DATE'].apply(lambda x: x.year)

# ----------------------------------------------------------------------------

# Function to get X-y data based on date range for selected index_df
def getNormalizedIndexData(fcn_df, boolIncludeMonthYear = False,fcn_month_step = 1, fcn_dateStart = '01.12.1969', fcn_dateEnd = '01.01.2021'):
    # convert datestring to date format
    datstart_date = datetime.datetime.strptime(fcn_dateStart, "%d.%m.%Y")
    datend_date = datetime.datetime.strptime(fcn_dateEnd, "%d.%m.%Y")
    
    # set date mask for fcn_df
    mask = (fcn_df['DATE'] >= datstart_date) & (fcn_df['DATE'] <= datend_date)
    df_CI = fcn_df.loc[mask] 
    
    # find index of first row in selected date range
    start_idx = df_CI.iloc[:,0].index.values[0]
    # get all rows within date range, which corresponds to selected step width
    df_CI = df_CI[(df_CI.index-start_idx) % fcn_month_step == 0]
    
    # get reference df_CI_ref without NaN values, to normalize all data to the same
    # start value at the same date
    df_CI_ref = df_CI.dropna()    
    for i_column in range(df_CI.shape[1]):
        if i_column > 0:
            # normalize data based on first value of reference df_CI_ref
            df_CI.iloc[:,i_column]= df_CI.iloc[:,i_column].apply(lambda x: x * 100 / df_CI_ref.iloc[0,i_column])
    if boolIncludeMonthYear:
        df_CI['MONTH'] = df_CI['DATE'].apply(lambda x: x.month)
        df_CI['YEAR'] = df_CI['DATE'].apply(lambda x: x.year)

    # return normalized dataframe, which fulfills all given criteria
    return df_CI

# ----------------------------------------------------------------------------
# Collection of PLOT FUNCTIONS for easier multiple usage (using Seaborn)
# ----------------------------------------------------------------------------

# set a common ColorPalette and figure size
strColorPalette ='PuBuGn'
fig_width = 30
fig_height = 16

fontsize_title = 30
fontsize_subtitle = 28
fontsize_legend = 28
fontsize_xlabel = 28
fontsize_ylabel = 28
fontsize_ticks = 24
fontsize_heatmap = 20

# simple function to replace some used chars 
# within the graphs title to save the figure / mainly beautiness purposes
def strSavePath(old_title,eval_no):
    new_title = old_title.replace(" \n", "")    
    new_title = new_title.replace("\n", "")    
    new_title = new_title.replace("/", "per")
    new_title = new_title.replace("%", "")
    new_title = new_title.replace("[", "")
    new_title = new_title.replace("]", "")    
    new_title = new_title.replace(" ", "_")
    new_title = new_title.replace("(", "")
    new_title = new_title.replace(")", "")
    new_title = new_title.replace("|", "")
    new_title = new_title.replace(":", "")
    
        
    if new_title[-1] == '_':
        new_title = new_title[:-1]
        
    # save path relative to working directory / folder "Graphs" already exists
    new_title = strFolderPath_Graphs + "{:02d}".format(eval_no) + '_' + new_title + '.png'
    return new_title


# function to actually show, label, save and clear the figure
def plotLabelling(plot,plt,x_label,y_label,title,eval_no):
    plt.xlabel(x_label, fontsize=fontsize_xlabel, weight='bold')
    plt.ylabel(y_label, fontsize=fontsize_ylabel, weight='bold')
    plt.tick_params(labelsize=fontsize_ticks)       
    
    sns.despine() # remove top and right border frame
    plt.show()
    plot.get_figure().savefig(strSavePath(title,eval_no))
    plot.get_figure().clf()

#----------------------------------------------------------------------
# function to create line plot
def plotLineSeaborn(fcn_df_plot, title, eval_no, subtitle = '', xAx='DATE', yAx='value',
                       x_label='Date',y_label='Index Value', hueAx='variable', 
                       colorPalette = strColorPalette):    
    
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
    
    plot = sns.lineplot(x=xAx, y=yAx, hue=hueAx, 
                 data=pd.melt(fcn_df_plot, [xAx]),
                 palette=colorPalette).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    plt.legend(fontsize=fontsize_legend)
        
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    
    
# function to create line plot for several columns
def plotLineSeabornHue(fcn_df_plot, title, eval_no, subtitle = '', xAx='DATE', yAx='value',
                       x_label='Date',y_label='Index Value', hueAx='variable', 
                       colorPalette = strColorPalette):    
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
        
    plot = sns.lineplot(x=xAx, y=yAx, hue=hueAx, 
                 data=fcn_df_plot,
                 palette=colorPalette).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    plt.legend(fontsize=fontsize_legend)
    
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    

# function to create lmp plot (regression)
def plotLMPlot(fcn_df_plot, title, eval_no, no_order, xAx='DATE', yAx='value',
                       x_label='Date',y_label='Index Value', hueAx='variable',
                       colorPalette = strColorPalette, boolXestim = True, 
                       boolYLim = False):    
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
    
    # use mean as x_estimator, if parameter boolXestim is set
    if boolXestim:    
        plot = sns.lmplot(x=xAx, y=yAx, hue=hueAx,
                     data=fcn_df_plot, x_estimator=np.mean,
                     palette=colorPalette,
                     legend_out=False,
                     height=fig_height,
                     aspect=fig_width/fig_height,
                     x_jitter=.4,
                     order=no_order)
    else:
        plot = sns.lmplot(x=xAx, y=yAx, hue=hueAx,
                     data=fcn_df_plot,
                     palette=colorPalette,
                     legend_out=False,
                     height=fig_height,
                     aspect=fig_width/fig_height,
                     x_jitter=.4,
                     order=no_order)
    
    # use defined YLim if boolYLim was sent with True
    if boolYLim:
        plot.set(ylim=(0, 2))
    
    # different saving algorithm for regression plot due to a special
    # behaviour in spyder 3.8
    plot = plot.set_axis_labels(x_label, y_label)
    
    plot.fig.suptitle(title, fontsize=fontsize_title, weight='bold')  
    
    plt.xlabel(x_label, fontsize=fontsize_xlabel, weight='bold')
    plt.ylabel(y_label, fontsize=fontsize_ylabel, weight='bold')
    plt.tick_params(labelsize=fontsize_ticks)    
    
    plt.legend(loc='upper left')
    plt.legend(fontsize=fontsize_legend)
    
    sns.despine()
    
    plt.show()
    plot.savefig(strSavePath(title,eval_no),dpi=200)
    

from scipy import stats    
def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2
    
def plotJointplot(fcn_df_plot, title, eval_no, xAx='DATE', yAx='value',
                       x_label='Date',y_label='Index Value',
                       colorPalette = strColorPalette):
    
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle('Jointplot ' + title, fontsize=fontsize_title, weight='bold') 
    
    plot = sns.jointplot(x=xAx, y=yAx,                          
                     height=fig_height,
                     data=fcn_df_plot, kind="reg")
    
    plot = plot.set_axis_labels(x_label, y_label)
    
    plot.fig.suptitle(title, fontsize=fontsize_title, weight='bold')  
    
    plt.xlabel(x_label, fontsize=fontsize_xlabel, weight='bold')
    plt.ylabel(y_label, fontsize=fontsize_ylabel, weight='bold')
    plt.tick_params(labelsize=fontsize_ticks)    
    
    # plt.legend(loc='upper left')
    # plt.legend(fontsize=fontsize_legend)
    
    sns.despine()
    
    plt.show()
    plot.savefig(strSavePath(title + '_JointPlot',eval_no),dpi=200)

# function to create bar plot
def plotBarSeaborn(fcn_df_plot, title, eval_no, subtitle = '', xAx = 'DATE', yAx = 'value',x_label='Date',y_label='Index Value', bool_Legend = False, colorPalette = strColorPalette):
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
        
    plot = sns.barplot(x=xAx, y=yAx, data=fcn_df_plot,
                 palette=colorPalette).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    if bool_Legend:
        plt.legend(fontsize=fontsize_legend)
    
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    

# function to create grouped bar plot
def plotGroupedBarSeaborn(fcn_df_plot, title, eval_no, subtitle = '', xAx = 'StartingYear', yAx = 'value',hueAx = 'InvestmentType', x_label='Date',y_label='Index Value', bool_Legend = False, colorPalette = strColorPalette):
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
        
    plot = sns.barplot(x=xAx, y=yAx, hue=hueAx ,data=fcn_df_plot,
                 palette=colorPalette).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    if bool_Legend:
        plt.legend(fontsize=fontsize_legend)
    
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    

# function to create density plot (kdeplot)
def plotDensitySeaborn(fcn_df_plot, title, eval_no, subtitle = '', xAx = 'DATE', yAx = 'value',x_label='Date',y_label='Index Value', colorPalette = strColorPalette):
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
        
    plot = sns.kdeplot(x=xAx, y=yAx, data=fcn_df_plot,
                 palette=colorPalette).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    plt.legend(fontsize=fontsize_legend)
    
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    

# function to create Boxplot
def plotBoxplotSeaborn(fcn_df_plot, title, eval_no, subtitle = '', xAx = 'DATE', yAx = 'value',x_label='Date',y_label='Index Value',bool_sort = False, bool_Legend = False, colorPalette = strColorPalette):
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
        
    plot = sns.boxplot(x=xAx, y=yAx, data=fcn_df_plot,
                 palette=colorPalette, boxprops=dict(alpha=.7)).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    
    # get median values (for labeling position)
    medians = fcn_df_plot.groupby([xAx],sort=bool_sort)[yAx].median()
        
    # get max, min values for labeling
    tmax = fcn_df_plot[yAx].max()
    tmedian = fcn_df_plot[yAx].median()
    tmin = fcn_df_plot[yAx].min()

    # calculate an usefull offset for the labeling (based on tmax, tmedian and tmin)
    vertical_offset = ((tmax-tmin)/(tmedian-tmin)) * 13 # offset from median for display    
    if vertical_offset > tmax:
        vertical_offset = tmedian*0.02
    if tmax < 50:
        vertical_offset = vertical_offset / 200
           
    # get unique values and sort it eventually
    arr_uni = np.array(fcn_df_plot[xAx].unique())
    arr_uni.sort()
    arr_uni2 = arr_uni.dtype
    
    # get type and determine index (relevant, because index values can
    # vary and string indexes shall work as well using range)
    if str(arr_uni2) == 'int64':        
        loop_range = arr_uni
    else:
        loop_range = range(0,np.size(medians, 0))
    xtick = 0
    
    
    strNumFormat = "{:.3f}"
    if tmax > 10:
        strNumFormat = "{:.2f}"
    if tmax > 100:
        strNumFormat = "{:.1f}"
    if tmax > 1000:
        strNumFormat = "{:.0f}"        
    
    # run for-loop to set the median labels corresponding to their relative position
    for xloop in loop_range:        
        plt.text(xtick , medians[xloop] + vertical_offset,strNumFormat.format(medians[xloop]), 
                horizontalalignment='center',fontsize=fontsize_heatmap,color='k',weight='semibold')        
        xtick = xtick + 1
    
    if bool_Legend:
        plt.legend(fontsize=fontsize_legend)
    
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    

# function to create Violin Swarm Plot
def plotViolinSwarmSeaborn(fcn_df_plot, title, eval_no, subtitle = '', xAx = 'DATE', yAx = 'value',x_label='Date',y_label='Index Value', bool_Legend = False, colorPalette = strColorPalette):
    # Set figure size with matplotlib    
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
        
    # Create plot
    plot = sns.violinplot(x= xAx, y= yAx, data= fcn_df_plot, 
                   inner=None, # Remove the bars inside the violins
                   palette=colorPalette)
     
    plot = sns.swarmplot(x= xAx, y= yAx, data= fcn_df_plot, 
                  color='k', # black points
                  alpha=0.7).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    if bool_Legend:
        plt.legend(fontsize=fontsize_legend)
    
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    

# function to create Heatmap for profit triangles
def plotHeatmap(fcn_df_plot, title, eval_no, subtitle = '', xAx = 'MONTH_since', yAx = 'MONTH_hold',x_label='Month since',y_label='Month start', colorPalette = 'BrBG', dbl_vmin=-10,dbl_vmax=10):
    
    piv_fcn_df_plot = fcn_df_plot.pivot(yAx, xAx, 'value')
    
    # Set figure size with matplotlib    
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
    
    # setup, whether labels shall be shown or not in reference to the
    # number of rows and set an usefull linewidth
    if piv_fcn_df_plot.shape[1]> 48:
        boolAnnot= False
        dblLinewidths = .01
    else:
        boolAnnot= True
        dblLinewidths = .1
        
    # Create plot
    plot = sns.heatmap(piv_fcn_df_plot, vmin= dbl_vmin, vmax= dbl_vmax, 
                       annot=boolAnnot,
                       annot_kws={"size": fontsize_heatmap},fmt='.1f', 
                       linewidths= dblLinewidths, linecolor='whitesmoke',
                       cmap=colorPalette).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    
    #set y-limits based on df shape
    shape1 = 0
    shape2 = piv_fcn_df_plot.shape[1]    
    plt.ylim(shape1,shape2) 
    
    #set x-limits based on df shape
    shape1 = 0
    shape2 = piv_fcn_df_plot.shape[0]
    plt.xlim(shape1,shape2) 
    
    #rotate labels accordingly
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=fontsize_heatmap)
    
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    

# function to create Heatmap for pearson correlation matrix
def plotHeatmapPearson(fcn_df_plot, title, eval_no, subtitle = '', xAx = 'MONTH_since', yAx = 'MONTH_hold',x_label='Month since',y_label='Month start', colorPalette = 'BrBG', dbl_vmin=-10,dbl_vmax=10):
    
    # setup, whether labels shall be shown or not in reference to the
    # number of rows and set an usefull linewidth
    if fcn_df_plot.shape[1]> 48:
        boolAnnot= False
        dblLinewidths = .01
    else:
        boolAnnot= True
        dblLinewidths = .1
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(fcn_df_plot, dtype=bool))
    
    # Set figure size with matplotlib  
    fig = plt.figure(figsize=(fig_width,fig_height))
    fig.suptitle(title, fontsize=fontsize_title, weight='bold') 
    
    # Draw the heatmap with the mask and correct aspect ratio
    plot = sns.heatmap(fcn_df_plot, mask=mask, vmin= dbl_vmin, vmax= dbl_vmax, 
                       center=0, annot=boolAnnot,
                       annot_kws={"size": fontsize_heatmap},fmt='.3f', 
                       linewidths= dblLinewidths, cmap=colorPalette,
                       linecolor='whitesmoke',
                square=True).set_title(subtitle, fontsize=fontsize_subtitle, weight='normal') 
    #to shrink the colorbar this code can be used: cbar_kws={"shrink": 0.75}
    
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=fontsize_heatmap)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    
    plotLabelling(plot,plt,x_label,y_label,title + ' ' + subtitle,eval_no)    
       
# ----------------------------------------------------------------------------
# ANALYZE DATA
# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 1. Explanation of index level: net vs gross vs price
print(str(eval_no) + '. Explanation of index level: net vs gross vs price')
df_msci_orig_world = df_msci[['DATE','WORLD_LMC_NET','WORLD_LMC_GROSS','WORLD_LMC_PRICE']]

# use predefined plot function to plot the index data
plotLineSeaborn(getNormalizedIndexData(df_msci_orig_world, False, 1), 
                'Cumulative Index Performance (Net vs Gross vs Price)',eval_no, 'WORLD LMC')

print('Succesfull.')
print(' ')
# The considered index is the MSCI World Index. This index covers the developed
# markets in 23 developed countries. Find more details to this index on the
# official page of MSCI:
# https://www.msci.com/developed-markets

# Price: the price index shows the normal index value. In this index all dividend
# payments will not be reinvested in the index. The investor might use this money
# for consumption.

# Gross: in this index the dividends will be reinvested in the index completely. 
# This money will increase the shares of the investor. For the gross index no 
# taxes are considered, therefor this index cannot be achieved for the investor.

# Net: in this index the net dividens will be reinvisted in the index. This is
# a commonly realistic investment behavior and mirrors accumulating indexes best.

# ----------------------------------------------------------------------------
# Normalizing

eval_no = eval_no + 1
# 2. Explanation of index normalization
print(str(eval_no) + '. Explanation of index normalization (net-index')
df_msci_orig_world = df_msci[['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']]
plotLineSeaborn(df_msci_orig_world, 
                'Cumulative Net Index Performance',eval_no,'WORLD LMC vs EM LMC vs WORLD SC')
print('Succesfull.')
print(' ')
# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 3. Explanation of TER
print(str(eval_no) + '. Explanation of TER')
df_msci_TER_world = pd.DataFrame()
df_msci_TER_world['DATE'] = df_msci['DATE'].copy()
df_msci_TER_world['WORLD_LMC_NET_original'] = df_msci['WORLD_LMC_NET'].copy()
df_msci_TER_world['WORLD_LMC_NET_subtractTER'] = df_msciCorrected_Month['WORLD_LMC_NET'].copy()

plotLineSeaborn(getNormalizedIndexData(df_msci_TER_world, False, 1),
                'Cumulative Net Index Performance (Original vs subtraction of TER)',eval_no,'WORLD LMC')
print('Succesfull.')
print(' ')

# TER: Investors generally are not able to get all stock shares of the index for free.
# typically investors can buy ETFs (Exchange traded funds) to benefit from low 
# costs to cover a huge range of different stock shares. Still these ETFs are
# not for free and investors need to pay for administration and some costs. This
# costs are typically described by the TER (total expanse ratio). This TER is
# a procentual cost value, which continuously has to be paid. In general the 
# annual TER is in a range of 0.1% to 0.5% (depending on the index).
# For this evaluation a TER of 0.2% is considered for highly capitalized ETF like
# MSCI World Large+Mid Cap and Emerging Markets Large+Mid Cap. For the MSCI
# World Small Cap 0.35% are used.
# The graph shows the influence of this TER and should be more realistic for
# the normal investor than the original index.
# Be aware, that the TER is chosen based on todays cost structure for ETF in
# Germany. 
# In the past the costs probably have been much higher. This cost effect for 
# ETF wasn't considered for this evaluation. Conclusion: this corrected net
# graph still shows an optimistic index behavior.

# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 4. Explanation of market for the index (World vs. Emerging Markets) 
# and capitalization segments (Large+Mid Cap vs. Small Cap)
print(str(eval_no) + '. Explanation of market for the index (World vs. Emerging Markets) and capitalization segments (Large+Mid Cap vs. Small Cap)')

df_msci_net = df_msciCorrected_Month[['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']].copy()
df_msci_net = getNormalizedIndexData(df_msciCorrected_Month[['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']], True, 1,'01.01.2001')
plotLineSeaborn(df_msci_net[['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']],
                'Normalized Cumulative Net Index Performance',eval_no, 'WORLD LMC vs EM LMC vs WORLD SC')
print('Succesfull.')
print(' ')

## ----------------------------------------------------------------------
## Modeling and Evaluation parts ----------------------------------------
## ----------------------------------------------------------------------

## 5. Computing Volatility
eval_no = eval_no + 1
print(str(eval_no) + '. Computing Volatility')
print(' ')


df_msci_net['WORLD_LMC_NET_vola'] = df_msci_net['WORLD_LMC_NET'].rolling(window=2).std()
df_msci_net['EM_LMC_NET_vola'] = df_msci_net['EM_LMC_NET'].rolling(window=2).std()
df_msci_net['WORLD_SC_NET_vola'] = df_msci_net['WORLD_SC_NET'].rolling(window=2).std()

# intialise data of lists. 
tdata = {'IndexName':['MSCI World LMC', 'MSCI EM LMC', 'MSCI World SC'], 
        'IndexVolatility':[df_msci_net['WORLD_LMC_NET_vola'].mean(), df_msci_net['EM_LMC_NET_vola'].mean(), df_msci_net['WORLD_SC_NET_vola'].mean()]} 

# Create DataFrame 
df_volatility = pd.DataFrame(tdata) 
# Create bar plot
plotBarSeaborn(df_volatility, 'MSCI NET Volatility',eval_no,'', 'IndexName', 'IndexVolatility','Index Name','Index Volatility')

print('Volatility of MSCI World LMC:            ' + "{:.2f}".format(df_msci_net['WORLD_LMC_NET_vola'].mean()))
print('Volatility of MSCI Emerging Markets LMC: ' + "{:.2f}".format(df_msci_net['EM_LMC_NET_vola'].mean()))
print('Volatility of MSCI World SC:             ' + "{:.2f}".format(df_msci_net['WORLD_SC_NET_vola'].mean()))
print(' ')

print('Succesfull.')
print(' ')

# In the previous graph the capitalizaion segments already have been mentioned,
# but not explained.
# The market capitalization of a company is represented by a value (typically in
# USD). This value is the value of all outstanding shares for this company.
# The market cap is used to size the company into different ranges: 
# Mega-cap  >= 200   billion USD
# Large-cap >=  10   billion USD  and < 200   billion USD
# Mid-cap   >=   2   billion USD  and <  10   billion USD
# Small-cap >=   0.3 billion USD  and <   2   billion USD
# Micro-cap >=  50   million USD  and <   0.3 billion USD
# Nano-cap                            <  50   million USD

# Profit in the stock market highly depends on the risk of the investment.
# Due to the nature of financial posibilities and number of products etc.
# the risk for a small company is higher than for big company. It therefor has
# to be expected, that for typical stock market situations the profit for Small
# caps should be higher than for Mega and Large-Caps. 
# The graph shows, that the MSCI World Small Cap index could nearly quintuple
# within the last ~5 years. The MSCI World Large+Mid Cap "only" tripled its 
# value. This behavior supports the before mentioned hypothesis.

# Also you would think, that it is riskier to invest in Emerging Countries.
# The MSCI Emerging Markets index covers 26 countries (including China, Taiwan,
# Korea, India,...). Find more details on the official page of MSCI:
# https://www.msci.com/emerging-markets
# You might also visit this webpage for more information:
# https://www.msci.com/market-cap-weighted-indexes
# In Emerging countries next to other topics coruption and the legal situation 
# is more of a topic than in developed countries. There it also has to be
# expected, that the risk premium and therefor potential profit for the
# invester should be higher.
# If you check the graph, you can see that the emerging markets index even could
# perform better than the MSCI World Small Cap index. This behavior also supports
# this hypothesis.

# Despite this good performance the risk for MSCI World SC and MSCI Emerging 
# Markets LMC is higher. This can be seen in an higher volatility.
# The average volatility for the normalized data:
# Volatility of MSCI World LMC:            3.35
# Volatility of MSCI Emerging Markets LMC: 9.17
# Volatility of MSCI World SC:             6.51

# ----------------------------------------------------------------------------

# 6. Determine the Pearson correlation for normalized plots 
# +1 = full linear positive correlation, 
# -1 = full linear negative correlation, 
# 0 = no linear correlation

eval_no = eval_no + 1
print(str(eval_no) + '. Determine the Pearson correlation for normalized plots')
print(' ')

df_correlation = df_msci_net[['WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']].corr(method ='pearson')
print(df_correlation)
melted_df_correlation = pd.melt(df_correlation, 
                    var_name="Index_Name") # Name of melted variable
print(melted_df_correlation)
plotHeatmapPearson(df_correlation, 'Index Pearson Correlation', eval_no,'', 'Index_Name', 'value', 'Index Name', 'Index Name', 'BrBG', -1,1)

print('Succesfull.')
print(' ')

# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 7. Average anual profit
print(str(eval_no) + '. Average anual profit')

number_of_years = df_msci_net.iloc[-1,5] - df_msci_net.iloc[0,5]
msci_world_lmc_index_profit = df_msci_net.iloc[-1,1] / df_msci_net.iloc[0,1]
msci_em_lmc_index_profit = df_msci_net.iloc[-1,2] / df_msci_net.iloc[0,2]
msci_world_sc_index_profit = df_msci_net.iloc[-1,3] / df_msci_net.iloc[0,3]

# intialise data of lists. 
tdata = {'IndexName':['MSCI World LMC', 'MSCI EM LMC', 'MSCI World SC'], 
        'IndexValueEnd':[msci_world_lmc_index_profit * 100, msci_em_lmc_index_profit * 100, msci_world_sc_index_profit * 100],
        'AvgAnualProfit':[(msci_world_lmc_index_profit**(1/number_of_years)-1)*100, (msci_em_lmc_index_profit**(1/number_of_years)-1)*100, (msci_world_sc_index_profit**(1/number_of_years)-1)*100]} 

# Create DataFrame 
df_profit = pd.DataFrame(tdata) 

plotBarSeaborn(df_profit, 'Cumulated Performance [%]',eval_no,'WORLD LMC vs EM LMC vs WORLD SC \nfrom 29.12.2000 until 31.12.2020', 'IndexName', 'IndexValueEnd','Index Name','Cumulated Performance [%]')
plotBarSeaborn(df_profit, 'Average Annual Performance [%]',eval_no,'WORLD LMC vs EM LMC vs WORLD SC', 'IndexName', 'AvgAnualProfit','Index Name','Average annual performance [%]')

print('Successfull.')
print(' ')

print('Number of Years: ' + str(number_of_years))
print('MSCI World LMC:            ' + "{:.1f}".format(msci_world_lmc_index_profit * 100) + ' %')
print('Ø anual profit:              ' + "{:.1f}".format((msci_world_lmc_index_profit**(1/number_of_years)-1)*100) + ' %')
print(' ')
print('MSCI Emerging Markets LMC: ' + "{:.1f}".format(msci_em_lmc_index_profit * 100) + ' %')
print('Ø anual profit:              ' + "{:.1f}".format((msci_em_lmc_index_profit**(1/number_of_years)-1)*100) + ' %')
print(' ')
print('MSCI World SC:             ' + "{:.1f}".format(msci_world_sc_index_profit * 100) + ' %')
print('Ø anual profit:              ' + "{:.1f}".format((msci_world_sc_index_profit**(1/number_of_years)-1)*100) + ' %')
print(' ')

# The next bar graph shows the development of the three indexes (MSCI 
# World LMC, MSCI EM LMC and MSCI World SC) within the last 19 years.
# The average anual profit was between 5.5 and 8.2%, despite the fact that there 
# have been three major crises (DotCom-bubble 2002, economic crisis 2009 and Covid19 2020)
# The profit shown is before the consideration of capital gain tax, which would
# be dued, if you sell your ETFs. In Germany you would have to pay approximately
# 26.375% tax and additionally another 2-2.25 % church tax, if you belong to a
# church, which is eligible to get this tax (2020). Also the anual inflation of
# approx. 1-2% per year isn't reflected in this evaluation.

# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 8. Boxplots Indextype

print(str(eval_no) + '. Boxplots Indextype')
boxplot_df = df_msciCorrected_Month.drop(
    ['WORLD_LMC_GROSS', 'WORLD_LMC_PRICE', 'EM_LMC_GROSS','EM_LMC_PRICE',
     'WORLD_SC_GROSS','WORLD_SC_PRICE','WORLD_LMC_NET',
     'EM_LMC_NET','WORLD_SC_NET'], axis=1)

melted_df = pd.melt(boxplot_df, 
                    id_vars=["DATE", "YEAR", "MONTH"], # Variables to keep
                    var_name="Index_Name") # Name of melted variable
melted_df['value'] = melted_df['value']*100

plotBoxplotSeaborn(melted_df,'Boxplot Monthly Performance [%]',eval_no,'','Index_Name','value','Index Name','Performance [/ month] in %')

print('Successfull.')
print(' ')

# The boxplot shows some information about the distribution of your data. The 
# colored box itself shows, in which range 50% of the data can be found. The
# black, solid and horizontal line within this colored area marks the median 
# value. The Median divids the data into two halfs. 50% of the measured data
# has a higher, 50% a lower value. The median is more robust against outliers
# than the average value and therefor quite suitable for this evaluation. The 
# vertical lines (finishing in a shorter horizontal line) shows the area, in
# which regular data might be expected. Values higher than the upper end and
# lower than the lower end are considered as outliers.

# It can be seen, that the median for all three indexes is positive, slightly 
# higher than 0%. Be aware, that the return rate is on a monthly basis, not
# yearly as usual.
# It can also be seen, that the distribution span for the MSCI World LMC is lower
# than for the other indexes. Therefor the potential average profit is lower,
# but the potential losses as well.
# Also it can be seen, that the highest monthly loss was approx. 20% for the
# MSCI World LMC, approx. 28% for the MSCI EM LMC and about 24% for the MSCI
# World SC.
# The maximum profits / month are for all indexes lower than 20% per month.
# Conclusion: Gaining money with ETF is taking longer than potentially losing
# money. Due to the suprising nature of most crises this effect could be expected
# though.

# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 9. Seasonal influence on return rate
# The next evaluation shall check for all three indexes, whether there might be
# a seasonal influence on the return rate / profit. Therefor the influence of 
# the month shall be checked for each index individually.

print(str(eval_no) + '. Seasonal influence on return rate')
df_msciCorrected_Month_plot = df_msciCorrected_Month

df_msciCorrected_Month_plot['WORLD_LMC_NET_PERFORMANCE'] = df_msciCorrected_Month_plot['WORLD_LMC_NET_PERFORMANCE'] * 100
df_msciCorrected_Month_plot['EM_LMC_NET_PERFORMANCE'] = df_msciCorrected_Month_plot['EM_LMC_NET_PERFORMANCE'] * 100
df_msciCorrected_Month_plot['WORLD_SC_NET_PERFORMANCE'] = df_msciCorrected_Month_plot['WORLD_SC_NET_PERFORMANCE'] * 100

print(df_msciCorrected_Month.head(5))
plotBoxplotSeaborn(df_msciCorrected_Month_plot,'Boxplot Monthly Performance over Months [%]',eval_no,'MSCI World LMC','MONTH','WORLD_LMC_NET_PERFORMANCE','Month','Performance [ / month] in %',True)
plotBoxplotSeaborn(df_msciCorrected_Month_plot,'Boxplot Monthly Performance over Months [%]',eval_no,'MSCI EM LMC','MONTH','EM_LMC_NET_PERFORMANCE','Month','Performance [ / month] in %',True)
plotBoxplotSeaborn(df_msciCorrected_Month_plot,'Boxplot Monthly Performance over Months [%]',eval_no,'MSCI World SC','MONTH','WORLD_SC_NET_PERFORMANCE','Month','Performance [ / month] in %',True)


plotViolinSwarmSeaborn(df_msciCorrected_Month_plot,'Violin-Swarmplot Monthly Performance over Months [%]',eval_no,'MSCI World LMC','MONTH','WORLD_LMC_NET_PERFORMANCE','Month','Performance [ / month] in %')
plotViolinSwarmSeaborn(df_msciCorrected_Month_plot,'Violin-Swarmplot Monthly Performance over Months [%]',eval_no,'MSCI EM LMC','MONTH','EM_LMC_NET_PERFORMANCE','Month','Performance [ / month] in %')
plotViolinSwarmSeaborn(df_msciCorrected_Month_plot,'Violin-Swarmplot Monthly Performance over Months [%]',eval_no,'MSCI World SC','MONTH','WORLD_SC_NET_PERFORMANCE','Month','Performance [ / month] in %')

print('Successfull.')
print(' ')

# The three boxplot graphs show some interesting behaviors. It can be seen,
# that all indexes seem to have a lower return rate for the median in the month 
# from May to August. Still the distribution is pretty wide (ranging in an area 
# from +-10%) for nearly every month. The question now is: Can someone derive 
# an investment instruction based on this minor finding with a decreased median
# return rate in the summer month? Actually no, because the median is still 
# higher than 0% and therefor you have a 50% chance, that you will gain profit
# in the summer month. Therefor the chance to miss profit, if you wait with the
# investment in the summer month is higher than 50% and an investor will avoid
# this.

# There is another argument, which supports this hypothesis: the efficient market
# theory. If it would be commonly known, that you can earn money, if you invest
# based on recurring events/months, this effect would be reflected in the price of the
# stock markets already. Therefor this effect would arbitrate itself. Also the
# analyzation of investment month is pretty obvious and not difficult to do,
# therefor it would be surprising, if no-one ever would have decovered such kind
# of effects.

# In the graph you also can see, that the lowest profit months can occur in all
# months, pratically you cannot foresee those months. Still it is possible, that
# the last huge stock crashes happened in March (Corona, 2020) and September 
# (Dot-Com Bubble, 2002 and Economic Crisis, 2008) and October (Economic Crisi,
# 2008)

# The violin plots show the shape of the distribution for each month. Due to
# the fact, that the return rates are also displayed, it is possible to get an
# impression of the number of samples, which could be used for each index.
# Due to the much longer history of data the MSCI World LMC has the most sample
# points and therefor the best statistical quality, but still (from statistical 
# point of view) the number of sample data is rather low for all indexes.

# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 10. Influence on return rate over time
# The next evaluation shall check, whether there is proof for a decreasing of
# the return rate over time or not. Therefor we will plot the profit of the
# three net indexes from the start until now.

print(str(eval_no) + '. Influence on return rate over time')

df_net_profit = df_msciCorrected_Month[['DATE','WORLD_LMC_NET_PERFORMANCE','EM_LMC_NET_PERFORMANCE',
              'WORLD_SC_NET_PERFORMANCE','MONTH','YEAR']].copy()

# melt the df for seaborn
melted_df = pd.melt(df_net_profit, 
                              id_vars=['DATE','MONTH','YEAR'], # Variables to keep
                              var_name="IndexName") # Name of melted variable

plotLineSeabornHue(melted_df,'Monthly Performance [%]',eval_no,'over years','YEAR','value',
                    'Year','Performance / Month [%]','IndexName','RdYlBu')

plotLMPlot(melted_df,'Monthly Performance [%] | individual',eval_no,1,'YEAR','value',
            'Year','Performance / Month [%]','IndexName','RdYlBu',False)

plotLMPlot(melted_df,'Monthly Performance [%] | spanwidth',eval_no,1,'YEAR','value',
            'Year','Performance / Month [%]','IndexName','RdYlBu')

plotLMPlot(melted_df,'Monthly Performance [%] | spanwidth 0.02',eval_no,1,'YEAR','value',
            'Year','Performance / Month [%]','IndexName','RdYlBu',False,True)

plotLMPlot(melted_df,'Monthly Performance [%] | spanwidth order_3',eval_no,2,'YEAR','value',
            'Year','Performance / Month [%]','IndexName','RdYlBu')

plotLMPlot(melted_df,'Monthly Performance [%] | spanwidth 0.02 order_3',eval_no,2,'YEAR','value',
            'Year','Performance / Month [%]','IndexName','RdYlBu',False,True)

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------

eval_no = eval_no + 1
print(str(eval_no) + '. Non-Linear Regression')
print(' ')
# 11. Non-Linear Regression
# There aren't many parameters to work with, but still regressions over time
# are possible. The index behaviour seems to be not linear and shows a high
# volatility, but an exponential behaviour might be worth a try. Therefor 
# the exp-function will be written and curve_fit from the scipy.optimize
# package will be used. If there would be more parameters and linear relationsships,
# several further and easy posibilities would be available.

# import curve_fit
from scipy.optimize import curve_fit

# define exponential function
def func_exp(x, a, b, c):
        #c = 0
        return a * np.exp(b * (x)) + c

# function to perform exponential regression using curve_fit based on X = Months
def calculate_regression (df_msci_net_reg, 
                          index_name = "WORLD_LMC_NET"
                          ):
    # get Index and X value for regression
    df_reg = df_msci_net_reg[[index_name,"X"]].copy()
    df_reg = df_reg.dropna()
    
    y_data = df_reg[index_name].values
    x_data = df_reg["X"]
    
    # perform curve_fit using the defined exponential function and an initial guess
    # of -1, 0.01,1 and a maximum of 10000 iterations for optimization
    [popt, pcov] = curve_fit(func_exp, x_data, y_data, p0 = (-1, 0.01, 1),maxfev = 10000)
   
    # the regression data will be saved into the dataframe
    df_reg[[index_name + "_reg"]] = func_exp(x_data, *popt)
    df_msci_net_reg[[index_name + "_reg"]] = df_reg[[index_name + "_reg"]]

    # R2 in this non linear regression problem doesn't help for a conclusion.
    # Therefor there is no method offered to get this value easily.
    # A manual calculation will be shown below, but the value will not be
    # used intendently.
    # Another easy alternative would be to use the sklearn.metrics.r2_score function.
    # residuals = y_data - func_exp(x_data, *popt)
    # ss_res = np.sum(residuals**2)
    # ss_tot = np.sum((y_data-np.mean(y_data))**2)
    # r_squared = 1 - (ss_res / ss_tot)
    
    return df_msci_net_reg, popt, pcov

# save month value to "X" column for easier usage
df_msci_net["X"] = df_msci_net.index.values

print(df_msci_net.head(3))
print(df_msci_net.tail(3))

# Perform regression and get output for WORLD LMC NET. 
# Using popt, x-values and the defined
# exponential function the y-values for the regression could be calculated.
# In this analyses popt will not be used for further analyses.
[df_msci_net, popt, pcov] = calculate_regression(df_msci_net)

# melt the df for seaborn plot
df_msci_net_reg1 = df_msci_net[["X",'MONTH','YEAR',"WORLD_LMC_NET","WORLD_LMC_NET_reg"]]
melted_reg_df = pd.melt(df_msci_net_reg1, 
                              id_vars=['X','MONTH','YEAR'], # Variables to keep
                              var_name="IndexName") # Name of melted variable

# plot the regression model
plotLineSeabornHue(melted_reg_df, 'Monthly Performance [%]', eval_no,'Exponential Regression for WORLD LMC NET', xAx='X', yAx='value',
                       x_label='Time [months]',y_label='Normalized Monthly Performance [%]', hueAx='IndexName')

plotJointplot(df_msci_net_reg1, 'Jointplot Cumulated Performance [%]', eval_no, 'WORLD_LMC_NET', 'WORLD_LMC_NET_reg',
                       'WORLD_LMC_NET','WORLD_LMC_NET_reg',
                       'RdYlBu')

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
eval_no = eval_no + 1
# 12. Perform regression and get output like before for EM LMC NET.
print(str(eval_no) + '. Perform regression and get output like before for EM LMC NET.')
print(' ')

[df_msci_net, popt, pcov] = calculate_regression(df_msci_net, "EM_LMC_NET")

# melt the df for seaborn plot
df_msci_net_reg1 = df_msci_net[["X",'MONTH','YEAR',"EM_LMC_NET","EM_LMC_NET_reg"]]
melted_reg_df = pd.melt(df_msci_net_reg1, 
                              id_vars=['X','MONTH','YEAR'], # Variables to keep
                              var_name="IndexName") # Name of melted variable

# plot the regression model
plotLineSeabornHue(melted_reg_df, 'Monthly Performance [%]', eval_no,'Exponential Regression for EM LMC NET', xAx='X', yAx='value',
                       x_label='Time [months]',y_label='Normalized Monthly Performance [%]', hueAx='IndexName')

plotJointplot(df_msci_net_reg1, 'Jointplot Cumulated Performance [%]', eval_no, 'EM_LMC_NET', 'EM_LMC_NET_reg',
                       'EM_LMC_NET','EM_LMC_NET_reg',
                       'RdYlBu')

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
eval_no = eval_no + 1
# 13. Perform regression and get output like before for WORLD SC NET.
print(str(eval_no) + '. Perform regression and get output like before for WORLD SC NET.')
print(' ')

[df_msci_net, popt, pcov] = calculate_regression(df_msci_net, "WORLD_SC_NET")

# melt the df for seaborn plot
df_msci_net_reg1 = df_msci_net[["X",'MONTH','YEAR',"WORLD_SC_NET","WORLD_SC_NET_reg"]]
melted_reg_df = pd.melt(df_msci_net_reg1, 
                              id_vars=['X','MONTH','YEAR'], # Variables to keep
                              var_name="IndexName") # Name of melted variable

# plot the regression model
plotLineSeabornHue(melted_reg_df, 'Monthly Performance [%]', eval_no,'Exponential Regression for WORLD SC NET', xAx='X', yAx='value',
                       x_label='Time [months]',y_label='Normalized Monthly Performance [%]', hueAx='IndexName')

plotJointplot(df_msci_net_reg1, 'Jointplot Cumulated Performance [%]', eval_no, 'WORLD_SC_NET', 'WORLD_SC_NET_reg',
                       'WORLD_SC_NET','WORLD_SC_NET_reg',
                       'RdYlBu')

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
eval_no = eval_no + 1
# 14. Plot all regressions in one figure.
print(str(eval_no) + '. Plot all regressions in one figure.')
print(' ')

df_msci_net_reg1 = df_msci_net[["X",'MONTH','YEAR',"WORLD_LMC_NET","WORLD_LMC_NET_reg","EM_LMC_NET","EM_LMC_NET_reg","WORLD_SC_NET","WORLD_SC_NET_reg"]]

# melt the df for seaborn plot
melted_reg_df = pd.melt(df_msci_net_reg1, 
                              id_vars=['X','MONTH','YEAR'], # Variables to keep
                              var_name="IndexName") # Name of melted variable

# plot the regression models
plotLineSeabornHue(melted_reg_df, 'Monthly Performance [%]', eval_no,'Overview Exponential Regressionmodels', xAx='X', yAx='value',
                       x_label='Time [months]',y_label='Normalized Monthly Performance [%]', hueAx='IndexName')

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 15. Return triangle
# Now a return triangle will be created. With this triangle it shall be possible
# to see, what profit an investor would have been seen in the past for different
# periods of time.

# function to calculate the return triangle
def calculateReturnTriangle(fcn_df_corr, strIndexname = 'WORLD_LMC_NET'):
    print('Start')
    
    # some parameters are calculated (maximum number of months, the shares can be hold)
    fcn_df_corr['MONTH_hold_MAX'] = fcn_df_corr.iloc[:,0].index.values[:]-fcn_df_corr.iloc[:,0].index.values[0]
    fcn_df_corr['MONTH_SELL_IDX'] = fcn_df_corr.iloc[:,0].index.values[:]

    df_triangle_out = fcn_df_corr[['DATE',strIndexname,'MONTH','YEAR','MONTH_hold_MAX','MONTH_SELL_IDX']].copy()
    
    # get columns for each month hold / buy combination
    df_dummy = pd.get_dummies(df_triangle_out["MONTH_hold_MAX"])    
    df_dummy.drop([0], axis=1,inplace=True)
    df_dummy[df_dummy >= 0] = np.nan
    
    df_dummy = df_dummy.iloc[::-1]     
         
    numColumns = df_dummy.shape[1]
    
    # iterate over each column and row and calculate the return rate in defined
    # function calculateReturnRate
    for index, row in df_dummy.iterrows():    
        numColumns = numColumns-1
        for col in range(numColumns,-1,-1):  
            col2 = df_dummy.columns.values[col]                  
            if col2 > 0:
                df_dummy.loc[index,col2] = calculateReturnRate(df_triangle_out.loc[index,strIndexname],
                                                df_triangle_out.loc[index-col2,strIndexname],
                                                col2)
            else:
                df_dummy.loc[index,col2] = 0    
    
    #calculate return rate "per year" instead of "per month"
    df_anual_out = df_dummy.copy()
    df_dummy = df_dummy * 100
    df_anual_out = (((1 + df_anual_out) ** 12)-1)*100
       
    # merge data frame "df_triangle_out" and "df_dummy" 
    df_triangle_out['STR_DATE_SELL'] = df_triangle_out['DATE'].dt.strftime('%Y-%m')
    
    # concatenate dataframes of df_dummy (monthly return rate) and df_triangle_out 
    # (basically the original df for a specific index)
    df_triangle_month_out = pd.concat([df_dummy, df_triangle_out], axis=1)
    # concatenate dataframes of df_anual_out (yearly return rate) and df_triangle_out 
    # (basically the original df for a specific index)
    df_triangle_year_out = pd.concat([df_anual_out, df_triangle_out], axis=1)
    
    # drop unneccessary columns
    df_triangle_month_out = df_triangle_month_out.drop(columns=['MONTH_hold_MAX'])
    df_triangle_year_out = df_triangle_year_out.drop(columns=['MONTH_hold_MAX'])
    
    df_dummy = df_dummy.iloc[::-1]
    
    return df_triangle_month_out, df_triangle_year_out, df_dummy

# function to get melted triangle data for seaborn
def getMeltedTriangleData(fcndf_corr,index_evalname = 'WORLD_LMC_NET'):
    # calculate df_triangle based on index_name with previous function
    df_triangle_month, df_triangle_year, df_values = calculateReturnTriangle(fcndf_corr,index_evalname)
    
    # melt the resulting df_triangle (year) over the "MONTH_hold" column
    melted_df = pd.melt(df_triangle_year, 
                              id_vars=['STR_DATE_SELL','DATE',index_evalname,'MONTH','YEAR','MONTH_SELL_IDX'], # Variables to keep
                              var_name="MONTH_hold") # Name of melted variable
    
    # get the month_number, when the stock share was bought
    melted_df['MONTH_BUY_IDX'] = melted_df['MONTH_SELL_IDX']- melted_df['MONTH_hold']
    
    # set MONTH_BUY_IDX to nan for values lower than the starting month number of the dataset
    melted_df[melted_df['MONTH_BUY_IDX'] < fcndf_corr.iloc[:,0].index.values[0]] = np.nan
    
    # drop nan
    melted_df.dropna(inplace=True)
    
    # get a formated date based on the "MONTH_BUY_IDX" value and "DATE" column
    melted_df['STR_DATE_BUY'] = melted_df['MONTH_BUY_IDX'].apply(
                            lambda x: fcndf_corr.loc[x,'DATE'].strftime('%Y-%m')) 
    
    melted_df = melted_df.iloc[::-1]  
    return melted_df

# --------------------------------------
# get normalized Index Data and set step for months
step_Month = 12

# the max and min limit (+-15) for the color bar in the diagram
# here all values higher than 15% and lower than -15% get the full color
dblColorlimit = 15

# get normalized index data for WORLD_LMC_NET, EM_LMC_NET and WORLD_SC_NET
df_corr = getNormalizedIndexData(df_msciCorrected_Month[
    ['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']],True, step_Month,'01.12.2000')

# ----------------------------------------------------------------------------

# Evaluation for World LMC Net
index_evalname = 'WORLD_LMC_NET'
print(str(eval_no) + '. Return triangle ' + index_evalname)

# calculate the return rate triangle for given index
triangle_df = getMeltedTriangleData(df_corr, index_evalname)

# plot heatmap of return rate triangle 
# color set Red, Yellow, Green
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | RdYlGn','STR_DATE_SELL','STR_DATE_BUY',
            'Selling Date','Buying Date','RdYlGn',-dblColorlimit,dblColorlimit)
# color set Red, Yellow, Blue
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | RdYlBu','STR_DATE_SELL','STR_DATE_BUY',
            'Selling Date','Buying Date','RdYlBu',-dblColorlimit,dblColorlimit)


# Basically the same evaluation as before, but now the triangle will refer to
# holding times (how long did an invester posses the index)
# color set Red, Yellow, Green
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | Months Hold | RdYlGn','STR_DATE_SELL','MONTH_hold',
            'Selling Date','Holding Months','RdYlGn',-dblColorlimit,dblColorlimit)
# color set Red, Yellow, Blue
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | Months Hold | RdYlBu','STR_DATE_SELL','MONTH_hold',
            'Selling Date','Holding Months','RdYlBu',-dblColorlimit,dblColorlimit)

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 16. Return triangle
# Evaluation for EM LMC Net
index_evalname = 'EM_LMC_NET'
print(str(eval_no) + '. Return triangle ' + index_evalname)

# calculate the return rate triangle for given index
triangle_df = getMeltedTriangleData(df_corr, index_evalname)

# plot heatmap of return rate triangle 
# color set Red, Yellow, Green
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | RdYlGn','STR_DATE_SELL','STR_DATE_BUY',
            'Selling Date','Buying Date','RdYlGn',-dblColorlimit,dblColorlimit)
# color set Red, Yellow, Blue
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | RdYlBu','STR_DATE_SELL','STR_DATE_BUY',
            'Selling Date','Buying Date','RdYlBu',-dblColorlimit,dblColorlimit)

# Basically the same evaluation as before, but now the triangle will refer to
# holding times (how long did an invester posses the index)
# color set Red, Yellow, Green
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | Months Hold | RdYlGn','STR_DATE_SELL','MONTH_hold',
            'Selling Date','Holding Months','RdYlGn',-dblColorlimit,dblColorlimit)
# color set Red, Yellow, Blue
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | Months Hold | RdYlBu','STR_DATE_SELL','MONTH_hold',
            'Selling Date','Holding Months','RdYlBu',-dblColorlimit,dblColorlimit)

print('Successfull.')
print(' ')
# ----------------------------------------------------------------------------

eval_no = eval_no + 1
# 17. Return triangle
# Evaluation for WORLD SC NET

index_evalname = 'WORLD_SC_NET'
print(str(eval_no) + '. Return triangle ' + index_evalname)

# calculate the return rate triangle for given index
triangle_df = getMeltedTriangleData(df_corr, index_evalname)

# plot heatmap of return rate triangle 
# color set Red, Yellow, Green
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | RdYlGn','STR_DATE_SELL','STR_DATE_BUY',
            'Selling Date','Buying Date','RdYlGn',-dblColorlimit,dblColorlimit)
# color set Red, Yellow, Blue
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | RdYlBu','STR_DATE_SELL','STR_DATE_BUY',
            'Selling Date','Buying Date','RdYlBu',-dblColorlimit,dblColorlimit)


# Basically the same evaluation as before., but now the triangle will refer to
# holding times (how long did an invester posses the index)
# color set Red, Yellow, Green
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | Months Hold | RdYlGn','STR_DATE_SELL','MONTH_hold',
            'Selling Date','Holding Months','RdYlGn',-dblColorlimit,dblColorlimit)
# color set Red, Yellow, Blue
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | Months Hold | RdYlBu','STR_DATE_SELL','MONTH_hold',
            'Selling Date','Holding Months','RdYlBu',-dblColorlimit,dblColorlimit)


print('Successfull.')
print(' ')

# --------------------------------------

eval_no = eval_no + 1
# 18. Return triangle
# get normalized Index Data and set higher step for months
step_Month = 36
dblColorlimit = 15

df_corr = getNormalizedIndexData(df_msciCorrected_Month[
    ['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']],True, step_Month)

# Evaluation for World LMC Net
index_evalname = 'WORLD_LMC_NET'
print(str(eval_no) + '. Return triangle ' + index_evalname)

# calculate the return rate triangle for given index
triangle_df = getMeltedTriangleData(df_corr, index_evalname)


# plot heatmap of return rate triangle 
# color set Red, Yellow, Green
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | RdYlGn'+' | 36 months step full time','STR_DATE_SELL','STR_DATE_BUY',
            'Selling Date','Buying Date','RdYlGn',-dblColorlimit,dblColorlimit)
# color set Red, Yellow, Blue
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | RdYlBu'+' | 36 months step full time','STR_DATE_SELL','STR_DATE_BUY',
            'Selling Date','Buying Date','RdYlBu',-dblColorlimit,dblColorlimit)


# Basically the same evaluation as before, but now the triangle will refer to
# holding times (how long did an invester posses the index)
# color set Red, Yellow, Green
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | Months Hold | RdYlGn'+' | 36 months step full time','STR_DATE_SELL','MONTH_hold',
            'Selling Date','Holding Months','RdYlGn',-dblColorlimit,dblColorlimit)
# color set Red, Yellow, Blue
plotHeatmap(triangle_df,'Annual Performance Triangle \n '+index_evalname,eval_no,'[% / year] | Months Hold | RdYlBu'+' | 36 months step full time','STR_DATE_SELL','MONTH_hold',
            'Selling Date','Holding Months','RdYlBu',-dblColorlimit,dblColorlimit)

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
print('----------------------------------------------------------------------------')
print(' ')

eval_no = eval_no + 1
# 19. Investment strategies
print('Investment strategies')

# Investment strategy A: Market Timing
print(str(eval_no) + '. A | Market Timing MSCI World LMC')
print(' ')

# 1: Invest completely at start (-->InvStart)
# 2: Invest equaly splitted over Time (-->InvEqual)
# 3: Invest completely after Crash of at least 2% (-->InvCrash2)
# 4: Invest completely after Crash of at least 5% (-->InvCrash5)
# 5: Invest completely after Crash of at least 10% (-->InvCrash10)
# 6: Invest completely after Crash of at least 15% (-->InvCrash15)

############################################################################
# OVERVIEW OVER GENERAL DF STRUCTURE #######################################

#           DATE  WORLD_LMC_NET  EM_LMC_NET  WORLD_SC_NET  MONTH  YEAR  \
# 372 2000-12-29     100.000000  100.000000    100.000000     12  2000   
# 384 2001-12-31      83.006939   97.189070    100.473738     12  2001   
# 396 2002-12-31      66.364421   91.007501     84.088222     12  2002   
# 408 2003-12-31      88.162819  141.530822    132.232930     12  2003   
# 420 2004-12-31     100.941134  177.347542    163.812884     12  2004   

#      MONTH_hold_MAX  MONTH_SELL_IDX  
# 372               0             372  
# 384              12             384  
# 396              24             396  
# 408              36             408  
# 420              48             420  
#           DATE  WORLD_LMC_NET  EM_LMC_NET  WORLD_SC_NET  MONTH  YEAR  \
# 552 2015-12-31     175.760893  331.561711    327.223716     12  2015   
# 564 2016-12-30     188.585586  367.921514    367.540934     12  2016   
# 576 2017-12-29     230.375413  504.109900    449.274877     12  2017   
# 588 2018-12-31     209.882135  429.795460    385.651354     12  2018   
# 600 2019-12-31     267.431246  507.966869    484.982027     12  2019   

#      MONTH_hold_MAX  MONTH_SELL_IDX  
# 552             180             552  
# 564             192             564  
# 576             204             576  
# 588             216             588  
# 600             228             600 

#^^ OVERVIEW OVER GENERAL DF STRUCTURE ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
############################################################################

# function to calculate the capital development incl. relevant crash levels etc.
def calculateCapitalDev(fcn_df_investA, fcn_index_variant, str_variant = 'InvStart', 
                        fcn_investment_capital = 0,
                        fcn_TotalYears = 1,
                        fcn_investment_capital_per_step = 0, 
                        crash_level = 100 / 100):
    old_dbl_index = np.nan
    old_dbl_liquidity = np.nan
    old_dbl_depot = np.nan
    
    # iterratevly calculate liquidity, depot and profit development
    # the values of the previous iteration step for each relevant column 
    # are saved and marked with prefix "old_"    
    for index, row in fcn_df_investA.iterrows():
        dbl_index = row[fcn_index_variant]
        
        if not np.isnan(old_dbl_index):
            
            fcn_df_investA.loc[index, str_variant + '_Depot'] = \
                old_dbl_depot * (dbl_index / old_dbl_index)
            
            
            if old_dbl_liquidity - fcn_investment_capital_per_step >= 0:
                fcn_df_investA.loc[index, str_variant + '_Liquidity'] = \
                    old_dbl_liquidity - fcn_investment_capital_per_step
                    
            else:
                fcn_df_investA.loc[index, str_variant + '_Liquidity'] = 0
            
            # check, whether "crash_level", which was defined by the user,
            # was hit and set liquidity to 0, if so
            if (1-(dbl_index / old_dbl_index)) >= crash_level:
                fcn_df_investA.loc[index, str_variant + '_Liquidity'] = 0

            
            fcn_df_investA.loc[index, str_variant + '_Depot'] = \
                fcn_df_investA.loc[index, str_variant + '_Depot'] \
                + old_dbl_liquidity \
                - fcn_df_investA.loc[index, str_variant + '_Liquidity']
                
            fcn_df_investA.loc[index, str_variant + '_Profit'] = \
                fcn_df_investA.loc[index, str_variant + '_Depot'] \
                + fcn_df_investA.loc[index, str_variant + '_Liquidity'] \
                - fcn_investment_capital
            
            if fcn_TotalYears >= 1:        
                fcn_df_investA.loc[index, str_variant + '_ProfitRate'] = \
                    (((( fcn_df_investA.loc[index, str_variant + '_Profit'] \
                    + fcn_investment_capital) \
                    / fcn_investment_capital) \
                    ** (1 / fcn_TotalYears)) \
                    - 1) * 100
            else:
                fcn_df_investA.loc[index, str_variant + '_ProfitRate'] = 0
            
        old_dbl_index = dbl_index
        old_dbl_liquidity = fcn_df_investA.loc[index, str_variant + '_Liquidity']
        old_dbl_depot= fcn_df_investA.loc[index, str_variant + '_Depot']
        
    return fcn_df_investA

# function to calculate liquidity, cost and profit development for previous
# described scenarios
def calculateMarketTiming(starting_Month = 1,
                          starting_year = 2000,
                          step_Month = 3,
                          investment_capital = 10000,
                          index_variant = 'WORLD_LMC_NET'):
        
    fcn_df_investA = getNormalizedIndexData(df_msciCorrected_Month[
        ['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']],True, step_Month,'01.'+ str(starting_Month) +'.'+str(starting_year))
    
    # ensure, that plot will not show more than about 20 sections
    # all evaluations are performed until the year 2020
    # this should be automated to prevent error or to change the
    # time period for other analyses (e.g. 2000 until 2010)
    # In this evaluation there will be no focus on shorter or longer
    # periods
    
    # this function sets the step width for each row (for the following)
    # plot
    # It shall prevent, that there aren't too many bars, so that the graph
    # will be readable.
    if 2020-starting_year > 60:
        rowStep = 4
    elif 2020-starting_year > 40:
        rowStep = 3
    elif 2020-starting_year > 20:
        rowStep = 2
    elif 2020-starting_year <= 20:
        rowStep = 1
    
    # total number of rows
    TotalRowsIndex = range(0,2020-starting_year,rowStep)
    
    for rowIndex in TotalRowsIndex:
        current_year = starting_year + rowIndex
        df_investA2 = getNormalizedIndexData(df_msciCorrected_Month[
        ['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']],True, step_Month,'01.'+ str(starting_Month) +'.'+str(current_year))
        
        investment_capital_per_step = investment_capital / df_investA2.shape[0]
        
        # create columns for basic scenarios
        df_investA2['InvStart_'+str(current_year)+'_Liquidity'] = 0
        df_investA2['InvStart_'+str(current_year)+'_Depot'] = investment_capital
        df_investA2['InvStart_'+str(current_year)+'_Profit'] = 0
        df_investA2['InvStart_'+str(current_year)+'_ProfitRate'] = 0
        
        df_investA2['InvEqual_'+str(current_year)+'_Liquidity'] = investment_capital - investment_capital_per_step
        df_investA2['InvEqual_'+str(current_year)+'_Depot'] = investment_capital_per_step
        df_investA2['InvEqual_'+str(current_year)+'_Profit'] = 0
        df_investA2['InvEqual_'+str(current_year)+'_ProfitRate'] = 0
        
        df_investA2['InvCrash2_'+str(current_year)+'_Liquidity'] = investment_capital
        df_investA2['InvCrash2_'+str(current_year)+'_Depot'] = 0
        df_investA2['InvCrash2_'+str(current_year)+'_Profit'] = 0
        df_investA2['InvCrash2_'+str(current_year)+'_ProfitRate'] = 0
        
        df_investA2['InvCrash5_'+str(current_year)+'_Liquidity'] = investment_capital
        df_investA2['InvCrash5_'+str(current_year)+'_Depot'] = 0
        df_investA2['InvCrash5_'+str(current_year)+'_Profit'] = 0
        df_investA2['InvCrash5_'+str(current_year)+'_ProfitRate'] = 0
        
        df_investA2['InvCrash10_'+str(current_year)+'_Liquidity'] = investment_capital
        df_investA2['InvCrash10_'+str(current_year)+'_Depot'] = 0
        df_investA2['InvCrash10_'+str(current_year)+'_Profit'] = 0
        df_investA2['InvCrash10_'+str(current_year)+'_ProfitRate'] = 0
        
        df_investA2['InvCrash15_'+str(current_year)+'_Liquidity'] = investment_capital
        df_investA2['InvCrash15_'+str(current_year)+'_Depot'] = 0
        df_investA2['InvCrash15_'+str(current_year)+'_Profit'] = 0
        df_investA2['InvCrash15_'+str(current_year)+'_ProfitRate'] = 0
           
        TotalMonth = (max(TotalRowsIndex) - rowIndex) + 1
        
        # run function calculateCapitalDev for given scenarios
        df_investA2 = calculateCapitalDev(df_investA2, index_variant,'InvStart_'+str(current_year), 
                                          investment_capital, TotalMonth)        
        df_investA2 = calculateCapitalDev(df_investA2, index_variant,'InvEqual_'+str(current_year), 
                                          investment_capital, TotalMonth, 
                                          investment_capital_per_step)
        df_investA2 = calculateCapitalDev(df_investA2, index_variant,'InvCrash2_'+str(current_year), 
                                          investment_capital, TotalMonth, 
                                          0, 2 / 100) 
        df_investA2 = calculateCapitalDev(df_investA2, index_variant,'InvCrash5_'+str(current_year), 
                                          investment_capital, TotalMonth,
                                          0, 5 / 100) 
        df_investA2 = calculateCapitalDev(df_investA2, index_variant,'InvCrash10_'+str(current_year), 
                                          investment_capital, TotalMonth,
                                          0, 10 / 100) 
        df_investA2 = calculateCapitalDev(df_investA2, index_variant,'InvCrash15_'+str(current_year), 
                                          investment_capital, TotalMonth,
                                          0, 15 / 100)   
       
        df_investA2 = df_investA2.drop(columns=['DATE', 'WORLD_LMC_NET', 'EM_LMC_NET', 'WORLD_SC_NET', 'MONTH', 'YEAR'])
        fcn_df_investA = pd.concat([fcn_df_investA, df_investA2], axis=1)

    # merge data to dataframe    
    fcn_df_merged_mt = pd.melt(fcn_df_investA.tail(1), 
                              id_vars=['DATE', 'WORLD_LMC_NET', 'EM_LMC_NET', 'WORLD_SC_NET', 'MONTH', 'YEAR'], # Variables to keep
                              var_name="InvestmentTypeFull") # Name of melted variable
    fcn_df_merged_mt[['InvestmentType','StartingYear','FinancialPosition']] = list(map(lambda x:x.split('_'),fcn_df_merged_mt['InvestmentTypeFull']))
    return fcn_df_merged_mt

# function to plot Market Timing figures using seaborn
def plotMarketTiming(fcn_df_merged_mt, fcn_eval_no, fcn_starting_year,
                      fcn_step_month, fcn_capital, fcn_index_name):
    # plot realized profit in groups (bar graph)
    plotGroupedBarSeaborn(fcn_df_merged_mt[fcn_df_merged_mt['FinancialPosition']=='Profit'],
                          'Market Timing Variants | Cumulated Capital | Bargraph',
                          fcn_eval_no,
                          fcn_index_name
                          + '\nInitial year: ' + fcn_starting_year
                          + ' | Calculation Step: ' + fcn_step_month + ' [months]'
                          + ' | Start capital: ' + fcn_capital + ' [USD]',
                          'StartingYear','value',
                          'InvestmentType', 'Starting Year','Cumulated Capital [USD]',True)

    # plot profit rate in groups (bar graph)
    plotGroupedBarSeaborn(fcn_df_merged_mt[fcn_df_merged_mt['FinancialPosition']=='ProfitRate'],
                          'Market Timing Variants | Average annual performance [%] | Bargraph',
                          fcn_eval_no,
                          fcn_index_name
                          + '\nInitial year: ' + fcn_starting_year
                          + ' | Calculation Step: ' + fcn_step_month + ' [months]'
                          + ' | Start capital: ' + fcn_capital + ' [USD]',                       
                          'StartingYear','value',
                          'InvestmentType', 'Starting Year','Average annual performance [%]',True)
   
    # plot realized profit in groups (boxplot graph)
    plotBoxplotSeaborn(fcn_df_merged_mt[fcn_df_merged_mt['FinancialPosition']=='Profit'],
                          'Market Timing Variants | Cumulated Capital | Boxplot',
                          fcn_eval_no,
                          fcn_index_name
                          + '\nInitial year: ' + fcn_starting_year
                          + ' | Calculation Step: ' + fcn_step_month + ' [months]'
                          + ' | Start capital: ' + fcn_capital + ' [USD]',
                          'InvestmentType','value',
                          'Investment Variant','Cumulated Capital [USD]') 
    
    plotViolinSwarmSeaborn(fcn_df_merged_mt[fcn_df_merged_mt['FinancialPosition']=='Profit'],
                          'Market Timing Variants | Cumulated Capital | ViolinSwarm',
                          fcn_eval_no,
                          fcn_index_name
                          + '\nInitial year: ' + fcn_starting_year
                          + ' | Calculation Step: ' + fcn_step_month + ' [months]'
                          + ' | Start capital: ' + fcn_capital + ' [USD]',
                          'InvestmentType','value',
                          'Investment Variant','Cumulated Capital [USD]') 
    
    # plot profit rate in groups (boxplot graph)
    plotBoxplotSeaborn(fcn_df_merged_mt[fcn_df_merged_mt['FinancialPosition']=='ProfitRate'],
                          'Market Timing Variants | Average annual performance [%] | Boxplot',
                          fcn_eval_no,
                          fcn_index_name
                          + '\nInitial year: ' + fcn_starting_year
                          + ' | Calculation Step: ' + fcn_step_month + ' [months]'
                          + ' | Capital: ' + fcn_capital + ' [USD]',
                          'InvestmentType','value',
                          'Investment Variant','Average annual performance [%]')
    
    plotViolinSwarmSeaborn(fcn_df_merged_mt[fcn_df_merged_mt['FinancialPosition']=='ProfitRate'],
                          'Market Timing Variants | Average annual performance [%] | ViolinSwarm',
                          fcn_eval_no,
                          fcn_index_name
                          + '\nInitial year: ' + fcn_starting_year
                          + ' | Calculation Step: ' + fcn_step_month + ' [months]'
                          + ' | Capital: ' + fcn_capital + ' [USD]',
                          'InvestmentType','value',
                          'Investment Variant','Average annual performance [%]')

# ----------------------------------------------------------------------------
# 19. Investment strategies
# setup basic parameters, which can be changed individually    
starting_month = 12
starting_year = 1969
step_month = 1
capital = 10000
index_name = 'WORLD_LMC_NET'

# calculate the effects of market timing based on previous parameters
df_merged_mt = calculateMarketTiming(starting_month, starting_year,
                                      step_month, capital,
                                      index_name)

# plot the graphs for previous set parameters
plotMarketTiming(df_merged_mt,eval_no,str(starting_year),str(step_month),
                  str(capital),index_name)

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
# 20. Investment strategies
eval_no = eval_no + 1
print(str(eval_no) + '. A | Market Timing | 2000')
print(' ')

# setup basic parameters, which can be changed individually  
starting_month = 12
starting_year = 2000
step_month = 1
capital = 10000
index_name = 'WORLD_LMC_NET'

# calculate the effects of market timing based on previous parameters
df_merged_mt = calculateMarketTiming(starting_month, starting_year,
                                      step_month, capital,
                                      index_name)

# plot the graphs for previous set parameters
plotMarketTiming(df_merged_mt,eval_no,str(starting_year),str(step_month),
                  str(capital),index_name)

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
# 21. Investment strategies
eval_no = eval_no + 1
print(str(eval_no) + '. A | Market Timing | 2000 | EM LMC')
print(' ')

# setup basic parameters, which can be changed individually  
starting_month = 12
starting_year = 2000
step_month = 1
capital = 10000
index_name = 'EM_LMC_NET'

# calculate the effects of market timing based on previous parameters
df_merged_mt = calculateMarketTiming(starting_month, starting_year,
                                      step_month, capital,
                                      index_name)

# plot the graphs for previous set parameters
plotMarketTiming(df_merged_mt,eval_no,str(starting_year),str(step_month),
                  str(capital),index_name)

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
# 22. Investment strategies
eval_no = eval_no + 1
print(str(eval_no) + '. A | Market Timing | 2000 | WORLD SC')
print(' ')

# setup basic parameters, which can be changed individually  
starting_month = 12
starting_year = 2000
step_month = 1
capital = 10000
index_name = 'WORLD_SC_NET'

# calculate the effects of market timing based on previous parameters
df_merged_mt = calculateMarketTiming(starting_month, starting_year,
                                      step_month, capital,
                                      index_name)

# plot the graphs for previous set parameters
plotMarketTiming(df_merged_mt,eval_no,str(starting_year),str(step_month),
                  str(capital),index_name)

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
############################################################################
# Investment strategy B: Rebalancing effects #######################################
# 

dict_RebalancingResult = {"Label":[],"SUM_DEPOT":[], "ProfitRebalancing":[],
                          "ProfitRebalancingSteps":[], "Freq_World_LMC":[],
                          "Freq_EM_LMC":[], "Freq_World_SC":[],
                          "Reb_Costs_Abs":[], "Reb_Costs_Perc":[],
                          "CostVariant":[], "Capital":[],
                          "StartYear":[], "Step":[]};

# function to calculate the rebalancing effects based on portfolio and costs    
def calculateRebalancingDepot(df_investB2,fcn_step_month = 1,
                              fcn_frequency_World_LMC = 50 / 100, 
                              fcn_frequency_EM_LMC = 30 / 100, 
                              fcn_frequency_World_SC = 20 / 100,
                              fcn_rebalance_costs_abs = 1.5,
                              fcn_rebalance_costs_percentage = 1.5 / 100):
    
    # iterate rows to do the calculation
    # check for each step, whether the portfolio has to be changed and 
    # costs have to be applied
    for index, row in df_investB2.iterrows():         
        if index > df_investB2.index[0]:
            depot_WORLD_LMC_NET = \
                df_investB2.loc[index,'WORLD_LMC_NET'] \
                / df_investB2.loc[index-fcn_step_month,'WORLD_LMC_NET'] \
                * df_investB2.loc[index-fcn_step_month,'DEPOT_WORLD_LMC_NET']             
            
            depot_EM_LMC_NET = \
                df_investB2.loc[index,'EM_LMC_NET'] \
                / df_investB2.loc[index-fcn_step_month,'EM_LMC_NET'] \
                * df_investB2.loc[index-fcn_step_month,'DEPOT_EM_LMC_NET'] 

            depot_WORLD_SC_NET = \
                df_investB2.loc[index,'WORLD_SC_NET'] \
                / df_investB2.loc[index-fcn_step_month,'WORLD_SC_NET'] \
                * df_investB2.loc[index-fcn_step_month,'DEPOT_WORLD_SC_NET'] 

            sum_depot = depot_WORLD_LMC_NET + depot_EM_LMC_NET + depot_WORLD_SC_NET
            df_investB2.loc[index,'DEPOT_WORLD_LMC_NET'] = depot_WORLD_LMC_NET
            df_investB2.loc[index,'DEPOT_EM_LMC_NET'] = depot_EM_LMC_NET
            df_investB2.loc[index,'DEPOT_WORLD_SC_NET'] = depot_WORLD_SC_NET
            df_investB2.loc[index,'SUM_DEPOT'] = sum_depot
            
            depot_rebalanced_WORLD_LMC_NET = \
                df_investB2.loc[index,'WORLD_LMC_NET'] \
                / df_investB2.loc[index-fcn_step_month,'WORLD_LMC_NET'] \
                * df_investB2.loc[index-fcn_step_month,'DEPOT_rebalanced_WORLD_LMC_NET']             
            
            depot_rebalanced_EM_LMC_NET = \
                df_investB2.loc[index,'EM_LMC_NET'] \
                / df_investB2.loc[index-fcn_step_month,'EM_LMC_NET'] \
                * df_investB2.loc[index-fcn_step_month,'DEPOT_rebalanced_EM_LMC_NET'] 

            depot_rebalanced_WORLD_SC_NET = \
                df_investB2.loc[index,'WORLD_SC_NET'] \
                / df_investB2.loc[index-fcn_step_month,'WORLD_SC_NET'] \
                * df_investB2.loc[index-fcn_step_month,'DEPOT_rebalanced_WORLD_SC_NET']
            
            sum_rebalanced_depot = depot_rebalanced_WORLD_LMC_NET + depot_rebalanced_EM_LMC_NET + depot_rebalanced_WORLD_SC_NET
            df_investB2.loc[index,'DEPOT_rebalanced_WORLD_LMC_NET'] = sum_rebalanced_depot * fcn_frequency_World_LMC
            df_investB2.loc[index,'DEPOT_rebalanced_EM_LMC_NET'] = sum_rebalanced_depot * fcn_frequency_EM_LMC
            df_investB2.loc[index,'DEPOT_rebalanced_WORLD_SC_NET'] = sum_rebalanced_depot * fcn_frequency_World_SC
            df_investB2.loc[index,'SUM_rebalanced_DEPOT'] = sum_rebalanced_depot
            
            costs_abs = 0
            if sum_rebalanced_depot * fcn_frequency_World_LMC != depot_rebalanced_WORLD_LMC_NET:
                costs_abs =  costs_abs + fcn_rebalance_costs_abs          
            
            if sum_rebalanced_depot * fcn_frequency_EM_LMC != depot_rebalanced_EM_LMC_NET:
                costs_abs =  costs_abs + fcn_rebalance_costs_abs  
            
            if sum_rebalanced_depot * fcn_frequency_World_SC != depot_rebalanced_WORLD_SC_NET:
                costs_abs =  costs_abs + fcn_rebalance_costs_abs  
            
            df_investB2.loc[index,'SUM_DEPOT_COSTS_ABS'] = df_investB2.loc[index-fcn_step_month,'SUM_DEPOT_COSTS_ABS'] + costs_abs
            
            moved_money = \
                abs(depot_rebalanced_WORLD_LMC_NET - sum_rebalanced_depot * fcn_frequency_World_LMC) \
                + abs(depot_rebalanced_EM_LMC_NET - sum_rebalanced_depot * fcn_frequency_EM_LMC) \
                + abs(depot_rebalanced_WORLD_SC_NET - sum_rebalanced_depot * fcn_frequency_World_SC)
            
            df_investB2.loc[index,'SUM_DEPOT_COSTS_PERC'] = df_investB2.loc[index-fcn_step_month,'SUM_DEPOT_COSTS_PERC'] + moved_money * fcn_rebalance_costs_percentage
            df_investB2.loc[index,'SUM_REBALANCED_CAPITAL'] = df_investB2.loc[index-fcn_step_month,'SUM_REBALANCED_CAPITAL'] + moved_money    
                        
    return df_investB2

# function to calculate the df including the rebalancing effects
# this function calls the previous function "calculateRebalancingDepot"
def calculateRebalancingDF(fcn_dict_RebalancingResult,
                          starting_Month = 1,
                          starting_year = 2000,
                          step_Month = 3,
                          investment_capital = 10000,
                          fcn_frequency_World_LMC = 50 / 100,
                          fcn_frequency_EM_LMC = 30 / 100,
                          fcn_frequency_World_SC = 20 / 100,    
                          fcn_rebalance_costs_abs = 1.50,
                          fcn_rebalance_costs_percentage = 1.5 / 100,
                          fcn_consider_costs = ''):
    
    # get normalized Index Data for all three relevant indexes
    df_investB1 = getNormalizedIndexData(df_msciCorrected_Month[ 
        ['DATE','WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']],
        True, step_Month,'01.'+ str(starting_Month) +'.'+str(starting_year))
    # drop nan rows
    df_investB1.dropna(axis=0,inplace=True)
    
    # initialize relevant columns with value "0"
    df_investB1[['DEPOT_WORLD_LMC_NET',
                  'DEPOT_EM_LMC_NET',
                  'DEPOT_WORLD_SC_NET',
                  'SUM_DEPOT',
                  'DEPOT_rebalanced_WORLD_LMC_NET',
                  'DEPOT_rebalanced_EM_LMC_NET',
                  'DEPOT_rebalanced_WORLD_SC_NET',
                  'SUM_rebalanced_DEPOT',
                  'SUM_DEPOT_COSTS_ABS',
                  'SUM_DEPOT_COSTS_PERC',
                  'SUM_REBALANCED_CAPITAL']]= 0
       
    # set initial index and investment values depending on chosen parameters
    start_index = df_investB1.index[0]
    df_investB1.loc[start_index,'DEPOT_WORLD_LMC_NET'] = \
        investment_capital * fcn_frequency_World_LMC
          
    df_investB1.loc[start_index,'DEPOT_EM_LMC_NET'] = \
        investment_capital * fcn_frequency_EM_LMC
        
    df_investB1.loc[start_index,'DEPOT_WORLD_SC_NET'] = \
        investment_capital * fcn_frequency_World_SC
    
    df_investB1.loc[start_index,'SUM_DEPOT'] = \
        investment_capital
    
    # ------------------------
    
    df_investB1.loc[start_index,'DEPOT_rebalanced_WORLD_LMC_NET'] = \
        investment_capital * fcn_frequency_World_LMC
          
    df_investB1.loc[start_index,'DEPOT_rebalanced_EM_LMC_NET'] = \
        investment_capital * fcn_frequency_EM_LMC
        
    df_investB1.loc[start_index,'DEPOT_rebalanced_WORLD_SC_NET'] = \
        investment_capital * fcn_frequency_World_SC
    
    df_investB1.loc[start_index,'SUM_rebalanced_DEPOT'] = \
        investment_capital
        
    # ------------------------
    
    # perform rebalancing calculation
    df_investB1 = calculateRebalancingDepot(df_investB1, step_Month,
                        fcn_frequency_World_LMC, 
                        fcn_frequency_EM_LMC, 
                        fcn_frequency_World_SC,
                        fcn_rebalance_costs_abs,
                        fcn_rebalance_costs_percentage)
    
    # 100% are shown in the index as number = 100. Therefor this index has
    # to be divided by 100
    df_investB1[['WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']] = \
        df_investB1[['WORLD_LMC_NET','EM_LMC_NET','WORLD_SC_NET']] \
        * investment_capital / 100
    
    # depending on the chosen cost setup the corresponding column gets subtracted
    str_costs = ' | no rebalancing costs'
    str_channel = 'SUM_DEPOT_COSTS_ABS'
    if fcn_consider_costs == 'abs':
        df_investB1['SUM_rebalanced_DEPOT'] = df_investB1['SUM_rebalanced_DEPOT'] - df_investB1['SUM_DEPOT_COSTS_ABS']
        str_costs = ' | costs absolute = ' + str(fcn_rebalance_costs_abs)
        str_channel = 'SUM_DEPOT_COSTS_ABS'
    elif fcn_consider_costs == 'perc':
        df_investB1['SUM_rebalanced_DEPOT'] = df_investB1['SUM_rebalanced_DEPOT'] - df_investB1['SUM_DEPOT_COSTS_PERC']
        str_costs = ' | costs percentage = ' + str(fcn_rebalance_costs_percentage*100) + ' %'
        str_channel = 'SUM_DEPOT_COSTS_PERC'
        
    # plot line chart for depot, rebalanced depot and the single indices,
    # including thedepot costs and the rebalanced amount of money ("moved" money)
    plotLineSeaborn(df_investB1[['DATE','SUM_DEPOT','SUM_rebalanced_DEPOT',
                                  'WORLD_LMC_NET', 'EM_LMC_NET', 
                                  'WORLD_SC_NET',
                                  str_channel,
                                  'SUM_REBALANCED_CAPITAL'
                              ]],
                'Rebalancing effects for portfolio',eval_no,
                'Initial year: ' + str(starting_year)
                + ' | Rebalancing Step: ' + str(step_Month) + ' [months]'
                + ' | Investment Capital: ' + str(investment_capital) + ' [USD]'
                + str_costs
                + '\n Frequency World LMC / EM LMC / World SC: ' 
                + str(fcn_frequency_World_LMC*100) + "%"
                + ' / ' + str(fcn_frequency_EM_LMC*100) + "%"
                + ' / ' + str(fcn_frequency_World_SC*100)  + "%"               
                ,'DATE',
                'value','Date','Cumulated Capital in Portfolio [USD]','variable','PRGn_r') 
    
    str_label = 'Rebalanced Depot - Hold Depot: ' \
                + ' | Initial year: ' + str(starting_year) \
                + ' | Rebalancing Step: ' + str(step_Month) + ' [months]' \
                + ' | Investment Capital: ' + str(investment_capital) + ' [USD]' \
                + ' | Frequency World LMC / EM LMC / World SC: '  \
                + str(fcn_frequency_World_LMC*100) \
                + ' / ' + str(fcn_frequency_EM_LMC*100) \
                + ' / ' + str(fcn_frequency_World_SC*100) \
                + str_costs
    
    # get the difference of the normal depot vs. the rebalanced depot
    df_investB1['DIFF_REBALANCED_DEPOT'] = df_investB1['SUM_rebalanced_DEPOT'] - df_investB1['SUM_DEPOT']
    
    # plot the "DIFF_REBALANCED_DEPOT" column vs. the absolute costs
    plotLineSeaborn(df_investB1[['DATE','DIFF_REBALANCED_DEPOT',
                                  str_channel
                              ]],
                'Rebalanced Depot - Hold Depot'               
                ,eval_no,
                'Initial year: ' + str(starting_year)
                + ' | Rebalancing Step: ' + str(step_Month) + ' [months]'
                + ' | Investment Capital: ' + str(investment_capital) + ' [USD]'
                + str_costs
                + '\n Frequency World LMC / EM LMC / World SC: '  \
                + str(fcn_frequency_World_LMC*100) + "%" \
                + ' / ' + str(fcn_frequency_EM_LMC*100) + "%" \
                + ' / ' + str(fcn_frequency_World_SC*100) + "%"
                ,
                'DATE',
                'value','Date','Profit by rebalancing [USD]','variable','PRGn_r') 
    
    # save relevant values to result df "fcn_dict_RebalancingResult", which will
    # be returned by the function optionally
    fcn_dict_RebalancingResult["Label"].append(str_label)
    fcn_dict_RebalancingResult["SUM_DEPOT"].append(df_investB1.iloc[-1]['SUM_DEPOT']-investment_capital)
    fcn_dict_RebalancingResult["ProfitRebalancing"].append(df_investB1.iloc[-1]['SUM_rebalanced_DEPOT']-df_investB1.iloc[0]['SUM_rebalanced_DEPOT'])
    fcn_dict_RebalancingResult["ProfitRebalancingSteps"].append(df_investB1.iloc[-1]['DIFF_REBALANCED_DEPOT'])
    fcn_dict_RebalancingResult["Freq_World_LMC"].append(fcn_frequency_World_LMC*100)
    fcn_dict_RebalancingResult["Freq_EM_LMC"].append(fcn_frequency_EM_LMC*100)
    fcn_dict_RebalancingResult["Freq_World_SC"].append(fcn_frequency_World_SC*100)
    fcn_dict_RebalancingResult["Reb_Costs_Abs"].append(fcn_rebalance_costs_abs)
    fcn_dict_RebalancingResult["Reb_Costs_Perc"].append(fcn_rebalance_costs_percentage)	
    fcn_dict_RebalancingResult["CostVariant"].append(fcn_consider_costs)	
    fcn_dict_RebalancingResult["Capital"].append(investment_capital)	
    fcn_dict_RebalancingResult["StartYear"].append(starting_year)
    fcn_dict_RebalancingResult["Step"].append(step_Month)
    #print(fcn_dict_RebalancingResult)
    
    return df_investB1, fcn_dict_RebalancingResult



starting_year_eval = 2000
starting_month_eval = 12
investment_capital = 10000

rebalancing_steps = np.array([1, 3, 6, 12, 24, 36, 48, 60, 120])

portfolio_frequencies = np.array([[50 / 100, 30 / 100, 20 / 100],
                         [80 / 100, 15 / 100,  5 / 100],
                         [30 / 100, 30 / 100, 40 / 100],
                         [15 / 100, 80 / 100, 5 / 100],
                         [15 / 100, 5 / 100, 80 / 100]])
cost_variants = np.array(['','abs','perc'])
costs_abs = 1.50
costs_perc = 1.50 / 100

for i_port_freq in portfolio_frequencies:
    # 23. Evaluation of Rebalancing
    eval_no = eval_no + 1
    print(str(eval_no) + '. Evaluation of Rebalancing')
    print(i_port_freq)
    
    for i_rebalsteps in rebalancing_steps: 
        for i_cost in cost_variants:
            
            # calculate the rebalancing effects based on the set parameters within
            # the paranthesis
            [df_investB3, dict_RebalancingResult] = calculateRebalancingDF( dict_RebalancingResult,
                starting_month_eval, starting_year_eval, i_rebalsteps, investment_capital, i_port_freq[0], i_port_freq[1], i_port_freq[2], costs_abs, costs_perc, i_cost)
        print(str(i_rebalsteps) + '...')    
        
    print('Successfull.')
    print('------------------------------------')
    print(' ')


# ----------------------------------------------------------------------------
# Summary of Rebalancing Evaluation
eval_no = eval_no + 1
print(str(eval_no) + '. Summary of Rebalancing')

# calculate the rebalancing effects based on the set parameters within
# the paranthesis
df_return_Profit_data = pd.DataFrame.from_dict(dict_RebalancingResult)
print(df_return_Profit_data.head(10))


# plot result df_return_Profit_data
plotBoxplotSeaborn(df_return_Profit_data,
                    'Profit due to regular rebalancing',
                    eval_no,'including All Rebalancing Cost Variants | over Cost Variant','CostVariant','ProfitRebalancingSteps',
                    'Cost Variant','Profit [USD]')
plotBoxplotSeaborn(df_return_Profit_data,
                    'Accumulated Profit | Cost Variant',
                    eval_no,'including All Rebalancing Cost Variants | over Cost Variant','CostVariant','ProfitRebalancing',
                    'Cost Variant','Accumulated Profit [USD]')
plotBoxplotSeaborn(df_return_Profit_data,
                    'Profit due to regular rebalancing',
                    eval_no,'including All Rebalancing Cost Variants | over Rebalancing interval','Step','ProfitRebalancingSteps',
                    'Rebalancing Interval [months]','Profit [USD]')
plotBoxplotSeaborn(df_return_Profit_data,
                    'Accumulated Profit | Rebalancing interval',
                    eval_no,'including All Rebalancing Cost Variants | over Rebalancing interval','Step','ProfitRebalancing',
                    'Rebalancing Interval [months]','Accumulated Profit [USD]')
print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
eval_no = eval_no + 1
print(str(eval_no) + '. Summary 2 of Rebalancing')
plotBoxplotSeaborn(df_return_Profit_data[df_return_Profit_data['CostVariant']==''],
                    'Profit due to regular rebalancing',
                    eval_no,'No Rebalancing Costs | over Rebalancing interval','Step','ProfitRebalancingSteps',
                    'Rebalancing Interval [months]','Profit [USD]')

plotBoxplotSeaborn(df_return_Profit_data[df_return_Profit_data['CostVariant']=='abs'],
                    'Profit due to regular rebalancing',
                    eval_no,'Absolute Rebalancing Costs = 1.5USD | over Rebalancing interval','Step','ProfitRebalancingSteps',
                    'Rebalancing Interval [months]','Profit [USD]')

plotBoxplotSeaborn(df_return_Profit_data[df_return_Profit_data['CostVariant']=='perc'],
                    'Profit due to regular rebalancing',
                    eval_no,'Percentual Rebalancing Costs = 1.5% | over Rebalancing interval','Step','ProfitRebalancingSteps',
                    'Rebalancing Interval [months]','Profit [USD]')

print('Successfull.')
print(' ')

# ----------------------------------------------------------------------------
print('Script successfull.')
print(' ')