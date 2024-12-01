#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''


import sys
# sys.path.append(r"H:\20220330PredictExp\0Code\3TimeSformer")
sys.path.append(r"C:\code20140117\transformer")
#sys.path.pop( )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from torch.autograd import Variable
from transformer_raw import SpaceTimeTransformer,TimesFormer_2input
from config import configs
import pickle
from datasets import read_data,ImageDataset_array,ImageDataset_array_test
import h5py,timeit,time
from math import *
import csv
import scipy.ndimage
import os.path
from osgeo import gdalconst
# from gdalconst import *
from osgeo import gdal
from osgeo import osr
import numpy as np
import scipy
import h5py
import numpy.ma as ma
# from sklearn import preprocessing
import pandas as pd
from numpy import float32
import matplotlib.pyplot as plt
from scipy import stats
#basic method
def WriteGTiffFile(filename, nRows, nCols, data,geotrans,proj, noDataValue, gdalType):#向磁盘写入结果文件
    format = "GTiff"   
    driver = gdal.GetDriverByName(format)
    ds = driver.Create(filename, nCols, nRows, 1, gdalType, options=["COMPRESS=LZW"])
    ds.SetGeoTransform(geotrans)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).SetNoDataValue(noDataValue)
    ds.GetRasterBand(1).WriteArray(data)    
    ds = None
def Readxy(RasterFile): #读取每个图像的信息     
    ds = gdal.Open(RasterFile,gdal.GA_ReadOnly)
    if ds is None:
        print ('Cannot open ',RasterFile)
        sys.exit(1)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray(0,0,cols,rows)
    noDataValue = band.GetNoDataValue()
    projection=ds.GetProjection()
    geotransform = ds.GetGeoTransform()
    return rows,cols,geotransform,projection,noDataValue

def Read(RasterFile):#读取每个图像的信息
    ds = gdal.Open(RasterFile,gdal.GA_ReadOnly)    
    if ds is None:
        print ('Cannot open ',RasterFile)
        sys.exit(1)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray(0,0,cols,rows)  
    return data    
def read_samples0805(rootdir):   
    #读各X基本信息
    rows,cols,geotransform,projection,nDV_dem=Readxy('f://NORM_II//DEM_USA_II//m_SRTM_GTOPO_u30_mosaic.tif') #-9999
    rows,cols,geotransform,projection,nDV_hwsd=Readxy('f://NORM_II//HWSD_USA_II//m_T_CLAY.tif') #0
    # rows,cols,geotransform,projection,nDV_imerg=Readxy('f://NORM_II//IMERG_USA_II//m_3B-DAY.MS.MRG.3IMERG.20180101.tif') #-9999
    rows,cols,geotransform,projection,nDV_imerg=Readxy('f://NORM_II//IV_USA_III//m_st4.2018010112.24h.tif') #-9999    
    rows,cols,geotransform,projection,nDV_luc=Readxy('f://NORM_II//MCD12C1_USA_II//m_lc_2018_o.tif') #0
    rows,cols,geotransform,projection,nDV_lst=Readxy('f://NORM_II//MCD11A1_USA_II_ST//MCD11A1_2018001.tif') #-9999
    rows,cols,geotransform,projection,nDV_opt=Readxy('f://NORM_II//MCD43A4_USA_II_ST//band1//MCD43A4_2018001.tif') #32767  
    tiles=['h08v04','h08v05','h08v06','h09v04','h09v05','h09v06','h10v04','h10v05','h10v06','h11v04','h11v05','h12v04','h12v05','h13v04']
    file_demo = [[0.0]*cols]*rows 
    file_demo = np.array(file_demo)
    for t in range(0,len(tiles)):
        file_demo_temp = [[0.0]*cols]*rows 
        file_demo_temp = np.array(file_demo_temp)
        file_demo_temp = Read('f://DEMO//'+tiles[t]+'.tif')
        file_demo[file_demo_temp==1]=1
        print(np.sum(file_demo))
    row_v = []
    col_v = []
    for i in range(0,rows):
        for j in range(0,cols):
            if file_demo[i][j]==1:
                row_v.append(i)
                col_v.append(j)
    row_v = np.array(row_v)
    col_v = np.array(col_v)
    row_min = np.min(row_v)
    row_max = np.max(row_v)
    col_min = np.min(col_v)
    col_max = np.max(col_v)
    
    print('row range',row_min,row_max,'col range',col_min,col_max)
    if (row_max-row_min)%4 !=0:
        row_max += 4-((row_max-row_min)%4)
    if (col_max-col_min)%4 !=0:
        col_max += 4-((col_max-col_min)%4)
    print('new row range',row_min,row_max,'new col range',col_min,col_max)
    #读静态变量
    file_dem = [[0.0]*cols]*rows #栅格值和，二维数组
    file_dem = np.array(file_dem)
    file_dem = Read('f://NORM_II//DEM_USA_II//m_SRTM_GTOPO_u30_mosaic.tif')
    file_lat = [[0.0]*cols]*rows #栅格值和，二维数组
    file_lat = np.array(file_lat)
    file_lat = Read('f://NORM_II//DEM_USA_II//LAT.tif')
    file_lon = [[0.0]*cols]*rows #栅格值和，二维数组
    file_lon = np.array(file_lon)
    file_lon = Read('f://NORM_II//DEM_USA_II//LON.tif')
    file_cla = [[0.0]*cols]*rows #栅格值和，二维数组
    file_cla = np.array(file_cla)
    file_cla = Read('f://NORM_II//HWSD_USA_II//m_T_CLAY.tif')
    file_gra = [[0.0]*cols]*rows #栅格值和，二维数组
    file_gra = np.array(file_gra)
    file_gra = Read('f://NORM_II//HWSD_USA_II//m_T_GRAVEL.tif')
    file_bul = [[0.0]*cols]*rows #栅格值和，二维数组
    file_bul = np.array(file_bul)
    file_bul = Read('f://NORM_II//HWSD_USA_II//m_T_REF_BULK.tif')
    file_san = [[0.0]*cols]*rows #栅格值和，二维数组
    file_san = np.array(file_san)
    file_san = Read('f://NORM_II//HWSD_USA_II//m_T_SAND.tif')
    file_sil = [[0.0]*cols]*rows #栅格值和，二维数组
    file_sil = np.array(file_sil)
    file_sil = Read('f://NORM_II//HWSD_USA_II//m_T_SILT.tif')
    #读动态变量的文件名
    rootdir_imerg = 'f://NORM_II//IV_USA_III'
    imerg_date = []
    for dirpath,filename,filenames in os.walk(rootdir_imerg):#遍历源文件
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.tif' :#判断是否为tif格式
                imerg_date.append(int(filename.split('.')[1][0:8]))
    imerg_date = np.array(imerg_date)
    print('IV days',len(imerg_date),imerg_date[0])
    rootdir_lst = 'f://NORM_II//MCD11A1_USA_II_ST'
    lst_date = []
    for dirpath,filename,filenames in os.walk(rootdir_lst):#遍历源文件
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.tif' :#判断是否为tif格式
                lst_date.append(int(filename.split('_')[1][0:7]))
    lst_date = np.array(lst_date)
    print('LST days',len(lst_date))
    rootdir_opt = 'f://NORM_II//MCD43A4_USA_II_ST'
    opt_date = []
    for dirpath,filename,filenames in os.walk(rootdir_opt+'//band1'):#遍历源文件
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.tif' :#判断是否为tif格式
                opt_date.append(int(filename.split('_')[1][0:7]))
    opt_date = np.array(opt_date)
    print('OPT days',len(opt_date))
    rootdir_luc = 'f://NORM_II//MCD12C1_USA_II'
    luc_date = []
    for dirpath,filename,filenames in os.walk(rootdir_luc):#遍历源文件
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.tif' :#判断是否为tif格式
                luc_date.append(int(filename.split('_')[2]))
    luc_date = np.array(luc_date)
    print('LUC years',len(luc_date))
    n_sample = 0  
    # judge = 1    
    recorder_col = [] 
    recorder_row = []   
    # recorder_sum = [] #栅格值和，二维数组
    for i in range(row_min,row_max-6,8):
         for j in range(col_min,col_max-6,8):
            if np.any( file_demo[i:i+8,j:j+8]==1 )== True:   
                recorder_col.append(j)
                recorder_row.append(i)
                n_sample+=1
    # recorder_sum = np.array(recorder_sum)
    recorder_row = np.array(recorder_row)
    recorder_col = np.array(recorder_col)
    # print('recorder_sum',recorder_sum.shape)
    n_sample_1 = 0  
    recorder_col_1 = [] 
    recorder_row_1 = []   
    # recorder_sum = [] #栅格值和，二维数组
    for i in range(row_min+4,row_max-10,8):
         for j in range(col_min+4,col_max-10,8):
            if np.any( file_demo[i:i+8,j:j+8]==1 )== True:   
                recorder_col_1.append(j)
                recorder_row_1.append(i)
                n_sample_1+=1
    # recorder_sum = np.array(recorder_sum)
    recorder_row_1 = np.array(recorder_row_1)
    recorder_col_1 = np.array(recorder_col_1)
    n_sample_2 = 0  
    recorder_col_2 = [] 
    recorder_row_2 = []   
    # recorder_sum = [] #栅格值和，二维数组
    for i in range(row_min+2,row_max-12,8):
         for j in range(col_min+2,col_max-12,8):
            if np.any( file_demo[i:i+8,j:j+8]==1 )== True:   
                recorder_col_2.append(j)
                recorder_row_2.append(i)
                n_sample_2+=1
    # recorder_sum = np.array(recorder_sum)
    recorder_row_2 = np.array(recorder_row_2)
    recorder_col_2 = np.array(recorder_col_2)
    
    n_sample_3 = 0  
    recorder_col_3 = [] 
    recorder_row_3 = []   
    # recorder_sum = [] #栅格值和，二维数组
    for i in range(row_min+6,row_max-8,8):
         for j in range(col_min+6,col_max-8,8):
            if np.any( file_demo[i:i+8,j:j+8]==1 )== True:   
                recorder_col_3.append(j)
                recorder_row_3.append(i)
                n_sample_3+=1
    # recorder_sum = np.array(recorder_sum)
    recorder_row_3 = np.array(recorder_row_3)
    recorder_col_3 = np.array(recorder_col_3)
    for date in range(4,len(imerg_date)): #经步长32测试发现第一个样本从20180114开始
        sample_s = []
        sample_s_1 = []
        sample_s_2 = []
        sample_s_3 = []
        #读day one LST，IMERG,OPT，LUC
        file_lst_1 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_lst_1 = np.array(file_lst_1)
        file_lst_1 = Read(rootdir_lst+'//MCD11A1_'+str(lst_date[date])+'.tif')
        file_imerg_1 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_imerg_1 = np.array(file_imerg_1)
        file_imerg_1 = Read(rootdir_imerg+'//m_st4.'+str(imerg_date[date])+'12.24h.tif')
        file_luc_1 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_luc_1 = np.array(file_luc_1)                    
        file_luc_1 = Read(rootdir_luc+'//m_lc_'+str(imerg_date[date])[0:4]+'_o.tif')
        file_opt_1 = [[[0.0]*cols]*rows]*7 #栅格值和，二维数组
        file_opt_1 = np.array(file_opt_1)
        for ii in range(0,len(file_opt_1)):
            file_opt_1[ii] = Read(rootdir_opt+'//band'+str(ii+1)+'//MCD43A4_'+str(opt_date[date])+'.tif')
        #读day two LST，IMERG,OPT，LUC
        file_lst_2 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_lst_2 = np.array(file_lst_2)
        file_lst_2 = Read(rootdir_lst+'//MCD11A1_'+str(lst_date[date-1])+'.tif')
        file_imerg_2 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_imerg_2 = np.array(file_imerg_2)
        file_imerg_2 = Read(rootdir_imerg+'//m_st4.'+str(imerg_date[date-1])+'12.24h.tif')
        file_luc_2 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_luc_2 = np.array(file_luc_2)                    
        file_luc_2 = Read(rootdir_luc+'//m_lc_'+str(imerg_date[date-1])[0:4]+'_o.tif')
        file_opt_2 = [[[0.0]*cols]*rows]*7 #栅格值和，二维数组
        file_opt_2 = np.array(file_opt_2)
        for ii in range(0,len(file_opt_2)):
            file_opt_2[ii] = Read(rootdir_opt+'//band'+str(ii+1)+'//MCD43A4_'+str(opt_date[date-1])+'.tif')
        #读day three LST，IMERG,OPT，LUC
        file_lst_3 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_lst_3 = np.array(file_lst_3)
        file_lst_3 = Read(rootdir_lst+'//MCD11A1_'+str(lst_date[date-2])+'.tif')
        file_imerg_3 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_imerg_3 = np.array(file_imerg_3)
        file_imerg_3 = Read(rootdir_imerg+'//m_st4.'+str(imerg_date[date-2])+'12.24h.tif')
        file_luc_3 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_luc_3 = np.array(file_luc_3)                    
        file_luc_3 = Read(rootdir_luc+'//m_lc_'+str(imerg_date[date-2])[0:4]+'_o.tif')
        file_opt_3 = [[[0.0]*cols]*rows]*7 #栅格值和，二维数组
        file_opt_3 = np.array(file_opt_3)
        for ii in range(0,len(file_opt_3)):
            file_opt_3[ii] = Read(rootdir_opt+'//band'+str(ii+1)+'//MCD43A4_'+str(opt_date[date-2])+'.tif')
        #读day four LST，IMERG,OPT，LUC
        file_lst_4 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_lst_4 = np.array(file_lst_4)
        file_lst_4 = Read(rootdir_lst+'//MCD11A1_'+str(lst_date[date-3])+'.tif')
        file_imerg_4 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_imerg_4 = np.array(file_imerg_4)
        file_imerg_4 = Read(rootdir_imerg+'//m_st4.'+str(imerg_date[date-3])+'12.24h.tif')
        file_luc_4 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_luc_4 = np.array(file_luc_4)                    
        file_luc_4 = Read(rootdir_luc+'//m_lc_'+str(imerg_date[date-3])[0:4]+'_o.tif')
        file_opt_4 = [[[0.0]*cols]*rows]*7 #栅格值和，二维数组
        file_opt_4 = np.array(file_opt_4)
        for ii in range(0,len(file_opt_4)):
            file_opt_4[ii] = Read(rootdir_opt+'//band'+str(ii+1)+'//MCD43A4_'+str(opt_date[date-3])+'.tif')
        #读day four LST，IMERG,OPT，LUC
        file_lst_5 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_lst_5 = np.array(file_lst_5)
        file_lst_5 = Read(rootdir_lst+'//MCD11A1_'+str(lst_date[date-4])+'.tif')
        file_imerg_5 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_imerg_5 = np.array(file_imerg_5)
        file_imerg_5 = Read(rootdir_imerg+'//m_st4.'+str(imerg_date[date-4])+'12.24h.tif')
        file_luc_5 = [[0.0]*cols]*rows #栅格值和，二维数组
        file_luc_5 = np.array(file_luc_5)                    
        file_luc_5 = Read(rootdir_luc+'//m_lc_'+str(imerg_date[date-4])[0:4]+'_o.tif')
        file_opt_5 = [[[0.0]*cols]*rows]*7 #栅格值和，二维数组
        file_opt_5 = np.array(file_opt_5)
        for ii in range(0,len(file_opt_5)):
            file_opt_5[ii] = Read(rootdir_opt+'//band'+str(ii+1)+'//MCD43A4_'+str(opt_date[date-4])+'.tif')
                    
        for i in range(0,len(recorder_row)):
            sample = [[[[0.0]*8]*8]*18]*5 #栅格值和，二维数组
            sample = np.array(sample)
            sample[0][0] = file_dem[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][1] = file_lat[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][2] = file_lon[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][3] = file_cla[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][4] = file_gra[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][5] = file_bul[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][6] = file_san[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][7] = file_sil[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][8] = file_lst_1[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][9] = file_imerg_1[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][10] = file_luc_1[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][11] = file_opt_1[0][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][12] = file_opt_1[1][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][13] = file_opt_1[2][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][14] = file_opt_1[3][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][15] = file_opt_1[4][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][16] = file_opt_1[5][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[0][17] = file_opt_1[6][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            
            sample[1][0] = file_dem[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][1] = file_lat[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][2] = file_lon[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][3] = file_cla[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][4] = file_gra[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][5] = file_bul[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][6] = file_san[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][7] = file_sil[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][8] = file_lst_2[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][9] = file_imerg_2[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][10] = file_luc_2[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][11] = file_opt_2[0][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][12] = file_opt_2[1][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][13] = file_opt_2[2][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][14] = file_opt_2[3][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][15] = file_opt_2[4][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][16] = file_opt_2[5][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[1][17] = file_opt_2[6][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            
            sample[2][0] = file_dem[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][1] = file_lat[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][2] = file_lon[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][3] = file_cla[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][4] = file_gra[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][5] = file_bul[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][6] = file_san[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][7] = file_sil[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][8] = file_lst_3[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][9] = file_imerg_3[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][10] = file_luc_3[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][11] = file_opt_3[0][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][12] = file_opt_3[1][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][13] = file_opt_3[2][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][14] = file_opt_3[3][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][15] = file_opt_3[4][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][16] = file_opt_3[5][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[2][17] = file_opt_3[6][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            
            sample[3][0] = file_dem[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][1] = file_lat[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][2] = file_lon[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][3] = file_cla[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][4] = file_gra[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][5] = file_bul[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][6] = file_san[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][7] = file_sil[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][8] = file_lst_4[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][9] = file_imerg_4[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][10] = file_luc_4[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][11] = file_opt_4[0][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][12] = file_opt_4[1][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][13] = file_opt_4[2][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][14] = file_opt_4[3][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][15] = file_opt_4[4][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][16] = file_opt_4[5][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[3][17] = file_opt_4[6][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            
            sample[4][0] = file_dem[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][1] = file_lat[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][2] = file_lon[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][3] = file_cla[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][4] = file_gra[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][5] = file_bul[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][6] = file_san[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][7] = file_sil[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][8] = file_lst_5[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][9] = file_imerg_5[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][10] = file_luc_5[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][11] = file_opt_5[0][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][12] = file_opt_5[1][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][13] = file_opt_5[2][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][14] = file_opt_5[3][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][15] = file_opt_5[4][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][16] = file_opt_5[5][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample[4][17] = file_opt_5[6][recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8]
            sample_s.append(sample)
        for i in range(0,len(recorder_row_1)):
            sample_1 = [[[[0.0]*8]*8]*18]*5 #栅格值和，二维数组
            sample_1 = np.array(sample_1)
            sample_1[0][0] = file_dem[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][1] = file_lat[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][2] = file_lon[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][3] = file_cla[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][4] = file_gra[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][5] = file_bul[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][6] = file_san[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][7] = file_sil[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][8] = file_lst_1[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][9] = file_imerg_1[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][10] = file_luc_1[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][11] = file_opt_1[0][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][12] = file_opt_1[1][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][13] = file_opt_1[2][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][14] = file_opt_1[3][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][15] = file_opt_1[4][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][16] = file_opt_1[5][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[0][17] = file_opt_1[6][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            
            sample_1[1][0] = file_dem[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][1] = file_lat[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][2] = file_lon[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][3] = file_cla[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][4] = file_gra[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][5] = file_bul[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][6] = file_san[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][7] = file_sil[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][8] = file_lst_2[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][9] = file_imerg_2[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][10] = file_luc_2[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][11] = file_opt_2[0][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][12] = file_opt_2[1][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][13] = file_opt_2[2][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][14] = file_opt_2[3][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][15] = file_opt_2[4][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][16] = file_opt_2[5][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[1][17] = file_opt_2[6][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            
            sample_1[2][0] = file_dem[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][1] = file_lat[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][2] = file_lon[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][3] = file_cla[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][4] = file_gra[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][5] = file_bul[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][6] = file_san[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][7] = file_sil[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][8] = file_lst_3[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][9] = file_imerg_3[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][10] = file_luc_3[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][11] = file_opt_3[0][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][12] = file_opt_3[1][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][13] = file_opt_3[2][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][14] = file_opt_3[3][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][15] = file_opt_3[4][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][16] = file_opt_3[5][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[2][17] = file_opt_3[6][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            
            sample_1[3][0] = file_dem[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][1] = file_lat[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][2] = file_lon[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][3] = file_cla[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][4] = file_gra[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][5] = file_bul[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][6] = file_san[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][7] = file_sil[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][8] = file_lst_4[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][9] = file_imerg_4[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][10] = file_luc_4[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][11] = file_opt_4[0][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][12] = file_opt_4[1][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][13] = file_opt_4[2][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][14] = file_opt_4[3][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][15] = file_opt_4[4][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][16] = file_opt_4[5][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[3][17] = file_opt_4[6][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            
            sample_1[4][0] = file_dem[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][1] = file_lat[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][2] = file_lon[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][3] = file_cla[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][4] = file_gra[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][5] = file_bul[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][6] = file_san[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][7] = file_sil[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][8] = file_lst_5[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][9] = file_imerg_5[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][10] = file_luc_5[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][11] = file_opt_5[0][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][12] = file_opt_5[1][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][13] = file_opt_5[2][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][14] = file_opt_5[3][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][15] = file_opt_5[4][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][16] = file_opt_5[5][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            sample_1[4][17] = file_opt_5[6][recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8]
            
            sample_s_1.append(sample_1)
        for i in range(0,len(recorder_row_2)):
            sample_2 = [[[[0.0]*8]*8]*18]*5 #栅格值和，二维数组
            sample_2 = np.array(sample_2)
            sample_2[0][0] = file_dem[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][1] = file_lat[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][2] = file_lon[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][3] = file_cla[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][4] = file_gra[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][5] = file_bul[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][6] = file_san[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][7] = file_sil[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][8] = file_lst_1[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][9] = file_imerg_1[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][10] = file_luc_1[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][11] = file_opt_1[0][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][12] = file_opt_1[1][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][13] = file_opt_1[2][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][14] = file_opt_1[3][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][15] = file_opt_1[4][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][16] = file_opt_1[5][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[0][17] = file_opt_1[6][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            
            sample_2[1][1] = file_lat[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][2] = file_lon[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][3] = file_cla[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][4] = file_gra[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][5] = file_bul[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][6] = file_san[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][7] = file_sil[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][8] = file_lst_2[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][9] = file_imerg_2[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][10] = file_luc_2[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][11] = file_opt_2[0][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][12] = file_opt_2[1][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][13] = file_opt_2[2][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][14] = file_opt_2[3][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][15] = file_opt_2[4][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][16] = file_opt_2[5][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[1][17] = file_opt_2[6][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            
            
            sample_2[2][1] = file_lat[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][2] = file_lon[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][3] = file_cla[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][4] = file_gra[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][5] = file_bul[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][6] = file_san[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][7] = file_sil[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][8] = file_lst_3[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][9] = file_imerg_3[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][10] = file_luc_3[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][11] = file_opt_3[0][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][12] = file_opt_3[1][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][13] = file_opt_3[2][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][14] = file_opt_3[3][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][15] = file_opt_3[4][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][16] = file_opt_3[5][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[2][17] = file_opt_3[6][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            
            sample_2[3][1] = file_lat[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][2] = file_lon[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][3] = file_cla[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][4] = file_gra[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][5] = file_bul[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][6] = file_san[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][7] = file_sil[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][8] = file_lst_4[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][9] = file_imerg_4[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][10] = file_luc_4[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][11] = file_opt_4[0][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][12] = file_opt_4[1][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][13] = file_opt_4[2][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][14] = file_opt_4[3][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][15] = file_opt_4[4][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][16] = file_opt_4[5][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[3][17] = file_opt_4[6][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            
            sample_2[4][1] = file_lat[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][2] = file_lon[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][3] = file_cla[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][4] = file_gra[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][5] = file_bul[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][6] = file_san[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][7] = file_sil[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][8] = file_lst_5[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][9] = file_imerg_5[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][10] = file_luc_5[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][11] = file_opt_5[0][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][12] = file_opt_5[1][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][13] = file_opt_5[2][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][14] = file_opt_5[3][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][15] = file_opt_5[4][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][16] = file_opt_5[5][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_2[4][17] = file_opt_5[6][recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8]
            sample_s_2.append(sample_2)
        for i in range(0,len(recorder_row_3)):
            sample_3 = [[[[0.0]*8]*8]*18]*5 #栅格值和，二维数组
            sample_3 = np.array(sample_3)
            sample_3[0][0] = file_dem[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][1] = file_lat[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][2] = file_lon[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][3] = file_cla[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][4] = file_gra[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][5] = file_bul[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][6] = file_san[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][7] = file_sil[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][8] = file_lst_1[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][9] = file_imerg_1[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][10] = file_luc_1[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][11] = file_opt_1[0][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][12] = file_opt_1[1][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][13] = file_opt_1[2][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][14] = file_opt_1[3][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][15] = file_opt_1[4][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][16] = file_opt_1[5][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[0][17] = file_opt_1[6][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            
            sample_3[1][0] = file_dem[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][1] = file_lat[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][2] = file_lon[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][3] = file_cla[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][4] = file_gra[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][5] = file_bul[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][6] = file_san[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][7] = file_sil[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][8] = file_lst_2[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][9] = file_imerg_2[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][10] = file_luc_2[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][11] = file_opt_2[0][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][12] = file_opt_2[1][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][13] = file_opt_2[2][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][14] = file_opt_2[3][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][15] = file_opt_2[4][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][16] = file_opt_2[5][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[1][17] = file_opt_2[6][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            
            sample_3[2][0] = file_dem[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][1] = file_lat[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][2] = file_lon[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][3] = file_cla[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][4] = file_gra[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][5] = file_bul[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][6] = file_san[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][7] = file_sil[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][8] = file_lst_3[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][9] = file_imerg_3[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][10] = file_luc_3[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][11] = file_opt_3[0][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][12] = file_opt_3[1][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][13] = file_opt_3[2][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][14] = file_opt_3[3][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][15] = file_opt_3[4][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][16] = file_opt_3[5][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[2][17] = file_opt_3[6][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            
            sample_3[3][0] = file_dem[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][1] = file_lat[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][2] = file_lon[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][3] = file_cla[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][4] = file_gra[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][5] = file_bul[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][6] = file_san[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][7] = file_sil[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][8] = file_lst_4[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][9] = file_imerg_4[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][10] = file_luc_4[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][11] = file_opt_4[0][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][12] = file_opt_4[1][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][13] = file_opt_4[2][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][14] = file_opt_4[3][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][15] = file_opt_4[4][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][16] = file_opt_4[5][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[3][17] = file_opt_4[6][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            
            sample_3[4][0] = file_dem[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][1] = file_lat[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][2] = file_lon[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][3] = file_cla[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][4] = file_gra[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][5] = file_bul[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][6] = file_san[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][7] = file_sil[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][8] = file_lst_5[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][9] = file_imerg_5[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][10] = file_luc_5[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][11] = file_opt_5[0][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][12] = file_opt_5[1][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][13] = file_opt_5[2][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][14] = file_opt_5[3][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][15] = file_opt_5[4][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][16] = file_opt_5[5][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_3[4][17] = file_opt_5[6][recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8]
            sample_s_3.append(sample_3)
        sample_s = np.array(sample_s)
        sample_s = np.where(abs(sample_s) > 1, 0, sample_s)
        sample_s_1 = np.array(sample_s_1)
        sample_s_1 = np.where(abs(sample_s_1) > 1, 0, sample_s_1)
        sample_s_2 = np.array(sample_s_2)
        sample_s_2 = np.where(abs(sample_s_2) > 1, 0, sample_s_2)
        sample_s_3 = np.array(sample_s_3)
        sample_s_3 = np.where(abs(sample_s_3) > 1, 0, sample_s_3)
        
        s_x1 = []  #动态变量
        s_x2 = []  #静态变量
        for i in range (0,len(sample_s)):
            s_x1.append(sample_s[i:i+1,:,8:18,:,:]) #动态变量
            s_x2.append(sample_s[i:i+1,0:1,0:8,:,:]) #静态变量
        s_x1 = np.array(s_x1)
        s_x1 = np.squeeze(s_x1)
        s_x2 = np.array(s_x2)
        s_x2 = np.squeeze(s_x2)
        # print(s_y.shape[0], s_y.shape, s_x1.shape, s_x2.shape)
        s_x1_1 = []  #动态变量
        s_x2_1 = []  #静态变量
        for i in range (0,len(sample_s_1)):
            s_x1_1.append(sample_s_1[i:i+1,:,8:18,:,:]) #动态变量
            s_x2_1.append(sample_s_1[i:i+1,0:1,0:8,:,:]) #静态变量
        s_x1_1 = np.array(s_x1_1)
        s_x1_1 = np.squeeze(s_x1_1)
        s_x2_1 = np.array(s_x2_1)
        s_x2_1 = np.squeeze(s_x2_1)
        
        s_x1_2 = []  #动态变量
        s_x2_2 = []  #静态变量
        for i in range (0,len(sample_s_2)):
            s_x1_2.append(sample_s_2[i:i+1,:,8:18,:,:]) #动态变量
            s_x2_2.append(sample_s_2[i:i+1,0:1,0:8,:,:]) #静态变量
        s_x1_2 = np.array(s_x1_2)
        s_x1_2 = np.squeeze(s_x1_2)
        s_x2_2 = np.array(s_x2_2)
        s_x2_2 = np.squeeze(s_x2_2)
        
        s_x1_3 = []  #动态变量
        s_x2_3 = []  #静态变量
        for i in range (0,len(sample_s_3)):
            s_x1_3.append(sample_s_3[i:i+1,:,8:18,:,:]) #动态变量
            s_x2_3.append(sample_s_3[i:i+1,0:1,0:8,:,:]) #静态变量
        s_x1_3 = np.array(s_x1_3)
        s_x1_3 = np.squeeze(s_x1_3)
        s_x2_3 = np.array(s_x2_3)
        s_x2_3 = np.squeeze(s_x2_3)
        
        test_x1 = s_x1
        test_x2 = s_x2
        test_x1_1 = s_x1_1
        test_x2_1 = s_x2_1
        test_x1_2 = s_x1_2
        test_x2_2 = s_x2_2
        test_x1_3 = s_x1_3
        test_x2_3 = s_x2_3

        print('shape of original samples',test_x1.shape, test_x2.shape)
        test_x2 = test_x2[:,None,:,:].copy()
        print('shape of processed samples',test_x1.shape,test_x2.shape)
        # return recorder_row,recorder_col,test_x1,test_x2
        testset=ImageDataset_array_test(test_x1,test_x2)
        print(test_x1.shape, test_x2.shape)
        testLoader = torch.utils.data.DataLoader(
                testset,
                batch_size=configs.batch_size_test,
                shuffle=False)
        

        outsum = []
        testmodel = TimesFormer_2input(configs).to(configs.device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        testmodel.to(device)
        model_info = torch.load(r'D:\DL\para-test\08-5-20241029\checkpoint_best.pth.tar')#改这里,加载最后一个checkpoint文件
        testmodel.load_state_dict(model_info['state_dict'])
        testset=ImageDataset_array_test(test_x1,test_x2)
        testLoader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False)
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, imgs in enumerate(t):
            inputs1 = Variable(imgs["Ix1"]).to(device)  # B,S,C,H,W
            inputs2 = Variable(imgs["Ix2"]).to(device)
            inputs2 = torch.squeeze(inputs2,1)
            # inputs = Variable(imgs["Ix"]).to(device)  # B,S,C,H,W
            # label = Variable(imgs["Iy"]).to(device)  # B,S,C,H,W
            testoutput = testmodel(inputs1=inputs1,inputs2=inputs2,tgt=None, train=False) # B,S,C,H,W  
            # print(testoutput.shape)
            testoutput2 = testoutput.view(testoutput.shape[0], testoutput.shape[1], testoutput.shape[2],testoutput.shape[3]).cpu()
            result1 = testoutput2.detach().numpy()
            # print(result1.shape)
            outsum.append(result1)
        outsum = np.array(outsum)
        outsum = outsum.squeeze(axis=1).copy()
        outsum = np.squeeze(outsum)
        print(outsum.shape)
        output = [[-9999.0]*cols]*rows 
        output = np.array(output)
        for i in range(0,len(recorder_row)):
            output[recorder_row[i]:recorder_row[i]+8, recorder_col[i]:recorder_col[i]+8] = outsum[i:i+1,:,:]
            
            
        print('shape of original samples_1',test_x1_1.shape, test_x2_1.shape)
        test_x2_1 = test_x2_1[:,None,:,:].copy()
        print('shape of processed samples_1',test_x1_1.shape,test_x2_1.shape)
        # return recorder_row,recorder_col,test_x1,test_x2
        testset_1=ImageDataset_array_test(test_x1_1,test_x2_1)
        print(test_x1_1.shape, test_x2_1.shape)
        testLoader = torch.utils.data.DataLoader(
                testset_1,
                batch_size=configs.batch_size_test,
                shuffle=False)

        outsum_1 = []
        testmodel = TimesFormer_2input(configs).to(configs.device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        testmodel.to(device)
        model_info = torch.load(r'D:\DL\para-test\08-5-20241029\checkpoint_best.pth.tar')#改这里,加载最后一个checkpoint文件
        testmodel.load_state_dict(model_info['state_dict'])
        testset_1=ImageDataset_array_test(test_x1_1,test_x2_1)
        testLoader = torch.utils.data.DataLoader(testset_1, batch_size=1,shuffle=False)
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, imgs in enumerate(t):
            inputs1 = Variable(imgs["Ix1"]).to(device)  # B,S,C,H,W
            inputs2 = Variable(imgs["Ix2"]).to(device)
            inputs2 = torch.squeeze(inputs2,1)
            # inputs = Variable(imgs["Ix"]).to(device)  # B,S,C,H,W
            # label = Variable(imgs["Iy"]).to(device)  # B,S,C,H,W
            testoutput = testmodel(inputs1=inputs1,inputs2=inputs2,tgt=None, train=False) # B,S,C,H,W  
            # print(testoutput.shape)
            testoutput2 = testoutput.view(testoutput.shape[0], testoutput.shape[1], testoutput.shape[2],testoutput.shape[3]).cpu()
            result1 = testoutput2.detach().numpy()
            # print(result1.shape)
            outsum_1.append(result1)
        outsum_1 = np.array(outsum_1)
        outsum_1 = outsum_1.squeeze(axis=1).copy()
        outsum_1 = np.squeeze(outsum_1)
        print(outsum_1.shape)
        output_1 = [[-9999.0]*cols]*rows 
        output_1 = np.array(output_1)
        for i in range(0,len(recorder_row_1)):
            output_1[recorder_row_1[i]:recorder_row_1[i]+8, recorder_col_1[i]:recorder_col_1[i]+8] = outsum_1[i:i+1,:,:]
            
        print('shape of original samples_2',test_x1_2.shape, test_x2_2.shape)
        test_x2_2 = test_x2_2[:,None,:,:].copy()
        print('shape of processed samples_2',test_x1_2.shape,test_x2_2.shape)
        # return recorder_row,recorder_col,test_x1,test_x2
        testset_2=ImageDataset_array_test(test_x1_2,test_x2_2)
        print(test_x1_2.shape, test_x2_2.shape)
        testLoader = torch.utils.data.DataLoader(
                testset_2,
                batch_size=configs.batch_size_test,
                shuffle=False)

        outsum_2 = []
        testmodel = TimesFormer_2input(configs).to(configs.device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        testmodel.to(device)
        model_info = torch.load(r'D:\DL\para-test\08-5-20241029\checkpoint_best.pth.tar')#改这里,加载最后一个checkpoint文件
        testmodel.load_state_dict(model_info['state_dict'])
        testset_2=ImageDataset_array_test(test_x1_2,test_x2_2)
        testLoader = torch.utils.data.DataLoader(testset_2, batch_size=1,shuffle=False)
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, imgs in enumerate(t):
            inputs1 = Variable(imgs["Ix1"]).to(device)  # B,S,C,H,W
            inputs2 = Variable(imgs["Ix2"]).to(device)
            inputs2 = torch.squeeze(inputs2,1)
            # inputs = Variable(imgs["Ix"]).to(device)  # B,S,C,H,W
            # label = Variable(imgs["Iy"]).to(device)  # B,S,C,H,W
            testoutput = testmodel(inputs1=inputs1,inputs2=inputs2,tgt=None, train=False) # B,S,C,H,W  
            # print(testoutput.shape)
            testoutput2 = testoutput.view(testoutput.shape[0], testoutput.shape[1], testoutput.shape[2],testoutput.shape[3]).cpu()
            result1 = testoutput2.detach().numpy()
            # print(result1.shape)
            outsum_2.append(result1)
        outsum_2 = np.array(outsum_2)
        outsum_2 = outsum_2.squeeze(axis=1).copy()
        outsum_2 = np.squeeze(outsum_2)
        print(outsum_2.shape)
        output_2 = [[-9999.0]*cols]*rows 
        output_2 = np.array(output_2)
        for i in range(0,len(recorder_row_2)):
            output_2[recorder_row_2[i]:recorder_row_2[i]+8, recorder_col_2[i]:recorder_col_2[i]+8] = outsum_2[i:i+1,:,:]
            
        print('shape of original samples_3',test_x1_3.shape, test_x2_3.shape)
        test_x2_3 = test_x2_3[:,None,:,:].copy()
        print('shape of processed samples_3',test_x1_3.shape,test_x2_3.shape)
        # return recorder_row,recorder_col,test_x1,test_x2
        testset_3=ImageDataset_array_test(test_x1_3,test_x2_3)
        print(test_x1_3.shape, test_x2_3.shape)
        testLoader = torch.utils.data.DataLoader(
                testset_2,
                batch_size=configs.batch_size_test,
                shuffle=False)

        outsum_3 = []
        testmodel = TimesFormer_2input(configs).to(configs.device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        testmodel.to(device)
        model_info = torch.load(r'D:\DL\para-test\08-5-20241029\checkpoint_best.pth.tar')#改这里,加载最后一个checkpoint文件
        testmodel.load_state_dict(model_info['state_dict'])
        testset_3=ImageDataset_array_test(test_x1_3,test_x2_3)
        testLoader = torch.utils.data.DataLoader(testset_3, batch_size=1,shuffle=False)
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, imgs in enumerate(t):
            inputs1 = Variable(imgs["Ix1"]).to(device)  # B,S,C,H,W
            inputs2 = Variable(imgs["Ix2"]).to(device)
            inputs2 = torch.squeeze(inputs2,1)
            # inputs = Variable(imgs["Ix"]).to(device)  # B,S,C,H,W
            # label = Variable(imgs["Iy"]).to(device)  # B,S,C,H,W
            testoutput = testmodel(inputs1=inputs1,inputs2=inputs2,tgt=None, train=False) # B,S,C,H,W  
            # print(testoutput.shape)
            testoutput2 = testoutput.view(testoutput.shape[0], testoutput.shape[1], testoutput.shape[2],testoutput.shape[3]).cpu()
            result1 = testoutput2.detach().numpy()
            # print(result1.shape)
            outsum_3.append(result1)
        outsum_3 = np.array(outsum_3)
        outsum_3 = outsum_3.squeeze(axis=1).copy()
        outsum_3 = np.squeeze(outsum_3)
        print(outsum_3.shape)
        output_3 = [[-9999.0]*cols]*rows 
        output_3 = np.array(output_3)
        for i in range(0,len(recorder_row_3)):
            output_3[recorder_row_3[i]:recorder_row_3[i]+8, recorder_col_3[i]:recorder_col_3[i]+8] = outsum_3[i:i+1,:,:]    
        
        output_1[output_1<0] = np.nan
        output[output<0] = np.nan
        output_2[output_2<0] = np.nan
        output_3[output_3<0] = np.nan
        
        output_0 = []
        output_0.append(output)
        output_0.append(output_1)
        output_0.append(output_2)
        output_0.append(output_3)
        output_0 = np.array(output_0)
        output_f = np.nanmean(output_0,axis = 0 )
        output_f[np.isnan(output_f)] = -9999.0
        path = 'd://DL//TF-reconstruct//SM_'+str(imerg_date[date])+'_0.tif'
        WriteGTiffFile(path,rows,cols,output_f,geotransform,projection, -9999.0,gdalconst.GDT_Float32)  
        print (path)

if __name__ == "__main__":   
    print('start from here')
    root_dir = 'f://NORM_II' #sample_5
    
    
    read_samples0805(root_dir)    
    
    # testset=ImageDataset_array_test(testX1,testX2)
    # print(testX1.shape, testX2.shape)
    
    # testLoader = torch.utils.data.DataLoader(
    #         testset,
    #         batch_size=configs.batch_size_test,
    #         shuffle=True)

    # predY=test()
    
    # testY = testY.squeeze().copy()
    # testY = testY[:,:,:,None]

    print('end at here')

