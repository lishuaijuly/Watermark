#coding:utf-8
"""
    给图片加一个显性的（图片/文本）水印
    通用参数：
        水印位置
        透明度
        是否平铺
        缩放大小
        旋转角度

    图片参数：
        水印图片
        
    文字参数：
        水印文字
        文字颜色
        文字字体
        文字大小
"""

import copy
from io  import StringIO

import pygame
import cv2
from PIL import Image
pygame.init()

class VisWatermark:
    def scale_resize(self,im,maxsize):
        iw,ih = maxsize
        ww,wh = im.shape[:2] 
        wratio = ww/iw
        hratio = wh/ih
        if wratio >1 or hratio >1 :  #水印大小不能超过原图片
            scale = max(wratio,hratio)
            im = cv2.resize(im,(int(wh/scale),int(ww/scale)))
        return im


    def watermark_image(self,im, wmim, params):
        '''给图片加水印图片'''

        if  not 'position' in params:
            params['position']      = 'tile' #tile scale ,(0,0)
        if  not 'transparency' in params:
            params['transparency']  = 0.1       # 默认0.1，1：水印完全不透明 0：水印完全透明 。
        if  not 'rotate' in params:
            params['rotate']        = 0       # 默认不旋转

        ww,wh = wmim.shape[:2]
        rot = params['rotate']  #处理旋转
        if rot != 0:
            M=cv2.getRotationMatrix2D((wh/2,ww/2),45,1)
            wmim=cv2.warpAffine(wmim,M,(wh*2,ww*2))

        position = params['position']  #处理水印位置

        wmim = self.scale_resize(wmim,im.shape[:2])
        
        transp = params['transparency']  #调整透明度
        if transp > 0.99999 and transp < 0.00001:
            transp = 1

        iw,ih = im.shape[:2]
        ww,wh = wmim.shape[:2] 
        if position == 'tile':  #平铺
            for i in range(0,iw,ww):
                for j in range(0,ih,wh):
                    im[i:ww+i,j:wh+j,:] = im[i:ww+i,j:wh+j,:] * (1-transp) + wmim*transp
        elif position == 'scale':   #放大到全图  
            im = im*  (1-transp) + cv2.resize(wmim,(ih,iw))*transp 
        else: 
            posw,posh = position
            #wmim = cv2.resize(wmim,(ih-posh,iw - posw))
            if ww > iw - posw:
                ww = iw - posw
            if wh > ih - posh:
                wh = ih - posh
            wmim = cv2.resize(wmim,(wh,ww))
            im[posw:posw+ww,posh:posh+wh,:] = im[posw:posw+ww,posh:posh+wh,:] * (1-transp) + wmim*transp


        return im 
    

    def text2image(self,im, text,t2mparam):
        '''
            给图片添加文字水印，处理分成两步
            1、把文字转换成图片
            2、调用watermark_image
        '''
        if  not 'text-size' in t2mparam:
            t2mparam['text-size']     = 300     # 默认300      
        if  not 'text-font' in t2mparam:    
            t2mparam['text-font']     = './data/simsun.ttc'
        if  not 'text-color' in t2mparam:
            t2mparam['text-color']    = (250,240,230) #默认 亚麻灰
        if  not 'text-bgcolor' in t2mparam:
            t2mparam['text-bgcolor']  = (0,0,0) 


        pyfont = pygame.font.Font(t2mparam['text-font'],t2mparam['text-size'])
        rtext = pyfont.render(text,True,t2mparam['text-color'],t2mparam['text-bgcolor'])
        pio = pygame.image.tostring(rtext, 'RGBA')
        wmim = Image.frombytes(data=pio,mode='RGBA',size=rtext.get_size())
        return wmim
