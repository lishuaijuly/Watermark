#coding=utf-8
import sys
import logging
import time
import math

import cv2
import pywt
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class BlindWatermark():

    @staticmethod
    def _gene_signature(wm,size,key):
        '''
            提取特征，用来比对是否包含水印的，不用来恢复水印
            wm   : 水印图片
            size ：生成的特征文件大小
            key  ：生产的特征密钥
        '''
        wm = cv2.resize(wm,(size,size))        
        wU,_,wV = np.linalg.svd(np.mat(wm))

        sumU = np.sum(np.array(wU),axis=0)
        sumV = np.sum(np.array(wV),axis=0)

        sumU_mid = np.median(sumU)
        sumV_mid = np.median(sumV)

        sumU=np.array([1 if sumU[i] >sumU_mid else 0 for i in range(len(sumU)) ])
        sumV=np.array([1 if sumV[i] >sumV_mid else 0 for i in range(len(sumV)) ])

        uv_xor=np.logical_xor(sumU,sumV)

        np.random.seed(key)
        seq=np.random.randint(2,size=len(uv_xor))

        signature = np.logical_xor(uv_xor, seq)

        sqrts = int(np.sqrt(size))
        return np.array(signature,dtype=np.int8).reshape((sqrts,sqrts))

    @staticmethod
    def calc_sim(sig1,sig2s):
        max_sim =0 
        for  sig2 in sig2s:
            match_cnt = np.sum(np.equal(np.array(sig1,dtype=np.int),np.array(sig2,dtype=np.int)))
            sim = match_cnt/ len(sig1)
            max_sim =max(max_sim,sim)
        return max_sim

    def inner_embed(self,B,signature):
        pass

    def inner_extract(self,B,signature):
        pass

    def embed(self,ori_img, wm, key=10):
        B =  ori_img
        if len(ori_img.shape ) > 2 :
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YUV)
            signature = BlindWatermark._gene_signature(wm,256,key).flatten()
            B= img[:,:,0]
        
        w,h = B.shape[:2]
        if w< 64 or h <64 :
            print('原始图像的长度或者宽度小于 64 pixel.不能嵌入，返回原图.')
            return ori_img
        
        if len(ori_img.shape ) > 2 :
            img[:,:,0] = self.inner_embed(B,signature)  
            ori_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        else :
            ori_img = B
        return ori_img
         
    def extract(self,ori_wmimage,wm,key=10):
        B = ori_wmimage
        if len(ori_wmimage.shape ) > 2 :
            (B,G,R) = cv2.split(cv2.cvtColor(ori_wmimage, cv2.COLOR_BGR2YUV))

        signature = BlindWatermark._gene_signature(wm,256,key).flatten()

        ext_sig = self.inner_extract(B,signature)
        return BlindWatermark.calc_sim(signature,ext_sig)


class DCT_watermark(BlindWatermark):
    def __init__(self):
        self.Q = 10
        self.size = 2

    #把所有信息嵌入到 32×32 的方块里 （可以嵌入很多地方）
    #嵌入到是高频区域，高频反应的是图像的边缘等细节信息，但是'文档截图'的边缘是黑白的，嵌入任何信息都导致不可见性差。
    #如果嵌入到低频区域，问题失真更大。
    def inner_embed(self,B,signature):
        sig_size=np.int(np.sqrt(len(signature)))
        size = self.size
        
        #四个拐角最多嵌入四份，可用于检测边缘切割和旋转
        #  (0,0)    (0,w-32)
        #  (h-32,0)    (h-32,w-32)
        
        w,h = B.shape[:2]
        embed_pos = [(0,0)]
        if w > 2* sig_size*size  :
            embed_pos.append((w-sig_size*size,0))
        if h > 2* sig_size*size  :
            embed_pos.append((0,h-sig_size*size))
        if len(embed_pos) ==3:
            embed_pos.append((w-sig_size*size,h-sig_size*size))

        for x,y in embed_pos:
            for i in range(x,x+sig_size*size,size):
                for j in range(y,y+sig_size * size,size):
                    v = np.float32(B[i:i+size,j:j+size])
                    v = cv2.dct(v)
                    v[size-1,size-1] = self.Q*signature[((i-x)//size)*sig_size+(j-y)//size]
                    v = cv2.idct(v)
                    maxium = max(v.flatten())
                    minium = min(v.flatten())
                    if maxium > 255:
                        v = v - (maxium - 255)
                    if minium < 0:
                        v = v - minium
                    B[i:i+size,j:j+size]= v
        return B

    def inner_extract(self,B,signature):
        sig_size=np.int(np.sqrt(len(signature)))
        size = self.size
        
        ext_sigs =[] 
        #检测四个角的，并且检测旋转后的
        #四个拐角最多嵌入四份，可用于检测边缘切割和旋转
        #  (0,0)    (0,w-32)
        #  (h-32,0)    (h-32,w-32)
        w ,h = B.shape
        embed_pos =[(0,0)]
        embed_pos.append((w-sig_size*size,0))
        embed_pos.append((0,h-sig_size*size))
        embed_pos.append((w-sig_size*size,h-sig_size*size))

        for x,y in embed_pos:
            ext_sig = np.zeros(len(signature),dtype=np.int)
            
            for i in range(x,x+sig_size*size,size):
                for j in range(y,y+sig_size * size,size):
                    v = cv2.dct(np.float32(B[i:i+size,j:j+size]))
                    if v[size-1,size-1] > self.Q/2:
                        ext_sig[((i-x)//size)*sig_size+(j-y)//size] = 1 
                    

            ext_sigs.append(ext_sig)
            ext_sig_arr = np.array(ext_sig).reshape((sig_size,sig_size))
            ext_sigs.append(np.rot90(ext_sig_arr,1).flatten())
            ext_sigs.append(np.rot90(ext_sig_arr,2).flatten())
            ext_sigs.append(np.rot90(ext_sig_arr,3).flatten())
            
        return ext_sigs

class DWT_watermark(BlindWatermark):
    def __init__(self):
        pass

    def _gene_embed_space(self,vec):
        shape = vec.shape
        vec = vec.flatten()
        combo_neg_idx = np.array([1 if vec[i]<0  else 0 for i in range(len(vec))])

        vec_pos = np.abs(vec)
        int_part = np.floor(vec_pos)
        frac_part = np.round(vec_pos - int_part,2)
        
        bi_int_part=[] #数据转化为二维的，然后使用signature替换其中一位。
        for i in range(len(int_part)):
            bi=list(bin(int(int_part[i]))[2:])
            bie = [0] * (16 - len(bi))
            bie.extend(bi)
            bi_int_part.append(np.array(bie,dtype=np.uint16))
        bi_int_part = np.array(bi_int_part)

        sig = []
        for i in range(len(bi_int_part)):
            sig.append(bi_int_part[i][10])
        sig = np.array(sig).reshape(shape)
        return np.array(bi_int_part),frac_part.reshape(shape),combo_neg_idx.reshape(shape),sig


    def _embed_sig(self,bi_int_part,frac_part,combo_neg_idx,signature):
        shape = frac_part.shape

        frac_part = frac_part.flatten()
        combo_neg_idx = combo_neg_idx.flatten()
        
        m = len(signature)
        n = len(bi_int_part)
        logging.info('特征向量大小 ： {} ,嵌入空间大小： {}'.format(m,n))

        if m >= n :
            for i in range(n):
                bi_int_part[i][10] =signature[i]

        if m < n :  #全部嵌入
            rate = 1 # = n//m
            for i in range(m):
                for j in range(rate):
                    bi_int_part[i+j*m][10] =signature[i]
                

        #嵌入完成，组装回去
        em_int_part = []
        for i in range(len(bi_int_part)):
            s='0b'
            s+= (''.join([str(j) for j in bi_int_part[i]]))
            em_int_part.append(eval(s))
        
        em_combo = np.array(em_int_part) + np.array(frac_part)

        return  np.array([-1*em_combo[i] if combo_neg_idx[i]==1 else em_combo[i] for i in range(len(em_combo))]).reshape(shape)

    def _extract_sig(self,ext_sig,siglen):
        ext_sig = list(ext_sig.flatten())
        
        m = len(ext_sig)
        n = siglen
        ext_sigs=[]
        
        if n >= m :
            ext_sigs.append(ext_sig.extend([0] * (n-m)))
        
        if n < m:
            rate = 1 #= m//n
            for i  in range(rate):
                ext_sigs.append( ext_sig[i*n:(i+1)*n])
                
        return ext_sigs


    def inner_embed(self,B,signature):
        w,h =B.shape[:2]

        LL,(HL,LH,HH) = pywt.dwt2(np.array(B[:32*(w//32),:32*(h//32)]),'haar')  
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(LL,'haar')    
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(LL_1,'haar')   
        LL_3,(HL_3,LH_3,HH_3) = pywt.dwt2(LL_2,'haar')  
        LL_4,(HL_4,LH_4,HH_4) = pywt.dwt2(LL_3,'haar')  
       
        bi_int_part,frac_part,combo_neg_idx,_ = self._gene_embed_space(HH_3)
        HH_3 = self._embed_sig(bi_int_part,frac_part,combo_neg_idx,signature)
       
        LL_3 = pywt.idwt2((LL_4,(HL_4,LH_4,HH_4)),'haar')
        LL_2 = pywt.idwt2((LL_3,(HL_3,LH_3,HH_3)),'haar')   
        LL_1 = pywt.idwt2((LL_2,(HL_2,LH_2,HH_2)),'haar')
        LL   = pywt.idwt2((LL_1,(HL_1,LH_1,HH_1)),'haar')
        B[:32*(w//32),:32*(h//32)]  = pywt.idwt2((LL,(HL, LH,HH)), 'haar')

        return B
        
    def inner_extract(self,B,signature):
        w,h =B.shape[:2]
        
        LL,(HL,LH,HH) = pywt.dwt2(B[:32*(w//32),:32*(h//32)],'haar') 
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(LL,'haar') 
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(LL_1,'haar') 
        LL_3,(HL_3,LH_3,HH_3) = pywt.dwt2(LL_2,'haar')
        LL_4,(HL_4,LH_4,HH_4) = pywt.dwt2(LL_3,'haar')
        
        _,_,_,ori_sig = self._gene_embed_space(HH_3)
        
        ext_sigs=[]
        ext_sigs.extend(self._extract_sig(ori_sig,len(signature)))
        ext_sigs.extend(self._extract_sig(np.rot90(ori_sig,1),len(signature)))
        ext_sigs.extend(self._extract_sig(np.rot90(ori_sig,2),len(signature)))
        ext_sigs.extend(self._extract_sig(np.rot90(ori_sig,3),len(signature)))
        
        return ext_sigs