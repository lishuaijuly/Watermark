#coding=utf-8
'''
    不可见水印的生成和提取。
    PIL比opencv差多了。
'''
import sys
import logging
import time

import cv2
import pywt
import numpy as np

import script.util as imutil

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class LsbWatermark:
    @staticmethod
    def decompose(data): # bytes转化为普通数组
        v = []
        fsize = (len(data)+4) * 8 
        bs = imutil.intToBytes(fsize)
        bs += [b for b in data]

        for b in bs:
            for i in range(7, -1, -1):
                v.append((b >> i) & 0x1)

        return v     # 4 + 8*len

    @staticmethod
    def assemble(v):    #普通数组转化为bytes
        bs = []
        length = len(v)
        for idx in range(0, int(len(v)/8)):
            byte = 0
            for i in range(0, 8):
                if (idx*8+i < length):
                    byte = (byte<<1) + v[idx*8+i]  
                           
            bs .append(byte)

        #invalidsize = (len(bs) -4) % 8
        return bytes(bs[4:])
        
    @staticmethod        
    def calc_wmsize(v):
        bs = []
        for idx in range(0, int(len(v)/8)):
            byte = 0
            for i in range(0, 8):
                byte = (byte<<1) + v[idx*8+i]    
            bs.append(byte)

        fsize = imutil.bytesToInt(bs)
        return fsize

    def embed(self,img, wmdata, key=None,max_wm_size=2*1024+4):
        '''
            使用LSB的方式生成不可见水印
            img ：原图 ，Image
            wm : 任意bytes数据
        '''
        img_shape= img.shape
        logging.info("输入文件大小 : %dx%d pixels." % (img_shape[0] ,img_shape[1]))
        
        max_size =min(max_wm_size,img_shape[0] * img_shape[1]*3.0/8)
        logging.info("最大可保存水印信息量 : %.2f Bit." % (max_size))

        if key !=None:
            cipher = imutil.AESCipher(key)
            wmdata = cipher.encrypt(wmdata) #对数据做加密
 
        v = self.decompose(wmdata) #转化为0-1
        payload_size = len(v)/8
        logging.info("要保存的信息量 : %.3f Bit " % (payload_size))
        if (payload_size > max_size - 4):
            logging.fatal("数据太大，不能作为水印。")
            return None
            
        #LSB填充并输出
        idx = 0
        img = img.flatten()
        for i in range(len(v)):
            img[i]  = imutil.set_bit(img[i], 0, v[i])

        img = img.reshape(img_shape)
        return img
        
    def extract(self,wmd_img, key=None,max_wm_size=2*1024+4):
        '''
            提取LSB信息
        '''
        wd_shape = wmd_img.shape
        
        wmd_img =wmd_img.flatten()
        max_size = max_wm_size
        
        #提取信息
        v = []
        for i in range(max_size):
            if len(v) > max_size:
                break
            
            v.append(wmd_img[i] & 1)
            if i==31:
                fsize=self.calc_wmsize(v)  #先提取头部的长度信息
                max_size=min(max_wm_size,fsize)

        data_out = self.assemble(v)
        if key !=None:
            cipher = imutil.AESCipher(key)
            data_out = cipher.decrypt(data_out)  #解密

        return data_out

    

class DwtsvdWatermark:
    '''
        盲水印，DWT+SVD
    '''      

    def _gene_signature(self,wU,wV,key):
        '''提取特征，用来比对是否包含水印的，不用来恢复水印'''
        sumU = np.sum(wU,axis=0)
        sumV = np.sum(wV,axis=0)

        sumU_mid = np.median(sumU)
        sumV_mid = np.median(sumV)

        sumU=np.array([1 if sumU[i] >sumU_mid else 0 for i in range(len(sumU)) ])
        sumV=np.array([1 if sumV[i] >sumV_mid else 0 for i in range(len(sumV)) ])

        uv_xor=np.logical_xor(sumU,sumV)

        np.random.seed(key)
        seq=np.random.randint(2,size=len(uv_xor))

        signature = np.logical_xor(uv_xor, seq)
        return np.array(list(signature),dtype=np.float)

    def _gene_embed_space(self,LL_4,HH_4):
        LL_4 = LL_4.reshape([1,-1])
        HH_4 = HH_4.reshape([1,-1])
        combo_LL4_HH4 = np.append(LL_4,HH_4)
        combo_neg_idx = [1 if combo_LL4_HH4[i]<0  else 0 for i in range(len(combo_LL4_HH4))]

        combo_LL4_HH4_pos = np.abs(combo_LL4_HH4)
        int_part = np.floor(combo_LL4_HH4_pos)
        frac_part = np.round(combo_LL4_HH4_pos - int_part,2)
        
        bi_int_part=[] #数据转化为二维的，然后使用signature替换其中一位。
        for i in range(len(int_part)):
            bi=list(bin(int(int_part[i]))[2:])
            bie = [0] * (16 - len(bi))
            bie.extend(bi)
            bi_int_part.append(np.array(bie,dtype=np.uint16))
        bi_int_part = np.array(bi_int_part)

        return np.array(bi_int_part),frac_part,combo_neg_idx

    def _embed_signature(self,LL,signature):
        '''把特征嵌入到低频信号中'''
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(np.array(LL,dtype=np.int32),'haar') 
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(np.array(LL_1,dtype=np.int32),'haar') 
       
        bi_int_part,frac_part,combo_neg_idx = self._gene_embed_space(LL_2,HH_2)

        for i in range(len(signature)):   #只替换了前256个
            bi_int_part[i][10] = signature[i]

        #嵌入完成，组装回去
        em_int_part = []
        for i in range(len(bi_int_part)):
            s='0b'
            s+= (''.join([str(j) for j in bi_int_part[i]]))
            em_int_part.append(eval(s))
        
        em_int_part = np.array(em_int_part)
        em_combo = em_int_part + frac_part

        em_combo = np.array([-1*em_combo[i] if combo_neg_idx[i]==1 else em_combo[i] for i in range(len(em_combo))])
        
        em_LL_2 = em_combo[:4096].reshape((64,64))
        em_HH_2 = em_combo[4096:].reshape((64,64))
        em_LL_1 = pywt.idwt2((em_LL_2,(HL_2,LH_2,em_HH_2)),'haar')
        em_LL   = pywt.idwt2((em_LL_1,(HL_1,LH_1,HH_1)),'haar')

        return em_LL
        
    def embed(self, img,wm,key=10):
        w,h = img.shape[:2]
        img = cv2.resize(img,(512,512))
       
        LL,(HL,LH,HH) = pywt.dwt2(np.array(img),'haar') #使用红色通道的高频分量保存水印的特征值
        iU,iS,iV = np.linalg.svd(np.mat(HH))

        wm = cv2.resize(wm,(512,512))  #水印图像转成 灰度图像
        wU,wS,wV = np.linalg.svd(np.mat(wm))
        
        em_HH     = iU * np.diag(wS[:256]) *iV  #高频的特征值水印，对图像变化不敏感。用来恢复水印使用@

        signature = self._gene_signature(np.array(wU),np.array(wV),key)
        em_LL     = self._embed_signature(LL, signature)
       
        img = pywt.idwt2((em_LL,(HL, LH,em_HH)), 'haar')  

        return cv2.resize(img,(h,w))


    def extract(self,wmimage,wm,key=10):
        ''' 检测是否包含水印，如果包含的话输出水印'''
        wmimage = cv2.resize(wmimage,(512,512))
       
        LL,(HL,LH,HH) = pywt.dwt2(wmimage,'haar') 
        
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(LL,'haar') 
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(LL_1,'haar') 

        ori_w_shape = wm.shape
        wm = cv2.resize(wm,(512,512)) #水印图像转成 灰度图像
        wU,wS,wV = np.linalg.svd(np.mat(wm))

        signature = self._gene_signature (np.array(wU), np.array(wV), key)

        ext_signature = []
        bi_int_part,frac_part,_ = self._gene_embed_space(LL_2,HH_2)

        for i in range(len(signature)):   #替换
            ext_signature.append(bi_int_part[i][10] )

        signature = signature / np.sqrt(np.sum(signature*signature))
        ext_signature = np.array(ext_signature,dtype=np.uint16)
        ext_signature = ext_signature / np.sqrt(np.sum(ext_signature*ext_signature))
        corecoef = np.sum(signature*ext_signature)

        logging.info('提取的信息和水印信息相似度是：%f (1最大，0最小，建议参考值是0.7)'  % (corecoef))
        iU,iS,iV = np.linalg.svd(np.mat(HH))
        uS = list(iS)
        uS.extend(list(wS)[256:])
        em_rec_data = wU * np.diag(np.array(uS)) * wV

        return  cv2.resize(em_rec_data,ori_w_shape)*255,corecoef
