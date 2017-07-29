#coding=utf-8
import sys
import logging
import time

import cv2
import pywt
import numpy as np

import script.util as imutil

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class BlindWatermark():
    def _gene_signature(self,wm,size,key):
        '''提取特征，用来比对是否包含水印的，不用来恢复水印'''
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

    def embed(self,ori_img, wm, key=10):
        #ori_img = cv2.cvtColor(np.float32(ori_img), cv2.COLOR_BGR2YUV)

        if len(ori_img.shape)==3:
            img  = ori_img[:,:,0] 
        else:
            img  = ori_img  

        size =(min( img.shape[:2]) //4) *4


        LL,(HL,LH,HH) = pywt.dwt2(np.array(img[:size,:size]),'haar')  #512*512
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(LL,'haar')    #256
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(LL_1,'haar')  #128 
        print(LL_1.shape,HL_1.shape,LH_1.shape,HH_1.shape)
        embed = HH_1

        #生成和嵌入特征        
        signature = self._gene_signature(wm,64,key) #返回8×8的矩阵
        w,h = embed.shape[:2]
        if w < 32 or h < 32:
            logging.warn("输入图片大小是 : {} × {} ，小于 32×32 ，不能嵌入,原图返回。".format(w ,h))
            return ori_img

        Q = 10
        for i in range(0,w-4,4):
            for j in range(0,h-4,4):
                v = cv2.dct(np.float32(embed[i:i+4,j:j+4]))
                v[2,2] = Q * signature[i%8,j%8]

                maxium = max(v.flatten())
                minium = min(v.flatten())
                if maxium > 255:
                    v = v - (maxium - 255)
                if minium < 0:
                    v = v - minium
            
                embed[i:i+4,j:j+4] = cv2.idct(v)


        HH_1= embed
        print(LL_1.shape,HL_1.shape,LH_1.shape,HH_1.shape)
        #LL_1 = pywt.idwt2((LL_2,(HL_2,LH_2,HH_2)),'haar')
        LL   = pywt.idwt2((LL_1,(HL_1,LH_1,HH_1)),'haar')

        print(LL.shape,HL.shape,LH.shape,HH.shape)
        img[:size,:size]  = pywt.idwt2((LL,(HL, LH,HH)), 'haar')
        
        if len(ori_img.shape)==3:
            ori_img[:,:,0] =   img
        else:
            ori_img   = img

        #ori_img = cv2.cvtColor(ori_img, cv2.COLOR_YUV2BGR)
        return ori_img


    def extract(self,ori_wmimage,wm,key=10):
        #如果是rgb图像，使用红色分量
        #ori_wmimage = cv2.cvtColor(np.float32(ori_wmimage), cv2.COLOR_BGR2YUV)

        if len(ori_wmimage.shape)==3:
            wmimage = ori_wmimage[:,:,0]
        else:
            wmimage = ori_wmimage
       
        size = min(wmimage.shape[:2])
        
        LL,(HL,LH,HH) = pywt.dwt2(np.array(wmimage[:size,:size]),'haar')  #512*512
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(LL,'haar')    #256
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(LL_1,'haar')  #128 
        embed = HH_1
        w,h = embed.shape[:2]

        signature = self._gene_signature(wm,64,key) 

        ext_sig=np.zeros((w-4,h-4),dtype=np.int)
        Q = 10
        for i in range(w-4):
            for j in range(h-4):
                dct= cv2.dct(np.float32(embed[i:i+4,j:j+4]))
                if dct[2,2] > Q/2:
                    ext_sig[i,j] = 1
        max_sim = 0
        for i in range(w-12):
            for j in range(h-12):
                sim = np.sum(np.equal(signature,ext_sig[i:i+8,j:j+8]))/64
                if sim >max_sim:
                    max_sim =sim
                if max_sim > 0.9:
                    return max_sim

        return max_sim


class LsbWatermark:

    def _gene_signature(self,wm,key):
        '''提取特征，用来比对是否包含水印的，不用来恢复水印'''
        wm = cv2.resize(wm,(256,256))        
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
        return np.array(signature,dtype=np.int8)

    def embed(self,ori_img, wm, key=None):
        if len(ori_img.shape)==3:
            img  = ori_img[:,:,0] 
        else:
            img  = ori_img    
      
        #生成和嵌入特征        
        signature = self._gene_signature(wm,key).reshape((16,16))  

        w,h = img.shape[:2]
        if w < 16 or h < 16:
            logging.warn("输入图片大小是 : {} × {} ，小于 16×16 ，不能嵌入,原图返回。".format(w ,h))
            return ori_img

        #将图片分片，全部嵌入，值嵌入一个通道
        for i in range(0,w,16):
            for j in range(0,h,16):
                for ii in range(16):
                    for jj in range(16):
                        #print(ii,jj,img[ii,jj])
                        img[ii,jj] = imutil.set_bit(img[ii,jj],4,signature[ii][jj])
                        img[ii,jj] = imutil.set_bit(img[ii,jj],3,1)
                        img[ii,jj] = imutil.set_bit(img[ii,jj],2,1)
                        #print(ii,jj,img[ii,jj])
                        
        
        print(img[:16,:16])
        if len(ori_img.shape)==3:
            ori_img[:,:,0] =   img
        else:
            ori_img   = img

        return ori_img
        
    def ext_sig(self,img,size=16):
        w,h = img.shape[:2]
        ext_sig=[]
        print(img[:16,:16])
        for m in range(1): #,w-size):
            for n in range(1): #,h-size):
                one_sig=np.ones((size,size))
                for i in range(m,w,size):
                    for j in range(n,h,size):
                        for ii in range(size):
                            for jj in range(size):
                                one_sig[ii][jj] = imutil.get_bit(img[ii,jj],4)
                        ext_sig.append(one_sig)
        return ext_sig


    def extract(self,ori_wmimage,wm, key=None):
        '''
            提取LSB信息
        '''
        #如果是rgb图像，使用红色分量
        if len(ori_wmimage.shape)==3:
            wmimage = ori_wmimage[:,:,0]
        else:
            wmimage = ori_wmimage

        #生成和嵌入特征        
        signature = self._gene_signature(wm,key).reshape((16,16))  

        #提取嵌入的信息
        ext_sigs = self.ext_sig(wmimage,size=16)
        #ext_sigs.extend(self.ext_sig(np.rot90(wmimage,1)))
        #ext_sigs.extend(self.ext_sig(np.rot90(wmimage,2)))
        #ext_sigs.extend(self.ext_sig(np.rot90(wmimage,3)))

          #计算相似度
        similarity = 0 
        for sig in ext_sigs:
            print(sig)
            print(signature)
            one_similarity = list(np.array(sig.flatten()) - signature.flatten()).count(0) / len(signature.flatten())
            #logging.info('一个相似度 : {}'.format(one_similarity))
            similarity = max(similarity,one_similarity )
            break
        
        logging.debug('提取的信息和水印信息相似度是：%f (1最大，0最小，建议参考值是0.7)'  % (similarity))

        return similarity

class DwtsvdWatermark:
    ''' 盲水印，DWT+SVD '''      

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
        return np.array(signature,dtype=np.int8)

    def _embed_svd_sig(self,vec,signature):
        #特征是128位的，
        #vec 是128×128 ， 分成8×8的小图像，一共，256个
        #在vec的特征值里嵌入信息
        print(vec[:3,:3])
        Q= 32
        idx = 0 
        mask = np.random.random(size=64).reshape((8,8)) 
        for i in range(0,vec.shape[0],8):  #128*128
            for j in range(0,vec.shape[1],8):
                onepiece = vec[i:i+8,j:j+8]  #+ mask #遇到纯色的区域时，可以避免嵌入不了信息
                u,s,v = np.linalg.svd(np.mat(onepiece))
                eigen = s[0] + Q
                if idx +1 >=len(signature):
                    break
                eid = signature[idx]
                idx =idx +1
                z= eigen % Q
                t=eigen
                if  eid==1 and z < Q/4:
                    eigen =eigen -z - (Q/4)
                elif  eid==1 and z >= Q/4:
                    eigen = eigen -z + 3*(Q/4)
                elif  eid==0 and z < 3* (Q/4):
                    eigen = eigen -z + (Q/4)
                elif  eid==0 and z >= 3*(Q/4):
                    eigen = eigen -z + 5* (Q/4)
                s[0] = eigen
                print(eigen-t,t,eigen)
                vec[i:i+8,j:j+8] = u * np.diag(s) * v
        print(vec[:3,:3])
        return vec



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
            rate  = n//m
            for i in range(m):
                for j in range(rate):
                    bi_int_part[i+j*m][10] =signature[i]
            #for i in range(n-m*rate):
            #    bi_int_part[i+rate*m][10] =signature[i]
                

        #嵌入完成，组装回去
        em_int_part = []
        for i in range(len(bi_int_part)):
            s='0b'
            s+= (''.join([str(j) for j in bi_int_part[i]]))
            em_int_part.append(eval(s))
        
        em_combo = np.array(em_int_part) + np.array(frac_part)

        return  np.array([-1*em_combo[i] if combo_neg_idx[i]==1 else em_combo[i] for i in range(len(em_combo))]).reshape(shape)

    def _calc_var(self,nlist):
        '''计算方差，更具HVS，方差大的嵌入信息更不可见'''
        narray=np.array(nlist)
        mean=narray.sum()/len(nlist) 
        var = (narray*narray).sum()/len(nlist) - mean**2
        #print(nlist,mean,var)
        return var
   
    def _extract_sig(self,ext_sig,siglen):
        ext_sig = list(ext_sig.flatten())
        
        m = len(ext_sig)
        n = siglen
        ext_sigs=[]
        
        if n >= m :
            ext_sigs.append(ext_sig.extend([0] * (n-m)))
        
        if n < m:
            rate= m//n
            for i  in range(rate):
                ext_sigs.append( ext_sig[i*n:(i+1)*n])
                
        return ext_sigs

    def _extract_svd_sig(self,vec,siglen):
        Q = 32
        ext_sig=[]
      
        for i in range(0,vec.shape[0],8):  #128*128
            for j in range(0,vec.shape[1],8):
                u,s,v = np.linalg.svd(np.mat(vec[i:i+8,j:j+8]))
                z = s[0] % Q
                if z>=Q/2 :
                    ext_sig.append(1)                    
                else:
                    ext_sig.append(0)

        if siglen >len(ext_sig):
            logging.warning('extract svd sig is {},small  than needed {}'.format(len(ext_sig),siglen))
            ext_sig.extend([0] * (siglen - len(ext_sig)))
        else:
            ext_sig = ext_sig[:siglen]

        return [ext_sig]


##################################################################################################################################
    def embed(self, ori_img,wm,key=10):
        '''默认水印是灰度图像传进来的，即数据格式是2维的'''
        w,h = ori_img.shape[:2]
        ori_img = cv2.resize(ori_img,(512,512))
        wm = cv2.resize(wm,(64,64))
        
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YUV)
        #对于RGB图像使用红色通道的高频分量保存水印的特征值
        if len(ori_img.shape)==3:
            img  = ori_img[:,:,0] 
        else:
            img  = ori_img       

        LL,(HL,LH,HH) = pywt.dwt2(np.array(img),'haar')  #512*512
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(LL,'haar')    #256
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(LL_1,'haar')  #128 
        LL_3,(HL_3,LH_3,HH_3) = pywt.dwt2(LL_2,'haar')  #64
        LL_4,(HL_4,LH_4,HH_4) = pywt.dwt2(LL_3,'haar')  #32
       
        #高频的特征值水印，对图像变化不敏感。用来恢复水印使用
        iU,iS,iV = np.linalg.svd(np.mat(HH))
        wU,wS,wV = np.linalg.svd(np.mat(wm))
        '''
        if len(iS) <= len(wS):
            HH = iU * np.diag(wS[:len(iS)]) *iV 
        else:
            newIs = list(wS)
            newIs.extend([0] * (len(iS)-len(wS)))
            HH = iU * np.diag(newIs) *iV 
        '''
        #生成和嵌入特征
        signature = self._gene_signature(np.array(wU),np.array(wV),key)   

        #直接嵌入到高频域中
        print('-----> 1 ',HH_2[:5:5])
        bi_int_part,frac_part,combo_neg_idx,_ = self._gene_embed_space(HH_2)
        HH_2 = self._embed_sig(bi_int_part,frac_part,combo_neg_idx,signature)
        print('-----> 2 ',HH_2[:5:5])
        #嵌入到特征值中
        #LL_2 = self._embed_svd_sig(LL_2,signature)
       
        LL_3 = pywt.idwt2((LL_4,(HL_4,LH_4,HH_4)),'haar')
        LL_2 = pywt.idwt2((LL_3,(HL_3,LH_3,HH_3)),'haar')   
        LL_1 = pywt.idwt2((LL_2,(HL_2,LH_2,HH_2)),'haar')
        LL   = pywt.idwt2((LL_1,(HL_1,LH_1,HH_1)),'haar')
        img  = pywt.idwt2((LL,(HL, LH,HH)), 'haar')
        
        ####################
        img =np.round(img)
        print('===> 1',img[:5,:5])
        ori_img[:,:,0] = img
        print('===> 2',ori_img[:5,:5,0])
        
        LL,(HL,LH,HH) = pywt.dwt2(np.array(ori_img[:,:,0]),'haar')  #512*512
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(LL,'haar')    #256
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(LL_1,'haar')  #128 
        LL_3,(HL_3,LH_3,HH_3) = pywt.dwt2(LL_2,'haar')  #64
        LL_4,(HL_4,LH_4,HH_4) = pywt.dwt2(LL_3,'haar')  #32

        print('-----> 4 ',HH_4[:5:5])
        ###################


        #恢复图像
        if len(ori_img.shape)==3:
            ori_img[:,:,0] =   img
        else:
            ori_img   = img

        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_YUV2BGR)
        
        return cv2.resize(ori_img,(h,w)) 

    
    def extract(self,ori_wmimage,wm,key=10):
        ''' 检测是否包含水印，如果包含的话输出水印'''
        
        ori_wmimage = cv2.resize(ori_wmimage,(512,512))
        ori_wmimage = cv2.cvtColor(ori_wmimage, cv2.COLOR_BGR2YUV)
        
        w,h = wm.shape[:2]
        wm = cv2.resize(wm,(64,64))
        
        #如果是rgb图像，使用红色分量
        if len(ori_wmimage.shape)==3:
            wmimage = ori_wmimage[:,:,0]
        else:
            wmimage = ori_wmimage

        LL,(HL,LH,HH) = pywt.dwt2(wmimage,'haar') 
        LL_1,(HL_1,LH_1,HH_1) = pywt.dwt2(LL,'haar') 
        LL_2,(HL_2,LH_2,HH_2) = pywt.dwt2(LL_1,'haar') 
        LL_3,(HL_3,LH_3,HH_3) = pywt.dwt2(LL_2,'haar')
        LL_4,(HL_4,LH_4,HH_4) = pywt.dwt2(LL_3,'haar')
        iU,iS,iV = np.linalg.svd(np.mat(HH))

        #合成图像返回 (意义不大，接口不需要的时候可以去掉)
        wU,wS,wV = np.linalg.svd(np.mat(wm))
        if len(wS) <= len(iS):
            wS = iS[:len(wS)]
        else:
            wS = list(iS)
            wS.extend([0] * (len(wS) - len(iS)))
        em_rec_data = wU * np.diag(np.array(wS)) * wV
        

        #计算原始特征作对比
        signature = self._gene_signature (np.array(wU), np.array(wV), key)

        _,_,_,ori_sig = self._gene_embed_space(HH_2)
        #print('-----> 3 ',HH_4[:5:5])

        ext_sigs=[]
        ext_sigs.extend(self._extract_sig(ori_sig,len(signature)))
        ext_sigs.extend(self._extract_sig(np.rot90(ori_sig,1),len(signature)))
        ext_sigs.extend(self._extract_sig(np.rot90(ori_sig,2),len(signature)))
        ext_sigs.extend(self._extract_sig(np.rot90(ori_sig,3),len(signature)))
        
        #ext_sigs.extend(self._extract_svd_sig(LH_2,len(signature)))

        #计算相似度
        similarity = 0 
        for sig in ext_sigs:
            #print(sig)
            #print(list(signature))
            one_similarity = list(np.array(sig) - signature).count(0) / len(signature)
            #logging.info('一个相似度 : {}'.format(one_similarity))
            similarity = max(similarity,one_similarity )
        
        logging.debug('提取的信息和水印信息相似度是：%f (1最大，0最小，建议参考值是0.7)'  % (similarity))

        return  cv2.resize(em_rec_data,(h,w)),similarity
