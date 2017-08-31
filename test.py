#coding=utf-8

import copy
import cv2
import time
import math
import glob
import numpy as np
import script

def psnr(im1,im2):
    if im1.shape != im2.shape or  len(im2.shape)<2:
        return 0
    
    di = im2.shape[0] * im2.shape[1]
    if len(im2.shape)==3:
        di = im2.shape[0] * im2.shape[1] *  im2.shape[2]
    
    diff = np.abs(im1 - im2)
    rmse = np.sum(diff*diff) /di
    print(rmse)
    psnr = 20*np.log10(255/rmse)
    return psnr

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def attack(fname,type):
    img = cv2.imread(fname)    
        
    if type == "ori":  
        return img 

    if type == "blur":  
        kernel = np.ones((5,5),np.float32)/25
        return cv2.filter2D(img,-1,kernel)

    if type=="rotate180":
        return rotate_about_center(img,180)
        
    if type=="rotate90":
        return rotate_about_center(img,90)
                
    if type=="chop10":
        w,h = img.shape[:2]
        return img[int(w*0.1):,:]
    
    if type=="chop5":
        w,h = img.shape[:2]
        return img[int(w*0.05):,:]

    if type=="chop30":
        w,h = img.shape[:2]
        return img[int(w*0.3):,:]
        
    if type == "gray":
        return  cv2.imread(fname,cv2.IMREAD_GRAYSCALE)    

    if type == "redgray":
        return  img[:,:,0]

    if type == "saltnoise":  
        for k in range(1000):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
        return img


    # if type == "vwm":
    #     vwm = script.VisWatermark 
    #     mark =  cv2.imread('./data/wm.png')  
    #     params = {}
    #     params['position']      = (30,30)
    #     img =vwm.watermark_image(img, mark, params)
    #     return img

    
    if type == "randline":  
        cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
        cv2.rectangle(img,(0,0),(300,128),(255,0,0),3)
        cv2.line(img,(0,0),(511,511),(255,0,0),5)
        cv2.line(img,(0,511),(511,0),(255,0,255),5)
        
        return img

    if type == "cover":  
        cv2.circle(img,(256,256), 63, (0,0,255), -1)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Just DO it ',(10,500), font, 4,(255,255,0),2)
        return img


    if type == "brighter10":  
        w,h = img.shape[:2]
        for xi in range(0,w):
            for xj in range(0,h):
                img[xi,xj,0] = int(img[xi,xj,0]*1.1)
                img[xi,xj,1] = int(img[xi,xj,1]*1.1)
                img[xi,xj,2] = int(img[xi,xj,2]*1.1)
        return img

    if type == "darker10":  
        w,h = img.shape[:2]
        for xi in range(0,w):
            for xj in range(0,h):
                img[xi,xj,0] = int(img[xi,xj,0]*0.9)
                img[xi,xj,1] = int(img[xi,xj,1]*0.9)
                img[xi,xj,2] = int(img[xi,xj,2]*0.9)
        return img


    if type == "largersize":  
        w,h=img.shape[:2]
        return cv2.resize(img,(int(h*1.5),w))

    if type == "smallersize":  
        w,h=img.shape[:2]
        return cv2.resize(img,(int(h*0.5),w))

    return img


attack_list ={}
attack_list['ori']          = '原图'
#attack_list['blur']         = '模糊'
attack_list['rotate180']    ='旋转180度'
attack_list['rotate90']     = '旋转90度'
attack_list['chop5']        = '剪切掉5%'
attack_list['chop10']       = '剪切掉10%'
attack_list['chop30']       = '剪切掉30%'
attack_list['saltnoise']    ='椒盐噪声'
attack_list['vwm']          = '增加明水印'
attack_list['randline']     = '随机画线'
attack_list['cover']        = '随机遮挡'
attack_list['brighter10']   = '亮度提高10%'
attack_list['darker10']     = '亮度降低10%'
#attack_list['largersize']   = '图像拉伸'
#attack_list['smallersize']  = '图像缩小'
#attack_list['gray']         ='自然灰度处理'
#attack_list['redgray']      ='红色灰度处理'


def test_blindwm(alg,imgname,wmname,times=1):
    handle = script.dctwm
    
    if alg == 'DCT':
        handle  = script.dctwm
    if alg == 'DWT':
        handle  = script.dwtwm

    print('\n##############测试'+alg+'盲提取算法，以及鲁棒性')

    btime=time.time() 
    for i in range(times):
        img = cv2.imread('./data/'+imgname)
        wm  = cv2.imread('./data/'+wmname,cv2.IMREAD_GRAYSCALE)
        wmd = handle.embed(img,wm)
        outname = './output/'+alg+'_'+imgname

    cv2.imwrite(outname,wmd)
    print('嵌入完成，文件保存在 :{},平均耗时 ：{} 毫秒 ,psnr : {}'.format(outname,int((time.time()-btime)*1000/times),psnr(img,wmd)))

    for  k,v in attack_list.items():
        wmd = attack(outname,k)
        cv2.imwrite('./output/attack/'+k+'_'+imgname,wmd)
        btime=time.time() 
        wm  = cv2.imread('./data/'+wmname,cv2.IMREAD_GRAYSCALE)
        sim = handle.extract(wmd,wm) 
        print('{:10} : 提取水印 {}，提取信息相似度是：{} ,耗时：{} 毫秒.'.format(v,'成功' if sim>0.7 else '失败'  ,sim,int((time.time()-btime)*1000)))

def test_report():
    #I:使用8张图片生成 8张水印图和 11×8种攻击后的图片
        # 包括黑底白字截图，不同大小的白底黑字截图、表格截图、人物照片、其他照片
        #攻击类型： 单边剪切 %3，%10，%30，提高亮度，降低亮度，随机画线、随机遮挡、全图增加噪点、旋转90度，旋转180度、
    #II:随机下载70张网络图片，包括不同大小的，大部分是文档和桌面截图，少部分是风景

    probsum = 0 
    maxsim= 0 
    num = 0
    for name in glob.glob('./output/test/*'):
        wmd =cv2.imread(name)
        wm  = cv2.imread('./data/wm.png',cv2.IMREAD_GRAYSCALE)
        sim = script.dctwm.extract(wmd,wm) 
        probsum+=sim
        maxsim= max(maxsim,sim)
        num+=1
        print ('{}  has wm prob : {}'.format(name,sim))
    print('avg prob {},max prob {}'.format(probsum/num,maxsim))

    probsum = 0 
    minsim= 1.0 
    num = 0
    for name in glob.glob('./output/attack/*'):
        wmd =cv2.imread(name)
        wm  = cv2.imread('./data/wm.png',cv2.IMREAD_GRAYSCALE)
        sim = script.dctwm.extract(wmd,wm) 
        probsum+=sim
        minsim= min(minsim,sim)
        num+=1
        print ('{}  has wm prob : {}'.format(name,sim))
    print('avg prob {} ,min prob {}'.format(probsum/num,minsim))



    #1 :召回率

    #2 :准确率

    #3 :时间性能
    # 1024×1023

    #200×500

    #100×100

    #32×32


if __name__ == '__main__':  
    test_blindwm('DCT','ts.jpg','wm.png')
    test_blindwm('DCT','lena.jpg','wm.png')
    test_blindwm('DCT','ts.jpg','wm.png')
    test_blindwm('DCT','tm.jpg','wm.png')
    test_blindwm('DCT','ta.png','wm.png')
    test_blindwm('DCT','tb.jpg','wm.png')
    test_blindwm('DCT','td.jpg','wm.png')
    test_blindwm('DCT','ss.jpg','wm.png')
    test_blindwm('DCT','bm.jpg','wm.png')

    test_report()
    # test_blindwm('DWT','lena.jpg','wm.png')
    # test_blindwm('DWT','tm.jpg','wm.png')
    # test_blindwm('DWT','ts.jpg','wm.png')
    # test_blindwm('DWT','td.jpg','wm.png')
