#coding=utf-8

import copy
import cv2
import time

from  script.watermark_invisiable import  LsbWatermark,DwtsvdWatermark
from script.watermark_visiable  import VisWatermark


def test_visiable_watermark():
    vwm = VisWatermark()  
    
    print('\n##################测试默认的水印效果')
    btime=time.time()
    im =  cv2.imread('./data/lena.png')
    mark =  cv2.imread('./data/wm.png')  
    params = {}
    t1 =vwm.watermark_image(im, mark, params)
    cv2.imwrite('./output/v_default.jpg',t1) 
    print('保存到：{},耗时:{} 毫秒'.format('./output/v_default.jpg',int((time.time()-btime)*1000)))

    print('\n##################测试固定水印位置')
    btime=time.time()
    im =  cv2.imread('./data/lena.png')
    mark =  cv2.imread('./data/wm.png')  
    params = {}
    params['position']      = (30,30)
    t2 =vwm.watermark_image(im, mark, params)
    cv2.imwrite('./output/v_pos.jpg',t2) 
    print('保存到：{},耗时:{} 毫秒'.format('./output/v_pos.jpg',int((time.time()-btime)*1000)))

    print('\n##################测试文字转图片')
    btime=time.time()
    t2mparams={}   
    twm = vwm.text2image(im, "合肥Syni", t2mparams)
    twm.save('./output/textimage.jpg')
    print('保存到：{},耗时:{} 毫秒'.format('./output/textimage.jpg',int((time.time()-btime)*1000)))

    twm = cv2.imread('./output/textimage.jpg')
    im =  cv2.imread('./data/lena.png')
    params = {}
    params['position']=(30,30)    
    t4=vwm.watermark_image(im, twm, params) # coords=None,font =font, color=(250,240,230))
    cv2.imwrite('./output/v_text_scale.jpg',t4)

def test_invisiable_watermark():
    img = cv2.imread('./data/lena.png')    

    print('\n##################测试LSB算法')
    btime=time.time()
    lsb = LsbWatermark()
    enc_wmdata="要加密的水印".encode()
    img2 = lsb.embed(img,enc_wmdata,'9527')
    cv2.imwrite('./output/lsb_lena.png',img2)  #只能保存为png，jpg压缩会导致解码不出来。
    print('嵌入成功，文件保存在：{},耗时 ：{} 毫秒'.format('./output/lsb_lena.png',int((time.time()-btime)*1000)))

    btime=time.time()  
    dec_wmdata = lsb.extract(cv2.imread('./output/lsb_lena.png'),'9527')
    print("提取的信息是： {}  ,耗时 ：{} 毫秒".format(dec_wmdata.decode(),int((time.time()-btime)*1000)))


    print('\n##################测试盲提取算法，以及鲁棒性')
    btime=time.time() 
    img = cv2.imread('./data/lena.png',cv2.IMREAD_GRAYSCALE) 
    wm  = cv2.imread('./data/wm.png',cv2.IMREAD_GRAYSCALE)/255
    ds  = DwtsvdWatermark()
    ds1 = ds.embed(img,wm)
    cv2.imwrite('./output/dwt_lena.jpg',ds1)
    print('嵌入完成，文件保存在 :{},耗时 ：{} 毫秒'.format('./output/dwt_lena.jpg',int((time.time()-btime)*1000)))

    btime=time.time() 
    wmd = cv2.imread('./output/dwt_lena.jpg',cv2.IMREAD_GRAYSCALE)
    #wmd = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)
    #wmd = attack(wmd,'chop')

    wm  = cv2.imread('./data/wm.png',cv2.IMREAD_GRAYSCALE)/255
    wmr,score = ds.extract(wmd,wm) 
    if score >0.7:
        cv2.imwrite('./output/dwt_wm.jpg',wmr)
        print('提取水印完成，保存在 {} ,耗时：{} 毫秒'.format('./output/dwt_wm.jpg',int((time.time()-btime)*1000)))

def attack(img,type):
    if type == "blur":
        kernel = np.ones((5,5),np.float32)/25
        return cv2.filter2D(img,-1,kernel)

    if type == "resize":
        return cv2.resize(img,(256,256))

    if type=="rotate": #不支持
        rows,cols=img.shape
        M=cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        return cv2.warpAffine(img,M,(cols,rows))

    if type=="chop":
        return img[:500,:]
    return img


if __name__ == '__main__':  
    test_visiable_watermark()
    test_invisiable_watermark()