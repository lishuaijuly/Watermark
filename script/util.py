import hashlib
from Crypto import Random
from Crypto.Cipher import AES

'''
Thanks to
http://stackoverflow.com/questions/12524994/encrypt-decrypt-using-pycrypto-aes-256
'''
class AESCipher:

    def __init__(self, key): 
        self.bs = 16	# Block size
        self.key = hashlib.sha256(key.encode()).digest()	# 32 bit digest

    def encrypt(self, raw):
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(self._pad(raw))

    def decrypt(self, enc):
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:]))

    def _pad(self, s):
        return s + bytes((self.bs - len(s) % self.bs) *  [self.bs - len(s) % self.bs])

    @staticmethod
    def _unpad(s):
        return s[: -ord(s[len(s)-1:]) ]


def set_bit(n, i, x):
    '''
        n的第i位设置为x
    '''
    mask = 1 << i
    n &= ~mask
    if x:
        n |= mask
    return n

def get_bit(n, i):
    '''
        获取n的第i位
    '''
    t = n>>i    
    return bin(t)[-1]

def intToBytes(value):
    bs = []  
    bs.append (value & 0x000000FF)        
    bs.append ((value & 0x0000FF00)>>8)
    bs.append((value & 0x00FF0000)>>16  )
    bs.append((value & 0xFF000000)>>24)
    return bs 

def bytesToInt (bs,offset=0 ):
    return   (bs[offset]&0xFF)| ((bs[offset+1]<<8) & 0xFF00)  | ((bs[offset+2]<<16)& 0xFF0000)   | ((bs[offset+3]<<24) & 0xFF000000)

def gen_signature(wm,key):
        '''提取特征，用来比对是否包含水印的，不用来恢复水印'''
        wU,wS,wV = np.linalg.svd(np.mat(wm))
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
        return np.array(signature)