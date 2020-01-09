'''
                  Scale-Invariant Feature Transform  尺度不变特征变换

原理：  对于之前的关于角点检测的Harris方法等，它们具有旋转不变特性，即使图片发生了旋转，
       也能找到同样的角点，意味着图片旋转后角点依然是角点。但是对图片进行缩放后，角点就不一定就是
       角点了。2004年的SIFT算法可以帮助我们提取图像中的关键点和计算它们的描述符

不同的尺度空间不能使用相同的窗口检测极值点，对小的角点用小的窗口，对大的角点使用大的窗口，为了达到这个目的
要是用尺度空间滤波器(该滤波器可以使用一些具有不同方差a的高斯卷积核构成)，使用具有不同方差值a的高斯拉普拉斯
算子(LoG)对图像进行卷积，LoG由于具有不同的方差值a所以可以用来检测不同大小的斑点（当LoG的方差a与斑点直径
相等时能够使斑点完全平滑），简单来说方差a就是一个尺度变换因子，小方差的高斯卷积核可以很好的检测出小的角点
大方差的高斯卷积核可以很好的检测出大的角点，所以在尺度空间和二维平面中检测到局部最大值(x,y,a)，表示在a尺度
中（x,y）可能是一个关键点，高斯方差大小与窗口的大小存在一个倍数关系，窗口大小等于6倍方差加1，所以方差大小
也决定了窗口的大小。

由于LoG的计算量非常大，所以SIFT算法使用高斯差分算子DoG对LoG做近似，可以通过减少采样(如只取奇数行或奇数列)
来构成一组图像尺寸(1,0.5,0.25等)不同的金字塔，然后对这一组图像中的每一张图像使用具有不同方差a的高斯卷积核
构建出具有不同分辨率的图像金字塔（不同的尺度空间），DoG就是这组具有不同分辨率的图像金字塔中相邻两层之间
的差值

在DoG搞定之后，就可以在不同的尺度空间和2D平面中搜索局部最大值了。对于图像中的一个像素而言，他需要与自己
周围的8领域以及尺度尺度空间中上下两层中的18（2x9）个点相比，如果是局部最大值，它就可能是一个关键点，
基本上来说关键点是图像在相应尺度空间中的最好代表。

SIFT算法作者给出SIFT参数的经验值：octaves=4,通过降采样从而减小图像的尺寸，构成尺寸减小的图像金字塔4层，
尺度空间为5，也就是每个尺寸使用5个不同方差的高斯核进行卷积，初始方差是1.6，k等于根号2等

关键点（极值点）定位：   
                       一旦找到关键点，就要对它们进行修正从而得到更准确的结果，作者使用尺度空间的泰勒
                       级数展开来获得极值的准确位置，如果极值点的灰度值小于阈值就会被忽略掉，opencv中
                       这种阈值被称为contrastThreshold.
                       DoG算法对边界很敏感，所以必须要把边界去除，Harris算法除了用于角点检测之外，
                       还可以用于检测边界，作者就是使用了同样的思路，作者使用2x2的Hessian矩阵计算
                       曲率，从Harris角点检测算法中，知道当一个特征值远远大于另外一个特征值时检测到
                       的是边界，使用了一个简单函数，如果比例高于阈值（opencv中称为边界阈值），这个
                       关键点就会被忽略，文章中给出的边界阈值是10
                       所以低对比度的关键点和边界关键点都会被去除，剩下来的就是感兴趣的关键点了。
为关键点指定方向参数

关键点描述符

关键点匹配：
             采用关键点特征向量的欧氏距离来作为两幅图像中关键点的相似性判定度量，取第一个图的某个
             关键点，通过遍历找到第二幅图像中的距离最近那个关键点，但有些情况第二个距离最近的关键点
             与第一个距离最近的关键点靠的太近，可能是噪声等引起的，所以此时要计算第二近距离的比值，
             如果比值大于0.8，就忽略掉，这会去除90%的错误匹配，同时只去除5%的正确匹配
        

'''

from PIL import Image
from numpy import *
from pylab import *
import os

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
    """ Process an image and save the results in a file. """

    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')  #.convert('L') 将RGB图像转为灰度模式，灰度值范围[0,255]
        im.save('tmp.pgm')                       #将灰度值图像信息保存在.pgm文件中
        imagename = 'tmp.pgm'
   
    cmmd = str("G:/Opencv/SIFT_python/sift.exe "+imagename+" --output="+resultname+
                " "+params)
    os.system(cmmd)                              #执行sift可执行程序，生成resultname(test.sift)文件
    print('processed', imagename, 'to', resultname)


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    
    f = loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors


def plot_features(im,locs,circle=True):
    """ Show image with features. input: im (image as array), 
        locs (row, col, scale, orientation of each feature). """

    def draw_circle(c,r):
        t = arange(0,1.01,.01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x,y,'b',linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2]) 
    else:
        plot(locs[:,0],locs[:,1],'ob')
    axis('off')

 

if __name__ == '__main__':
    imname = ('G:/Opencv/images/home.jpg')               #待处理图像路径
    im=Image.open(imname)
    process_image(imname,'test.sift')
    l1,d1 = read_features_from_file('test.sift')           #l1为兴趣点坐标、尺度和方位角度 l2是对应描述符的128 维向
    figure()
    gray()
    plot_features(im,l1,circle = True)
    title('sift-features')
    show()