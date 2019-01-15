import numpy as np

import numpy
world_alcohol=numpy.genfromtxt("world_alcohol.txt",delimiter=',')
print(type(world_alcohol))

vector=numpy.array([5,10,15,20])
matrix=numpy.array([[5, 10, 15], [20, 25, 30], [35, 40,45]])
print( vector)
print (matrix)

vector=numpy.array([1,2,3,4])
print(vector)
print(vector.shape)
matrix=numpy.array([[5, 10, 15], [20, 25, 30]])
print(matrix)
print(matrix.shape)

vector=numpy.array([1,2,3,4])
print(vector)
print(vector.dtype)
vector=numpy.array([1,2,3,4.0])
print(vector)
#类型全部统一
print(vector.dtype)


world_alcohol=numpy.genfromtxt("world_alcohol.txt",delimiter=',')
#由于数据类型统一，string会被强制转数字
print(world_alcohol)

#跳过第一行注解
world_alcohol=numpy.genfromtxt("world_alcohol.txt",delimiter=',',dtype='str',skip_header=1)
#由于数据类型统一，string会被强制转数字
print(world_alcohol)

#读取行列
uruguay_other_1986=world_alcohol[1,4]
print(uruguay_other_1986)

vector=numpy.array([1,2,3,4])
print(vector[0:3])

#从所有行去第1列
matrix=numpy.array([[5, 10, 15], [20, 25, 30], [35, 40,45]])
print(matrix[:,1])
print(matrix[:,0:2])

#判断数组每一个值是否与10等
vector=numpy.array([1,2,3,4,10])
equal10=vector==10
print(equal10)
print(vector[equal10])
#判断数组每一个值是否与10等
matrix=numpy.array([[5, 10, 15], [20, 25, 30], [35, 40,45]])
print(matrix)

equal25=matrix[:,1]==25
print(equal25)
#匹配的行
print(matrix[equal25])
#在所有列中找=25那一行
print(matrix[equal25,:])
#列出等于25的列
print(matrix[:,equal25])

vector=numpy.array([5,10,15,20])
equal_to_10_and_5=(vector==10)&(vector==5)
print(equal_to_10_and_5)
equal_to_10_or_5=(vector==10)|(vector==5)
print(equal_to_10_or_5)

vector=numpy.array([5,10,15,20])
#均值
print(vector.mean())
print(vector.max())
print(vector.min())
#print(help(numpy.array))

matrix=numpy.array([[5, 10, 15], [20, 25, 30], [35, 40,45]])
#每行相加
print(matrix.sum(axis=1))
#每列相加
print(matrix.sum(axis=0))

world_alcohol=numpy.genfromtxt("world_alcohol.txt",delimiter=',',dtype=str)
print(world_alcohol)
#nan_to_num
ls_value_empty=numpy.nan_to_num(world_alcohol[:,4])

print(numpy.arange(15))
a=np.arange(15).reshape(3,5)
print(a)

print(a.shape)
#a的维度
print(a.ndim)

a=np.arange(15).reshape(1,3,5)
print(a)

print(a.shape)
#a的维度
print(a.ndim)
#多少个元素
print(a.size)


a=np.zeros((3,4))
print(a)

a=np.ones((3,4))
print(a)

np.arange(12).reshape(3,4)
print(np)

#0~1之间
np.random.random((2,3))
#-1~1之间
np.random.random((2,3))*2-1

from numpy import pi
array=np.linspace(0,2*pi,100)
print(array)


a=np.array([20,30,40,50])
b=np.arange(4)
print(a)
print(b)

print(a-b)
print(a-1)
print(b**2)

#矩阵相乘，矩阵內积
A=np.array([[1,1],[0,1]])
B=np.array([[2,0],[3,4]])


print('---------')
print(A)
print('---------')
print(B)
print('----內积，对应位置相乘-----')
print(A*B)
print('----矩阵乘法，每个位置行乘列相加-----')
print(A.dot(B))
print('---------')
print(np.dot(A,B))

B=np.arange(3)
print(B)
#返回e的幂次方,e是一个常数为2.71828
print(np.exp(B))

#开根号
print(np.sqrt(B))

a=np.floor(10*np.random.random((3,4)))
print(a)
#把矩阵拉平
print(a.ravel())
print('---------')
a.shape=(6,2)
print(a)
print('-----转置----')
print((a.T))

#-1可以用来计算可计算的形状
a=np.arange(15).reshape(5,-1)
print(a)

a=np.floor(10*np.random.random((2,2)))
b=np.floor(10*np.random.random((2,2)))

print(a)
print(b)
print('-------按行拼接--------')
print(np.hstack((a,b)))
print('-------按列拼接--------')
print(np.vstack((a,b)))


a=np.floor(10*np.random.random((2,12)))
print(a)
print('-------切3份--------')
print(np.hsplit(a,3))
print('-------按行3和4之间切切份--------')
print(np.hsplit(a,(3,4)))
print('-------按列3之间切切份--------')
a=np.floor(10*np.random.random((12,2)))
print(a)
print(np.vsplit(a,3))
#深复制，防止共享内存
b=a.copy()

#获取max值的index
data=np.sin(np.arange(20)).reshape(5,4)
print(data)
#axis按列找
ind=data.argmax(axis=0)
print(ind)
#行列
print(data[1,3])
#每一列最大值打印出来
print(data[ind,range(data.shape[1])])

#a复制成一个矩阵
a=np.arange(0,40,10)
print(a)
b=np.tile(a,(1,4))
print(b)

c=np.tile(a,(2,4))
print(c)
print(c.shape)
print(c.ndim)