import matplotlib
import numpy as np
from matplotlib import colors as mcolors
import tensorflow as tf
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = dict(((name,color))
                for name, color in colors.items())
color=[]
target=[]
label=[([]*2) for i in range(700)]
test=[([]*2) for i in range(70)]
list1=[]
list1_label=[]
list1_test=[]
target_label=[]
target_test=[]
target_label_tensor=[[[],[],[]]for i in range (700)]
a=open('target.txt')
i=0
b=a.readlines()
for line in b:
    target.append(line)
    target[i]=str(target[i]).split(' ')
    while '' in target[i]:   
        target[i].remove('')
    i=i+1
for name,hex in matplotlib.colors.cnames.items():
    color.append(name)
for x in range(len(target)):
    for y in range(len(target[x])):
        for z in target[x][y]:
            if z.isdigit()==True:
                target[x][y]='#'+target[x][y]
                target[x][0]=target[x][y]
                target[x][y]=target[x][2]
                break              
for x in range(len(target)):
    for y in range(len(target[x])):
        if (target[x][y] not in color)and(target[x][y][0]!='#'):
            target[x][y]=[]
for x in range(len(target)):
    while [] in target[x]:
        target[x].remove([])
for x in range(len(target)):
    for y in range(len(target[x])):
        if target[x][y] in by_hsv:
            target[x][y]=by_hsv[target[x][y]]
for x in range(len(target)):
    for y in range(len(target[x])):
            target[x][y]=tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(target[x][y])[:3]))
target_new=[([0] * 3) for i in range(len(target))]
for x in range(len(target)):
    target[x]=np.array(target[x])
    target_new[x]=sum(target[x])/len(target[x])

for x in range(766):
    target_new[x][0]=(target_new[x][0]*360)
    target_new[x][1]=(target_new[x][1]*100)
    target_new[x][2]=(target_new[x][2]*100)
    target_new[x]=target_new[x].astype(np.int32).tolist()
a=open('picture.txt')
i=0
b=a.readlines()
for line in b:
    list1.append(line)
    list1[i]=str(list1[i]).split(' ')
    while '' in list1[i]:   
        list1[i].remove('')
    i=i+1
    

for x in range(len(list1)):
    if ((float(list1[x][1]))>=0 and (float((list1[x][2]))>0)):
        list1[x][1]=[1,0,0,0,0]
        del list1[x][2]
    elif ((float(list1[x][1]))>=0 and (float(list1[x][2])<=0)):
        list1[x][1]=[0,1,0,0,0]
        del list1[x][2]
    elif ((float(list1[x][1]))<0 and (float(list1[x][2]))<=0):
        list1[x][1]=[0,0,1,0,0]
        del list1[x][2]
    elif ((float(list1[x][1]))+(float(list1[x][2]))<=0):
        list1[x][1]=[0,0,0,1,0]
        del list1[x][2]
    elif ((float(list1[x][1]))+(float(list1[x][2]))>0):
        list1[x][1]=[0,0,0,0,1]
        del list1[x][2]

for x in range(0,700):
    #list1[x][1]=tf.one_hot(list1[x][1],5)
    list1_label.append([list1[x][1]])
    target_label.append([target_new[x]])
    #target_label[x]=np.array(target_label[x]).T

for x in range(700,766):  
    list1_test.append([list1[x][1]])
    #list1[x]=tf.one_hot()
    target_test.append([target_new[x]])    

for x in range(0,66):
    test[x].append([target_test[x]])   
    test[x].append([list1_test[x]]) 

for x in range(0,700):
    label[x].append(target_label[x])   
    label[x].append(list1_label[x])
#print(target_label[1].shape)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy.eval,feed_dict={xs:v_xs,ys:v_ys})
    return result

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
#list1_test_one_hot=tf.one_hot(list1_test,5)
xs=tf.placeholder(tf.float32,[None,3])

ys=tf.placeholder(tf.float32,[None,5])
#ys_one_hot=tf.one_hot(ys,5)
prediction=add_layer(xs,3,5,activation_function=tf.nn.softmax)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))#loss

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(0,700):
    #target_label=tf.convert_to_tensor(target_label[i],dtype=tf.float32)
    #list1_label=tf.convert_to_tensor(list1_label[i],dtype=tf.float32)
    #batch_xs,batch_ys=label[i]
    sess.run(train_step,feed_dict={xs:target_label[i],ys:list1_label[i]})
    if i%70==0:
        target_test=tf.convert_to_tensor(target_test[i],dtype=tf.float32)
        list1_test=tf.convert_to_tensor(list1_test[i],dtype=tf.float32)
        print(compute_accuracy(target_test[i],list1_test[i]))

