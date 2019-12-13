import numpy as np 
import pandas as pd 
import os
from progress.bar import Bar
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
from datetime import datetime
import multiprocessing as mtp
import random

def generate_msgdata(p):
    p[p>0]=1
    p=p.astype(int)
    for i in range(len(p)):
        if i==0:
            idx=[i]
        elif not np.array_equal(p[i],p[i-1]):
            idx.append(i)
    p=p[idx]

    for i in range(len(p[0])):
        if i==0:
            idx=[i]
        elif not np.array_equal(p[:,i],p[:,i-1]):
            idx.append(i)
    p=p[:,idx]

    if np.all(p[0]==0): p=p[1:]
    if np.all(p[-1]==0) and  random.random() > 0.5: p=p[:-1]
    if np.all(p[:,0]==0): p=p[:,1:]
    if np.all(p[:,-1]==0): p=p[:,:-1]
    

    cY=p.shape[0]
    cX=p.shape[1]

    if np.sum(p[1::2])>0 and np.sum(p[0::2])>0:
        valid=0
    else:
        valid=1


    topoSig=''.join(map(str, p.flatten()))

    return [topoSig, cX, cY, valid]





class hsd:
    def __init__(self, input_type, mode):
        self.input_type=input_type
        if input_type=='euv':
            self.model = squish_dl()


class squish_dl:
    def __init__(self, 
                 init_lr=0.001,
                 lr_decay=0.7,
                 max_iter=6000,
                 batch_size=64,
                 test_batch_size=300,
                 img_size=16,
                 img_channel=1,
                 lr_decay_itr=2000,
                 sv_itr=1000,
                 show_itr=50,
                 resnet=True, 
                 gpu_id='0',
                 model_path='./models/'):
        #neural networks spec
        self.init_lr=init_lr
        self.lr_decay=lr_decay
        self.lr_decay_itr=lr_decay_itr
        self.max_iter=max_iter
        self.batch_size=batch_size
        self.img_size=img_size
        self.img_channel=img_channel
        self.sv_itr=sv_itr  #iters to save checkpoint
        self.resnet=int(resnet)
        self.show_itr=show_itr
        self.model_path=model_path
        self.gpu_id=gpu_id
        self.noise=tf.placeholder(tf.float32, shape=[None, 64])
        self.lr_placeholer=tf.placeholder(tf.float32, shape=[])
        self.input_placeholder=tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.img_channel))
        #self.label_placeholder=tf.placeholder(tf.float32, shape=(None, 2))
    def processlabel(self, label, cato=2, delta1 = 0, delta2=0):
        softmaxlabel=np.zeros(len(label)*cato, dtype=np.float32).reshape(len(label), cato)
        for i in range(0, len(label)):
            if int(label[i])==0:
                softmaxlabel[i,0]=1-delta1
                softmaxlabel[i,1]=delta1
            if int(label[i])==1:
                softmaxlabel[i,0]=delta2
                softmaxlabel[i,1]=1-delta2
        return softmaxlabel
    def loss_to_bias(self, loss, alpha=1, threshold=0.3):
        if loss >= threshold:
            bias = 0
        else:
            bias = 1.0/(1+np.exp(alpha*loss))
        return bias
    def squish2img(self, topo, delta_x, delta_y, dir):
        tmp=Image.new(mode='L', size=(80,80))
        x0=0
        y0=0
        draw=ImageDraw.Draw(tmp)
        for i in range(len(delta_x)):
            for j in range(len(delta_y)):
                x1 = x0+delta_x[i]
                y1 = y0+delta_y[j]

                if topo[j, i]>0 and delta_x[i]>0 and delta_y[j]>0:
                    draw.rectangle([x0,y0,x1,y1], fill=255)
                y0=y1

            x0=x1
            y0=0
        if not dir==False:
            tmp.save(dir)
        else:
            return np.array(tmp)
    def run(self, data):
        self.build_model(True)
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list =self.gpu_id
        delta=np.ones(16) * 5

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=100)
            lr=self.init_lr
            for step in range(self.max_iter):
                train_data = data.getTrainBatchDP(self.batch_size)
                recon_loss, recon, _ = sess.run([self.recon_loss, self.reconstruct, self.op], feed_dict={
                                                 self.input_placeholder: np.expand_dims(train_data[:,:,:,0], axis=-1), 
                                                 self.lr_placeholer: lr})
                if step>0 and step % self.lr_decay_itr==0:
                    lr = lr* self.lr_decay
                if step % self.show_itr == 0:
                    print("%s: Step[%g/%g], Loss[%f]"%(datetime.now(), step, self.max_iter, recon_loss))

                if step % self.sv_itr==0 or step==self.max_iter-1:
                    test_data = data.getTestBatchDP(batch_size=self.batch_size)
                    test_recon = sess.run(self.reconstruct, feed_dict={
                                                 self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                                                 self.lr_placeholer: lr})
                    saver.save(sess, self.model_path+'step-'+str(step))
                    for s in range(10):
                        topo_in = test_data[s,:,:,0]
                        topo_out = test_recon[s,:,:,0]
                        in_dir = os.path.join(self.model_path,'sample/'+'step-'+str(step)+'-in-'+str(s)+'.png')
                        out_dir = os.path.join(self.model_path,'sample/'+'step-'+str(step)+'-out-'+str(s)+'.png')
                        self.squish2img(topo_in, delta, delta, in_dir)
                        self.squish2img(topo_out, delta, delta, out_dir)
          
    def test(self, data):
        self.build_model(False)
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list =self.gpu_id
        delta = np.ones(16)*5

        with tf.Session(config=config) as sess:
            idx=1
            saver = tf.train.Saver(max_to_keep=100)
            ckpt=tf.train.get_checkpoint_state(self.model_path)
            ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
            print (ckpt_name)
            saver.restore(sess, os.path.join(self.model_path, ckpt_name))
            test_path=os.path.join(self.model_path, 'test/')
            test_data = data.getTestBatchDP(batch_size=10)

            topo_in = test_data[idx,:,:,0]
            
            #original input
            print("Step 1")
            self.squish2img(topo_in, delta, delta, test_path+'origin.png')
            #original reconstruction
            test_recon, fm = sess.run([self.reconstruct, self.fm], feed_dict={
                                  self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                                  self.noise: np.zeros((10, 32))*1.0})

            self.squish2img(test_recon[idx,:,:,0], delta, delta, test_path+'recon.png')

            #no latent var
            test_recon= sess.run(self.reconstruct, feed_dict={
                                  self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                                  self.noise: -fm})

            self.squish2img(test_recon[idx,:,:,0], delta, delta, test_path+'recon_no_lat.png')

            #add gaussian noise
            print("Step 2")
            barat = Bar('Transforming with Gaussian', max=100)
            f=plt.figure(figsize=(80,80))
            axes=[f.add_subplot(10,10,i+1) for i in range(100)]


            name = test_path+'fm-gaussian.pdf'
            for a in axes:
                noise=np.random.normal(size=(10,32))

                noise_recon = sess.run(self.reconstruct, feed_dict={
                                    self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                                    self.noise: noise})
                
                tmp_img=self.squish2img(noise_recon[idx,:,:,0], delta, delta, dir=False)
                a.imshow(tmp_img, cmap='gray')
                a.axis('off')            
                a.set_aspect('equal')  
                barat.next()   
            f.subplots_adjust(hspace=0, wspace=0)     
            f.savefig(name)
            f.clf()
            plt.close()
            barat.finish() 
 
            #what does individual feature map represent?
            print("Step 3")
            size_x=320
            size_y=20

            barat = Bar('Transforming Latent Feature Maps', max=100)
            f=plt.figure(figsize=(80,80))
            axes=[f.add_subplot(10,10,i+1) for i in range(100)]
            name = test_path+'fm-affine.pdf'

            for x in range(10):     
                for y in range(10):
                    noise = np.zeros((10,32))
                    a=axes[x*10+y]
                    fm_idx=x
                    noise[:,fm_idx]=noise[:,fm_idx]+y-5.0
                    tmp_recon = sess.run(self.reconstruct, feed_dict={
                                    self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                                    self.noise: noise})
                    tmp_img=self.squish2img(tmp_recon[idx,:,:,0], delta, delta, dir=False)
                    a.imshow(tmp_img, cmap='gray')
                    a.axis('off')            
                    a.set_aspect('equal')  
                    barat.next()   
                    
        
            f.subplots_adjust(hspace=0, wspace=0)     
            f.savefig(name)
            f.clf()
            plt.close()
            barat.finish() 
       
            #linear combination of  patterns
            barat = Bar('Transforming Latent Feature Maps', max=100)
            name = test_path+'linear-comb'+'.pdf'
            fm = sess.run(self.fm, feed_dict={
                    self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                    self.noise: np.zeros((10, 32))*1.0})
            print (fm[0])
            f=plt.figure(figsize=(80,80))
            axes=[f.add_subplot(1,10,i+1) for i in range(10)]
            i=0
            for a in axes:
                alpha=i/10.0
                noise=np.tile(np.expand_dims(fm[0]*(alpha-1)+(1-alpha)*fm[3], axis=0), reps=(10,1))
                tmp_recon = sess.run(self.reconstruct, feed_dict={
                                self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                                self.noise: noise})                    
                tmp_img=self.squish2img(tmp_recon[idx,:,:,0], delta, delta, dir=False)
                a.imshow(tmp_img, cmap='gray')
                a.axis('off')
                a.set_aspect('equal')  
                i+=1                
                barat.next()
            f.subplots_adjust(hspace=0, wspace=0)     
            f.savefig(name)
            f.clf()
            plt.close()
            barat.finish()
            

            p=mtp.Pool(8)
            #Noise Final
     
            test_data_all = data.getTestBatchDP(batch_size=1000)
            num_noise=1000
            for ii in range(100):
                test_data=test_data_all[ii*10:(ii+1)*10]
                bar= Bar('Enumerating Noises', max=num_noise)
                for i in range(num_noise):
                    noise=np.random.normal(size=(10,32))
                    noise_recon = sess.run(self.reconstruct, feed_dict={
                                        self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                                        self.noise: noise})
                    if i == 0:
                        noise_patterns = noise_recon[:,:,:,0]
                        self.squish2img(noise_recon[0,:,:,0], delta, delta, dir=test_path+'noise11111.png')
                    else:
                        noise_patterns = np.concatenate((noise_patterns, noise_recon[:,:,:,0]), axis=0)
                    
                    bar.next()
                bar.finish()
                tmp = []
                tmp.append(p.map(generate_msgdata, noise_patterns))
                tmp=tmp[0]
         
                noise_df=pd.DataFrame(tmp, columns=['topoSig','cX','cY','valid'])
                noise_df.to_msgpack(test_path+'noise_data_'+str(ii)+'.msgpack')


            
            #Span of the Training set

            test_data=data.get_batch_with_same_cplx_beta()
            #10 16 16 3
            fm = sess.run(self.fm, feed_dict={
                          self.input_placeholder: np.expand_dims(test_data[:,:,:,0], axis=-1), 
                          self.noise: np.zeros((10,32))*1.0})
            num_span=100000
            bar = Bar('Enumerating Span', max=num_span)
            #fms = np.tile(np.expand_dims(fm, axis=0), reps=(10,1,1))
            input = np.zeros((10,16,16,1))*1.0
            
            for i in range(num_span):
                noise = np.zeros((10,32))*1.0
                for j in range(10):
                    comb = np.random.randint(100, size=(10,1))*1.0
                    comb = comb/np.sum(comb) #10 1
                    noise[j] = np.sum(fm* comb, axis=0)
                sup_recon = sess.run(self.reconstruct, feed_dict={
                                    self.input_placeholder: input, 
                                    self.noise: noise})
                if i % 1000 == 0:
                    span_patterns = sup_recon[:,:,:,0]
                else:
                    span_patterns = np.concatenate((span_patterns, sup_recon[:,:,:,0]), axis=0)
                if (i+1) % 1000 == 0:
                    id = int(i / 1000)
                    tmp=[]
                    tmp.append(p.map(generate_msgdata, span_patterns))
                    tmp=tmp[0]
              
                    span_df=pd.DataFrame(tmp, columns=['topoSig','cX','cY','valid'])
                    span_df.to_msgpack(test_path+'span_data_'+str(id)+'.msgpack')   
                bar.next()
            bar.finish()
        

    def build_model(self, is_training):
        #self.feature=self.discriminator(self.input_placeholder, is_training)
        #if is_training:
        #    self.reconstruct=self.generator(self.feature, is_training)
        #    self.recon_loss=tf.reduce_mean(tf.squared_difference(self.input_placeholder, self.reconstruct))
        #    self.op=tf.train.RMSPropOptimizer(self.lr_placeholer).minimize(self.recon_loss)
        #else:
        #    self.reconstruct=self.generator(self.feature+self.noise, is_training)

        if is_training:
            self.reconstruct = self.cae(input=self.input_placeholder, is_training=True)
            self.recon_loss=tf.reduce_mean(tf.squared_difference(self.input_placeholder, self.reconstruct))
            self.op=tf.train.RMSPropOptimizer(self.lr_placeholer).minimize(self.recon_loss)
        else:
            self.input_placeholder=tf.placeholder(tf.float32, shape=(10, self.img_size, self.img_size, self.img_channel))
            self.noise=tf.placeholder(tf.float32, shape=[None, 32])
            self.reconstruct = self.cae(input=self.input_placeholder, is_training=False, noise=self.noise)
        
    def cae(self, input, is_training, noise=0):
        net=input
        with tf.variable_scope('cae', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, stride=1, padding='SAME',
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                biases_initializer=tf.constant_initializer(0.0),
                                weights_regularizer=slim.l2_regularizer(0.001)):
                #net = slim.conv2d(net, 64, [5, 5], scope='conv0_1')
                #net = slim.conv2d(net, 64, [5, 5], scope='conv0_2')
                #net = slim.conv2d(net, 64, [5, 5], scope='conv0_3')
                print (net.get_shape())
                self.pool1 = slim.conv2d(net, 128, [5, 5], stride=2, scope='pool1') #8 8 
                net=self.pool1
                #net = slim.conv2d(self.pool1, 128, [5, 5], scope='conv1_1')
                #net = slim.conv2d(net, 128, [5, 5], scope='conv1_2')
                #net = slim.conv2d(net, 128, [5, 5], scope='conv1_3')
                print (net.get_shape())     
                self.pool2 = slim.conv2d(net, 256, [5, 5], stride=2, scope='pool2') #4 4 
                net = self.pool2
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                    weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                    biases_initializer=tf.constant_initializer(0.0),
                    weights_regularizer=slim.l2_regularizer(0.01)):
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1024, scope='fc1')
                if is_training:
                    self.fm = slim.fully_connected(net, 32, scope='fc2')
                else:
                    self.fm = slim.fully_connected(net, 32, scope='fc2') + noise
                net =self.fm
                net = slim.fully_connected(net, 1024, scope='fc3')
                net = slim.fully_connected(net, 4*4*256, scope='fc4')
                net = tf.reshape(net, shape=self.pool2.get_shape())
                print (net.get_shape())
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=tf.nn.relu, padding='SAME', stride=1,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                biases_initializer=tf.constant_initializer(0.0),
                                weights_regularizer=slim.l2_regularizer(0.001)):
                self.upool2 = slim.conv2d_transpose(net, 128, [5, 5], stride=2, padding='SAME', scope='upool2') #8 8
                #net = slim.conv2d(self.upool2, 128, [5, 5], scope='gconv1_1')
                #net = slim.conv2d(net, 128, [5, 5], scope='gconv1_2') #
                #net = slim.conv2d(net, 128, [5, 5], scope='gconv1_3')
                print (net.get_shape())
                net = self.upool2
                self.upool1 = slim.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', scope='upool1') #16 16
                net = self.upool1
                print (net.get_shape())
                #net = slim.conv2d(self.upool1, 64, [5, 5], scope='gconv0_1')
                #net = slim.conv2d(net, 64, [5, 5], scope='gconv0_2')

                #net = slim.conv2d(net, self.img_channel, [5, 5], scope='gconv0_3')

            return net   












