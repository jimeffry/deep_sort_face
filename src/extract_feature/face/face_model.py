# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/07/2 14:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  caffe and tensorflow
####################################################
import sys
import os
import cv2
os.environ['GLOG_minloglevel'] = '3'
try:
    import caffe
except:
    pass
import mxnet as mx
import numpy as np
import time
sys.path.append(os.path.join(os.path.dirname(__file__),'../../configs'))
from face_config import config
#import mxnet as mx


def L2_distance(feature1,feature2,lenth):
    print("feature shape: ",np.shape(feature1))
    feature1 = np.asarray(feature1)
    feature2 = np.asarray(feature2)
    [len1,] = feature1.shape
    [len2,] = feature2.shape
    assert len1 ==lenth and len2==lenth
    if 1:
        f_mod1 = np.sqrt(np.sum(np.power(feature1,2)))
        f_mod2 = np.sqrt(np.sum(np.power(feature2,2)))
        feature1/=f_mod1
        feature2/=f_mod2
        loss = np.sqrt(np.sum(np.power((feature1-feature2),2)))
    else:
        mean1 = np.mean(feature1)
        mean2 = np.mean(feature2)
        print("feature mean: ",mean1, mean2)
        f_center1 = feature1-mean1
        f_center2 = feature2-mean2
        std1 = np.sum(np.power(f_center1,2))
        std2 = np.sum(np.power(f_center2,2))
        std1 = np.sqrt(std1/lenth)
        std2 = np.sqrt(std2/lenth)
        norm1 = f_center1/std1
        norm2 = f_center2/std2
        loss =np.sqrt(np.sum(np.power((norm1-norm2),2))/lenth)
    return loss

def get_by_ratio(x,new_x,y):
    ratio = x / float(new_x)
    new_y = y / ratio
    return np.floor(new_y)

def Img_Pad(img,crop_size):
    '''
    img: input img data
    crop_size: [h,w]
    '''
    img_h,img_w = img.shape[:2]
    d_h,d_w = crop_size
    pad_l,pad_r,pad_u,pad_d = [0,0,0,0]
    if img_w > d_w or img_h > d_h :
        if img_h> img_w:
            new_h = d_h
            new_w = get_by_ratio(img_h,new_h,img_w)
            if new_w > d_w:
                new_w = d_w
                new_h = get_by_ratio(img_w,new_w,img_h)
                if new_h > d_h:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_u = np.round((d_h - new_h)/2.0)
                    pad_d = d_h - new_h - pad_u
            else:
                pad_l = np.round((d_w - new_w)/2.0)
                pad_r = d_w - new_w - pad_l
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
        else:
            new_w = d_w
            new_h = get_by_ratio(img_w,new_w,img_h)
            if new_h > d_h:
                new_h = d_h
                new_w = get_by_ratio(img_h,new_h,img_w)
                if new_w > d_w:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_l = np.round((d_w - new_w)/2.0)
                    pad_r = d_w - new_w - pad_l
            else:
                pad_u = np.round((d_h - new_h)/2.0)
                pad_d = d_h - new_h - pad_u
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
    elif img_w < d_w or img_h < d_h:
        if img_h < img_w:
            new_h = d_h
            new_w = get_by_ratio(img_h,new_h,img_w)
            if new_w > d_w:
                new_w = d_w
                new_h = get_by_ratio(img_w,new_w,img_h)
                if new_h > d_h:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_u = np.round((d_h - new_h)/2.0)
                    pad_d = d_h - new_h - pad_u
            else:
                pad_l = np.round((d_w - new_w)/2.0)
                pad_r = d_w - new_w - pad_l
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
        else:
            new_w = d_w
            new_h = get_by_ratio(img_w,new_w,img_h)
            #print("debug1",new_h,new_w)
            if new_h > d_h:
                new_h = d_h
                new_w = get_by_ratio(img_h,new_h,img_w)
                #print("debug2",new_h,new_w)
                if new_w > d_w:
                    print("could not get pad:org_img,dest_img",(img_h,img_w),crop_size)
                    return cv2.resize(img,(int(d_w),int(d_h)))
                else:
                    pad_l = np.round((d_w - new_w)/2.0)
                    pad_r = d_w - new_w - pad_l
            else:
                pad_u = np.round((d_h - new_h)/2.0)
                pad_d = d_h - new_h - pad_u
            #print("up",new_h,new_w)
            img_out = cv2.resize(img,(int(new_w),int(new_h)))
    elif img_w==d_w and img_h==d_h:
        img_out = img
    if not [pad_l,pad_r,pad_u,pad_d] == [0,0,0,0] :
        color = [255,255,255]
        #print("padding",[pad_l,pad_r,pad_u,pad_d])
        img_out = cv2.copyMakeBorder(img_out,top=int(pad_u),bottom=int(pad_d),left=int(pad_l),right=int(pad_r),\
                                    borderType=cv2.BORDER_CONSTANT,value=color) #BORDER_REPLICATE
    return img_out


class FaceRecognize(object):
    def __init__(self,protxt,caffemodel,out_layer_name):
        #caffe.set_device(0)
        #caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        self.face_p = protxt
        self.face_m = caffemodel
        self.out_layer = out_layer_name
        self.load_model()

    def load_model(self):
        self.face_net = caffe.Net(self.face_p,self.face_m,caffe.TEST)
        print("face model load successful **************************************")
        if config.model_resave:
            caffe_model_path = "../models/sphere/sph_test.caffemodel"
            self.face_net.save(caffe_model_path)
            print("************ model resaved over******")

    def extractfeature(self,img):
        img = np.asarray(img)
        batch_size,h,w,chal = img.shape
        assert  batch_size==config.batch_size,"img batch size is not equal to config"
        #print(img.shape)
        #print("img shape ",img.shape)
        _,net_chal,net_h,net_w = self.face_net.blobs['data'].data.shape
        #print("net shape ",net_h,net_w)
        if h !=net_h or w !=net_w:
            #caffe_img = cv2.resize(img,(net_w,net_h))
            #caffe_img = Img_Pad(img,(net_h,net_w))
            print("input imgs is not equal to net input")
            return []
        else:
            caffe_img = img
        #mt_extract = TimeR('net.extract')
        caffe_img = (caffe_img -127.5)/128.0
        if config.feature_expand:
            self.face_net.blobs['data'].reshape(2,net_chal,net_h,net_w)
        else:
            self.face_net.blobs['data'].reshape(batch_size,net_chal,net_h,net_w)
        t = time.time()
        #caffe_img = np.swapaxes(caffe_img,0,2)
        #np.savetxt("img.txt",(caffe_img[:,:,0]))
        #print("model",caffe_img[0,1,:])
        #caffe_img = np.expand_dims(caffe_img,0)
        #mt_tra = TimeR('net.extract.bfor')
        caffe_img = np.transpose(caffe_img, (0,3,1,2))
        caffe_img = np.asarray(caffe_img,dtype=np.float32)
        if config.feature_expand:
            img_flip = np.flip(caffe_img,axis=3,dtype=np.float32)
            self.face_net.blobs['data'].data[0] = caffe_img
            self.face_net.blobs['data'].data[1] = img_flip
        else:
            self.face_net.blobs['data'].data[...] = caffe_img
        mt_tra.finish()
        #mt_for = TimeR('net.extract.forward')
        net_out = self.face_net.forward()
        #mt_for.finish()
        features = net_out[self.out_layer]
        t1 = time.time()-t
        if config.time:
            print("caffe forward time cost: ",t1)
        if config.feature_expand:
            feature_org = features[0]
            feature_flip = features[1]
            feature_sum = feature_org + feature_flip
            _norm=np.linalg.norm(feature_sum)
            features = feature_sum / _norm
        mt_extract.finish()
        return np.reshape(features,(batch_size,config.feature_lenth))

    def calculateL2(self,feat1,feat2,c_type='euclidean'):
        assert np.shape(feat1)==np.shape(feat2)
        [len_,] = np.shape(feat1)
        #print("len ",len_)
        if c_type == "cosine":
            s_d = distance.cosine(feat1,feat2)
        elif c_type == "euclidean":
            #s_d = np.sqrt(np.sum(np.square(feat1-feat2)))
            #s_d = distance.euclidean(feat1,feat2,w=1./len_)
            s_d = distance.euclidean(feat1,feat2)
        elif c_type == "correlation":
            s_d = distance.correlation(feat1,feat2)
        elif c_type == "braycurtis":
            s_d = distance.braycurtis(feat1,feat2)
        elif c_type == 'canberra':
            s_d = distance.canberra(feat1,feat2)
        elif c_type == "chebyshev":
            s_d = distance.chebyshev(feat1,feat2)
        return s_d

class mx_FaceRecognize(object):
    def __init__(self,model_path,epoch_num,img_size,layer='fc1'):
        ctx = []
        if len(config.gpu_list) >0:
            for idx in config.gpu_list:
                ctx.append(mx.gpu(idx))
        else:
            ctx = [mx.cpu(0)]
        if config.mx_loadmodel:
            self.load_model2(ctx,img_size,model_path,epoch_num,layer)
        else:
            self.load_model(model_path,epoch_num,ctx)
        self.h, self.w = img_size

    def load_model(self,model_path,epoch_num,ctx):
        self.model_net = mx.model.FeedForward.load(model_path,epoch_num,ctx=ctx)

    def display_model(self,sym):
        data_shape = {"data":(1,3,112,112)}
        net_show = mx.viz.plot_network(symbol=sym,shape=data_shape)  
        net_show.render(filename="mxnet_rnet",cleanup=True)

    def load_model2(self,ctx, image_size, prefix,epoch_num,layer):
        print('loading',prefix, epoch_num)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch_num)
        all_layers = sym.get_internals()
        if config.debug:
            print("all layers ",all_layers.list_outputs())
        sym = all_layers[layer+'_output']
        if config.display_model:
            self.display_model(sym)
        self.model_net = mx.mod.Module(symbol=sym, context=ctx, data_names=('data',),label_names = None)
        #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        self.model_net.bind(data_shapes=[('data', (config.batch_size, 3, image_size[0], image_size[1]))],for_training=False)
        self.model_net.set_params(arg_params, aux_params)
        if config.model_resave:
            dellist = []
            for k,v in arg_params.iteritems():
                if k.startswith('fc7'):
                    dellist.append(k)
                elif k.startswith('Drop'):
                    dellist.append(k)
                if config.debug:
                    print("key name: ",k)
            for d in dellist:
                del arg_params[d]
            mx.model.save_checkpoint(prefix+"resave", 0, sym, arg_params, aux_params)
    
    def extractfeature(self,img):
        img = np.asarray(img)
        batch_size,h_,w_,chal = img.shape
        #assert  batch_size==config.batch_size,"img batch size is not equal to config"
        if h_ !=self.h or w_ !=self.w:
            print(">>>>********error: input imgs is not equal to net input ******\n")
            return []
        #mt_extract = TimeR('net.extract')
        t = time.time()
        #mt_tra = TimeR('net.extract.bfor')
        img = np.transpose(img,(0,3,1,2))
        if config.feature_expand:
            img_list = []
            img_flip = np.flip(img,axis=3)
            img_list.append(img)
            img_list.append(img_flip)
            img_input = img_list
        else:
            img_input = img #np.expand_dims(img,0)
        
        if config.mx_loadmodel:
            data = mx.nd.array(img_input)
            db = mx.io.DataBatch(data=(data,))
            #mt_tra.finish()
            #mt_for = TimeR('net.extract.forward')
            self.model_net.forward(db, is_train=False)
            #mt_for.finish()
            embedding = self.model_net.get_outputs()[0].asnumpy()
            if config.feature_expand:
                features = embedding[0] + embedding[1]
                #embedding = skpro.normalize(embedding)
                _norm=np.linalg.norm(features)
                features /= _norm
            else:
                features = embedding
        else: 
            #mt_for = TimeR('net.extract.predict')
            features = self.model_net.predict(img_input)
            #mt_for.finish()
            #embedding = features[0]
        t1 = time.time() - t
        if config.time:
            print("mxnet forward time cost: ",t1)
        if config.debug:
            print("feature shape ",np.shape(features))
        #mt_extract.finish()
        return features

if __name__ == '__main__':
    imgpath = "/home/lxy/Develop/Center_Loss/mtcnn-caffe/image/pics/test.jpg"
    img = cv2.imread(imgpath)
    print("org",img.shape)
    size_ = [112,112]
    img_o = Img_Pad(img,size_)
    print("out",img_o.shape)
    cv2.imshow("img",img_o)
    cv2.imshow("org",img)
    cv2.waitKey(0)