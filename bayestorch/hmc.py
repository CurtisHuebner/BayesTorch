import torch as t
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torch.autograd import Variable

import math

class HCMCSampler:
    """updates the model parameters by preforming n hcmc updates

    (larger description)
    """
    def __init__(self,model,data,nllh):
        self.model = model
        self.data = data
        self.nllh = nllh

    def resample_r(self):
        p_list = self.model.parameter_list()
        self.r_list = []
        for param in p_list:
            s = param.size()
            means = t.zeros(s)
            self.r_list.append(t.normal(means,1).cuda())#todo verify that this is correct

    def data_pass(self):
        p_list = self.model.parameter_list()

        def zero_grad(x):
            if x.grad is not None:
                x.grad.data.zero_()
        list(map(zero_grad,p_list))
        output = self.model.inference(self.data)
        loss = self.nllh(output,self.data)
        loss.backward()
        g_list = list(map((lambda x: x.grad.data),p_list))
        return g_list,loss

    def step(self,epsilon,n=1):
        self.resample_r()
        p_list = [x.data for x in self.model.parameter_list()]
        r_list = self.r_list
        def assign(x,y):
            x.data = y
        for i in range(n):
            #TODO:Verify that this is not nessesary
#           if (np.random.randn() > 0.5):
#               epsilon = -epsilon
            #TODO: Clean up implementation with getter and setters
            g_list,_ = self.data_pass()
            r_list = list(map(lambda x,y: x-y*epsilon/2,r_list,g_list))

            p_list = list(map(lambda x,y: x+y*epsilon,p_list,r_list))
            list(map(assign,self.model.parameter_list(),p_list))

            g_list,loss = self.data_pass()
            r_list = list(map(lambda x,y: x-y*epsilon/2,r_list,g_list))


class ParameterGroup:

    def __init__(self,parameter_dict):
        self.parameters = parameter_dict

    def get_prior_llh(self):
        prior = 0
        for value in self.parameters.values():
            prior += value.get_prior_llh()

    def parameter_list(self):
        p_list = []
        for value in self.parameters.values():
            p_list += value.parameter_list()
        return p_list

    def cuda(self):
        for value in self.parameters.values():
            value.cuda()

    def cpu(self):
        for value in self.parameters.values():
            value.cpu()

    def __getitem__(self,key):
        return self.parameters[key]

class TensorParameter:
    def __init__(self,shape,std_dev,zeros = True):
        if zeros:
            self.parameter = Variable(t.FloatTensor(np.random.normal(size=shape,
                scale=std_dev/100)).cuda(),requires_grad = True)
        else:
            self.parameter = Variable(t.FloatTensor(np.random.normal(size=shape,
                scale=std_dev)).cuda(),requires_grad = True)
        self.var = std_dev*std_dev
        self.shape = shape
    
    def parameter_list(self):
        return [self.parameter]
    
    def val(self):
        return self.parameter
    
    def cuda(self):
        self.parameter.cuda()

    def cpu(self):
        self.parameter.cpu()

    def get_prior_llh(self,dims):
        prob = -t.log(self.parameter)**2 \
                /(2*self.var)-t.log(2*math.pi)-t.log(self.var)/2
        for dim in dims:
            prob = sum(prob,dim)

        return t.squeeze(prob)


def test_hcmc():
    pass

def scalar(x):
    return x.data.cpu().numpy()[0]
    

if __name__ == "__main__":
    main()
