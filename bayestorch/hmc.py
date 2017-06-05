import torch as t
import torch.nn.functional as f
from torch.autograd import Variable
from torch.optim import Optimizer
import math
import matplotlib.pyplot as plt

class HMCSampler(object):
    """updates the model parameters by preforming n hcmc updates

    (larger description)
    """
    def __init__(self,parameters,epsilon=0.01,n=100):
        self.epsilon = epsilon
        self.n = n
        self.p_list = parameters

    def resample_r(self):
        p_list = self.p_list
        self.r_list = []
        for param in p_list:
            s = param.size()
            means = t.zeros(s)
            self.r_list.append(t.normal(means,1))#todo verify that this is correct

    def zero_grad(self):
        for param in self.p_list:
            if param.grad is not None:
                param.grad.data.zero_()

    def data_pass(self,closure):
        p_list = self.p_list
        self.zero_grad()
        loss = closure()
        loss.backward()
        g_list = list(map((lambda x: x.grad.data),p_list))
        return g_list,loss

    def step(self,closure):
        #TODO: add MC rejection
        self.resample_r()
        p_list = [x.data for x in self.p_list]
        r_list = self.r_list
        def assign(x,y):
            x.data = y
        epsilon = self.epsilon
        for i in range(self.n):
            #TODO: Clean up implementation with getter and setters
            g_list,_ = self.data_pass(closure)
            r_list = list(map(lambda x,y: x-y*epsilon/2,r_list,g_list))

            p_list = list(map(lambda x,y: x+y*epsilon,p_list,r_list))
            list(map(assign,self.p_list,p_list))

            g_list,loss = self.data_pass(closure)
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
                scale=std_dev/100)),requires_grad = True)
        else:
            self.parameter = Variable(t.FloatTensor(np.random.normal(size=shape,
                scale=std_dev)),requires_grad = True)
        self.var = std_dev*std_dev
        self.shape = shape
    
    def parameter_list(self):
        return [self.parameter]
    
    def val(self):
        return self.parameter
    
    def cuda(self):
        self.parameter = self.parameter.cuda()

    def cpu(self):
        self.parameter = self.parameter.cpu()

    def get_prior_llh(self,dims):
        prob = -t.log(self.parameter)**2 \
                /(2*self.var)-t.log(2*math.pi)-t.log(self.var)/2
        for dim in dims:
            prob = sum(prob,dim)

        return t.squeeze(prob)


def test_hmc():
    p = Variable(t.FloatTensor([0]),requires_grad = True)
    sampler = HMCSampler([p])
    p_values = []
    closure = lambda: p**2/2
    for i in range(20000):
        sampler.step(closure) 
        p_values.append(scalar(p))
    plt.hist(p_values)
    plt.show()



def scalar(x):
    return x.data.cpu().numpy()[0]
    

if __name__ == "__main__":
    test_hmc()
