import numpy as np

class SinPolicy:
    def __init__(self, para):
        self.theta = para.theta #0.3
        self.AD =  para.AD #max 0.03
        self.Phi =  para.Phi #1
        self.St =  para.St #0.3

        self.dAoA = self.theta * np.pi/180
        self.uAoA = self.dAoA
        self.dA = self.AD*32
        self.uA = self.dA
        # self.phi = self.Phi * np.pi/180
        self.phi = self.Phi

        self.offset0 = 0.0
        self.offset1 = 0.0

        self.omega = 2*np.pi * self.St / (1.0*0.16)
        self.period = 2*np.pi/self.omega

    def set_para(self, theta, AD, phi, st):
        self.theta = theta  # 0.3
        self.AD = AD  # max 0.03
        self.Phi = phi  # 1
        self.St = st  # 0.3

        self.dAoA = self.theta * np.pi/180
        self.uAoA = self.dAoA
        self.dA = self.AD*32
        self.uA = self.dA
        # self.phi = self.Phi * np.pi/180
        self.phi = self.Phi

        self.omega = 2*np.pi * self.St / (1.0*0.16)
        self.period = 2*np.pi/self.omega

    def set_global_offset(self, offset0, offset1):
        self.offset0 = offset0
        self.offset1 = offset1


    def choose_action(self, obs, dt, info=None):
        if obs==0:
            return np.array([0, 0])
        
            
        self.dt = dt
        self.y = info["CurrentY"]
        self.currentphi = info["CurrentPhi"]
        pitch = self.cal_pitch(obs)
        heave = self.cal_heave(obs)
        return np.array([heave, pitch])

    def cal_pitch(self, t):
        pitchAmp = self.dAoA

        #return (pitchAmp/0.16*np.sin(self.omega*(t))-
                #pitchAmp/0.16*np.sin(self.omega*t-self.dt))
        return (-pitchAmp*np.sin(self.omega*t+self.phi+self.offset0)-self.currentphi+np.pi)/self.dt
        
    
    def cal_heave(self, t):
        heaveAmp = self.dA
        
        # return heaveAmp * self.omega * np.cos(self.omega*t)
        
       # if t<= 0.010:
        #    return heaveAmp/0.16*np.sin(-self.phi)
        #else:
        #return (heaveAmp/0.16*np.sin(self.omega*(t)-self.phi)-
                    #heaveAmp/0.16*np.sin(self.omega*(t-self.dt)-self.phi))
        return (heaveAmp*np.sin(self.omega*t+self.offset1)+192 - self.y)/self.dt
        #return 0
