import numpy as np

kh = 3
kw = 3

# NCS
def ConvRef(input, weight, g):
    n = input.shape[0]

    ic = input.shape[1]
    ih = input.shape[2]
    iw = input.shape[3]

    oc = weight.shape[0]

    assert(ic == weight.shape[1])

    oh = ih - kh + 1
    ow = iw - kw + 1

    output = np.empty((n, oc, oh, ow), dtype = float, order = 'C')

    assert(ic % g == 0 and oc % g == 0)

    icpg = int(ic / g)
    ocpg = int(oc / g)
    
    for n0 in range(0, n):
        for g0 in range(0, g):
            for oh0 in range(0, oh):
                for ow0 in range(0, ow):
                    for ocpg0 in range(0, ocpg):
                        
                        value = 0
                        for kh0 in range(0, kh):
                            for kw0 in range(0, kw):
                                for icpg0 in range(0, icpg):
                                    ih0 = oh0 + kh0 
                                    iw0 = ow0 + kw0
                                    oc0 = g0 * ocpg + ocpg0
                                    ic0 = g0 * icpg + icpg0
                                    i = input[n0][ic0][ih0][iw0]
                                    w = weight[oc0][ic0][kh0][kw0]
                                    value += i * w

                        output[n0][oc0][oh0][ow0] = value
    return output

G = np.array([
    [2,  0, 0],
    [1,  1, 1],
    [1, -1, 1],
    [0,  0, 2]
])
G = G/2

BT = np.array([
    [1,  0, -1,  0],
    [0,  1,  1,  0],
    [0, -1,  1,  0],
    [0,  1,  0, -1]
])

AT = np.array([
    [1,  1,  1,  0],
    [0,  1, -1, -1]
])

ohb = 2
owb = 2


class Winograd:
    def TransWeight(self, weight):
        return np.dot(np.dot(G, weight), G.T)

    def TransInput(self, input):
        transInput = np.dot(np.dot(BT, input), BT.T)

        return transInput

    def CalculateM(self, transInput, transWeight):
        M = np.multiply(transInput, transWeight)

        return M

    def TransOutput(self, M):
        output = np.dot(np.dot(AT, M), AT.T)

        return output

    def Execute(self, input, weight, g):
        self.n = input.shape[0]

        self.ic = input.shape[1]
        self.ih = input.shape[2]
        self.iw = input.shape[3]

        self.oc = weight.shape[0]

        self.oh = self.ih - kh + 1
        self.ow = self.iw - kw + 1

        self.tileCH = int(np.ceil(self.oh / ohb))
        self.tileCW = int(np.ceil(self.ow / owb))

        self.g = g

        self.ocpg = int(self.oc / self.g)

        output = np.zeros((self.n, self.oc, self.oh, self.ow))

        for n0 in range(0, self.n):
            for g0 in range(0, self.g):
                for ocpg0 in range(0, self.ocpg):
                    oc0 = g0 * self.ocpg + ocpg0

                    for ic0 in range(0, self.ic):
                        transWeight = self.TransWeight(weight[oc0][ic0])
                        transInput = self.TransInput(input[n0][ic0])
                        transOutput = self.CalculateM(transInput, transWeight)
                        output[n0][oc0] += self.TransOutput(transOutput)

        return output

def TestConv2D(n, ih, iw, ic, oc, g):
    input = np.arange(n * ic * ih * iw, dtype = float).reshape((n, ic, ih, iw))
    weight = np.arange(start=1, stop=oc * ic * kh * kw + 1, dtype = float).reshape((oc, ic, kh, kw))

    outputRef = ConvRef(input, weight, g)
    outputAct = Winograd().Execute(input, weight, g)

    if max(outputRef.flatten() - outputAct.flatten()) > 0.01:
        assert(False)

if __name__=="__main__":
    TestConv2D(n=1, ih=4, iw=4, ic=1, oc= 1, g=1)
    TestConv2D(n=3, ih=4, iw=4, ic=14, oc= 26, g=2)

    print("Pass!")
