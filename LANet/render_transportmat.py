from scipy.io import loadmat
import h5py  # tables  # or
import numpy as np
import os
import torch
import pickle

class RenderWithTransportMat():

    def __init__(self, transportMatFname, lightHeight, doHalfMat=True):
        self.transportMatFname = transportMatFname
        self.mat_ImgHeight = lightHeight
        # self.renderHeight = renderHeight

        self.doHalfMat = doHalfMat

        self.matT, self.maskT = self.load()
        # self.renderVectors = None

    def load(self):
        typeMat = 'half' if self.doHalfMat else 'full'
        # print('[Render]loading: %s' % (os.path.split(self.transportMatFname)[1]))
        assert os.path.exists(self.transportMatFname), 'FileNotFound %s' % self.transportMatFname
        ext = os.path.split(self.transportMatFname)[1].split('.')[-1]
        if ext == 'mat':
            try:
                mat = loadmat(self.transportMatFname)
                matT = mat['T'].astype('float32')
                maskT = mat['mask'].astype('bool')
            except NotImplementedError:
                mat = h5py.File(self.transportMatFname, 'r')
                matT = mat.get('T')[:].astype('float32').transpose()
                maskT = mat.get('mask')[:].astype('bool').transpose()
                mat.close()
        elif ext == 'pkl':
            with open(self.transportMatFname, 'rb') as infile:
                mat = pickle.load(infile)
                matT = mat['T'].astype('float32')
                maskT = mat['mask'].astype('bool')
        elif ext == 'msgpack':
            import lz4.frame, msgpack, msgpack_numpy
            from functools import partial
            lz4open = partial(lz4.frame.open, block_size=lz4.frame.BLOCKSIZE_MAX1MB,
                            compression_level=lz4.frame.COMPRESSIONLEVEL_MIN)
            with lz4open(self.transportMatFname, "rb") as infile:
                raw = infile.read()
                mat = msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, max_str_len=2**32-1)
                matT = mat[b'T'].astype('float32')
                maskT = mat[b'mask'].astype('bool')
        # matT.shape == [r*r, eH*eW]
        return matT, maskT

    def rendering(self, ims, dst='vector'):
        if type(ims).__module__.find('numpy') >= 0:
            if self.doHalfMat:
                ims = ims.astype('float32')
                H = ims.shape[1]
                ims = ims[:, 0: H // 2, :, :]
                render_vector = self.renderingNP(ims)
                if dst == 'vector':
                    out = render_vector
                elif dst == 'image':
                    out = self.reshapeNP(render_vector)
            return out
        elif type(ims).__module__.find('torch') >= 0:
            if self.doHalfMat:
                H = self.mat_ImgHeight
                ims = ims[:, 0: H // 2, :, :]
                # print("Ims",ims.shape)
                render_vector = self.renderingTorch(ims)
                if dst == 'vector':
                    out = render_vector
                elif dst == 'image':
                    # print("Render", render_vector.shape)
                    out = self.reshapeTorch(render_vector)
                    out = out.permute(0,3,1,2)
                    # print("Out ", out.shape)
            return out

    def renderingTorch(self, ims):
        matT = torch.Tensor(self.matT).to(ims.device)
        nPixels = matT.shape[0]
        N, H, W, C = ims.shape
        assert H * W == matT.shape[1], '[ERROR] transportMat %d doesn\'t match image size %d' % (H * W, matT.shape[1])

        # operation from matlab, reshape in column order
        ims = ims.permute([0, 2, 1, 3])
        lights = ims.contiguous().view((-1, H * W, C))  # [N, #pixel, C]
        # transpose and reshape
        lightsT = lights.permute([1, 0, 2])  # [W*H, N, C]
        lightsTR = lightsT.contiguous().view([W * H, -1])  # [W*H, N*C]
        vectors = torch.matmul(matT, lightsTR)  # [#pixel, W*H] x [W*H, N*C)] -> [#pixel, N*C]
        vectors = vectors.view([nPixels, -1, C])  # [#pixel, N, C]
        vectors = vectors.permute([1, 0, 2])  # [N, #pixel, C]
        renderVectors = vectors
        return renderVectors

    def _reshapeTorch(self, tl_WHxC):
        # todo batch reshape
        C = tl_WHxC.shape[1]
        img = torch.zeros((self.maskT.shape[0], self.maskT.shape[1], C)).to(tl_WHxC.device)
        # img[torch.Tensor(self.maskT.astype(int)).nonzero()] = tl_WHxC. # non differentiable
        img[torch.Tensor(self.maskT.astype(int)).to(tl_WHxC.device) == 1] = tl_WHxC
        
        img = img.permute([1, 0, 2])
        return img

    def reshapeTorch(self, renderVectors):
        N = renderVectors.shape[0]
        ims = [self._reshapeTorch(renderVectors[i]) for i in range(N)]
        return torch.stack(ims)

    def renderingNP(self, ims):
        matT = self.matT
        nPixels = matT.shape[0]
        N, H, W, C = ims.shape
        assert H * W == matT.shape[1], '[ERROR] transportMat %d doesn\'t match image size %d' % (H * W, matT.shape[1])

        # operation from matlab, reshape in column order
        ims = np.transpose(ims, axes=[0, 2, 1, 3])
        lights = np.reshape(ims, (-1, H * W, C))  # [N, #pixel, C]
        # transpose and reshape
        lightsT = np.transpose(lights, axes=[1, 0, 2])  # [W*H, N, C]
        lightsTR = np.reshape(lightsT, [W * H, -1])  # [W*H, N*C]
        vectors = np.matmul(matT, lightsTR)  # [#pixel, W*H] x [W*H, N*C)] -> [#pixel, N*C]
        vectors = np.reshape(vectors, [nPixels, -1, C])  # [#pixel, N, C]
        vectors = np.transpose(vectors, axes=[1, 0, 2])  # [N, #pixel, C]
        renderVectors = vectors
        return renderVectors

    def _reshapeNP(self, tl_WHxC):
        # todo batch reshape
        C = tl_WHxC.shape[1]
        img = np.zeros((self.maskT.shape[0], self.maskT.shape[1], C))
        img[self.maskT] = tl_WHxC
        img = np.ndarray.transpose(img, [1, 0, 2])
        return img

    def reshapeNP(self, renderVectors):
        N = renderVectors.shape[0]
        ims = [self._reshapeNP(renderVectors[i]) for i in range(N)]
        return np.asarray(ims)

    def render_top_down(self, img,chw=True):
        if chw:
            img = img.permute(0,2,3,1)
        if type(img).__module__.find('numpy') >= 0:
            reverse_img = np.flip(img, [1])

        elif type(img).__module__.find('torch') >= 0:
            reverse_img = torch.flip(img, [1])
        else :
            print('[ERROR] Input img is neither torch nor numpy')

        return {"top" : self.rendering(img, dst='image'), "bottom": self.rendering(reverse_img, dst='image')}

def example():
    import time
    im_h = 256
    img_dim = 64
    rtm = RenderWithTransportMat(transportMatFname='transportMat.BumpySphereMiddle.top.e64.r64.half.mat', lightHeight=img_dim, doHalfMat=True)
    ims = np.ones([256, img_dim, img_dim*2, 3])
    print(ims.shape)
    # ims shape = [b,h,w,c]
    # pred shape = [b,c,h,w]

    # pred = [256,512]
    # pred = [64,128] = > [32,128]*2
    # what it wants = [128,64]=> [64,64]*2
    p = rtm.render_top_down(torch.from_numpy(ims).type('torch.FloatTensor'))
    print(p['top'].shape)
    print(p['bottom'].shape)

if __name__ == '__main__':
    import sys
    print(sys.version)
    example()