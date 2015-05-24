"""

made based on cifar10 file

"""
import os
import logging

import numpy
from theano.compat.six.moves import xrange

from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import contains_nan
from pylearn2.utils import serial
from pylearn2.utils import string_utils


_logger = logging.getLogger(__name__)


class Timeliner(dense_design_matrix.DenseDesignMatrix):

    """
    Parameters
    ----------
    which_set : str
        One of 'train', 'test'
    center : WRITEME
    rescale : WRITEME
    gcn : float, optional
        Multiplicative constant to use for global contrast normalization.
        No global contrast normalization is applied, if None
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    toronto_prepro : WRITEME
    preprocessor : WRITEME
    """

    def __init__(self, which_set, center=False, rescale=False, gcn=None,
                 start=None, stop=None, axes=('b', 0, 1, 'c'),
                 toronto_prepro = False, preprocessor = None):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)

        self.axes = axes

        # we define here:
        dtype = 'uint8'
        ntrain = 50000
        nvalid = 0  # artefact, we won't use it
        ntest = 10000

        # we also expose the following details:
        self.img_shape = (3, 32, 32)
        self.img_size = numpy.prod(self.img_shape)
        self.n_classes = 100
        # make sure that this is working (we can also copy it from meta file)
        self.label_names = range(1900,2000)

        import cPickle
        fo = open('datasets/data_batch')
        dict = cPickle.load(fo)
        fo.close()
        
        lenx = numpy.ceil((ntrain + nvalid) / 10000.) * 10000
        x = numpy.zeros((lenx, self.img_size), dtype=dtype)
        y = numpy.zeros((lenx, 1), dtype=dtype)

        # load train data
        #data = serial.load(datasets[fname])
        x[0:8305,:] = dict['data']
        #x[i * 10000:(i + 1) * 10000, :] = dict['data']
        #y[i * 10000:(i + 1) * 10000, 0] = dict['labels']

        #X = dict['data']
        y[0:8305,0] = dict['labels']
        
        # load test data
        #_logger.info('loading file %s' % datasets['test_batch'])
        #data = serial.load(datasets['test_batch'])

        # process this data
        #Xs = {'train': x[0:ntrain],
        #      'test': data['data'][0:ntest]}

        #Ys = {'train': y[0:ntrain],
        #      'test': data['labels'][0:ntest]}

        X = numpy.cast['float32'](x[0:8305])
        y = y[0:8305]
#        y = Ys[which_set]



        if isinstance(y, list):
            y = numpy.asarray(y).astype(dtype)

        self.center = center

        self.rescale = rescale

        self.toronto_prepro = toronto_prepro

        self.gcn = gcn
        if gcn is not None:
            gcn = float(gcn)
            X = global_contrast_normalize(X, scale=gcn)

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                                  axes)

        super(Timeliner, self).__init__(X=X, y=y, view_converter=view_converter,
                                      y_labels=self.n_classes)

        assert not contains_nan(self.X)

        if preprocessor:
            preprocessor.apply(self)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        # assumes no preprocessing. need to make preprocessors mark the
        # new ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i, :] /= numpy.abs(rval[i, :]).max()
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = numpy.clip(rval, -1., 1.)

        return rval

    def __setstate__(self, state):
        super(Timeliner, self).__setstate__(state)
        # Patch old pkls
        if self.y is not None and self.y.ndim == 1:
            self.y = self.y.reshape((self.y.shape[0], 1))
        if 'y_labels' not in state:
            self.y_labels = 10

    def adjust_to_be_viewed_with(self, X, orig, per_example=False):
        """
        .. todo::

            WRITEME
        """
        # if the scale is set based on the data, display X oring the
        # scale determined by orig
        # assumes no preprocessing. need to make preprocessors mark
        # the new ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i, :] /= numpy.abs(orig[i, :]).max()
            else:
                rval /= numpy.abs(orig).max()
            rval = numpy.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = numpy.clip(rval, -1., 1.)

        return rval

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return Timeliner(which_set='test', center=self.center,
                       rescale=self.rescale, gcn=self.gcn,
                       toronto_prepro=self.toronto_prepro,
                       axes=self.axes)
