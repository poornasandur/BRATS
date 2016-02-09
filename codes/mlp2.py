import numpy as np
import numpy
import cPickle
import gzip
import os
import sys
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams

from logistic_sgd import LogisticRegression
#from load_data import load_umontreal_data, load_mnist
from load_data import *
from updates import *

from sklearn.metrics import confusion_matrix

try:
    import PIL.Image as Image
except ImportError:
    import Image

from utils import tile_raster_images


##################################
## Various activation functions ##
##################################
#### rectified linear unit
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
            
        else:
            try:
                W = theano.shared(value=W, name='W')
            except:
                pass
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
            
        else:
            try:
                b = theano.shared(value=b, name='b')
            except:
                pass

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, activation=activation, W=W, b=b, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            activations,
            W = None,
            b = None,
            use_bias=True):

        #rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        
        W_temp = []
        b_temp = []
        for i in xrange(len(layer_sizes) - 1):
            if W is None:
                W_temp.append(None)
            else:
                W_temp.append(W[i])
                
            if b is None:
                b_temp.append(None)
            else:
                b_temp.append(b[i])

        self.params = []
        self.L2_sqr = 0
        self.L1_sqr=0
            
        #first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        
        layer_counter = 0        

        for n_in, n_out in weight_matrix_sizes[:-1]:

            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, W = W_temp[layer_counter], b = b_temp[layer_counter],
                    use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter + 1])


            # self.params.append(next_dropout_layer.params)

            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            self.L2_sqr += (next_dropout_layer.W ** 2).sum()
            self .L1_sqr += abs(next_dropout_layer.W).sum()
            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                W = W_temp[layer_counter],
                b = b_temp[layer_counter],
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # self.params.extend(output_layer.params)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.L1_sqr +=abs(dropout_output_layer.W).sum()   ### added today!!!!!!
        self.L2_sqr += (dropout_output_layer.W ** 2).sum()

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        self.pred = self.layers[-1].y_pred

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]

    


def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        mom_params,
        activations,
        dropout,
        dropout_rates,
        layer_sizes,
        dataset,
        use_bias,
        W = None,
        b = None,
        random_seed=1234,
        prefix = ''):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    print len(layer_sizes)
    print len(dropout_rates)
    assert len(layer_sizes) - 1 == len(dropout_rates)
    
    # extract the params for momentum
    # mom_start = mom_params["start"]
    # mom_end = mom_params["end"]
    # mom_epoch_interval = mom_params["interval"]
    
    # train_patch = '/media/brain/1A34723D34721BC7/BRATS/varghese/Recon_2013_data/BRATS_training_patches/u_trainpatch_2D_11x11_costpenalty_.npy'
    # train_label = '/media/brain/1A34723D34721BC7/BRATS/varghese/Recon_2013_data/BRATS_training_patches/u_trainlabel_2D_11x11_costpenalty_.npy'
    # valid_patch = '/media/brain/1A34723D34721BC7/BRATS/varghese/Recon_2013_data/BRATS_validation_patches/u_validpatch_2D_11x11_costpenalty_.npy'
    # valid_label = '/media/brain/1A34723D34721BC7/BRATS/varghese/Recon_2013_data/BRATS_validation_patches/u_validlabel_2D_11x11_costpenalty_.npy'

    train_patch, train_label, valid_patch, valid_label = dataset
    
    datasets = load_data(train_patch,train_label,valid_patch,valid_label)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    learning_rate = T.scalar('lr')
    # learning_rate = theano.shared(np.asarray(initial_learning_rate,
    #     dtype=theano.config.floatX))

    p1 = T.sum(T.eq(train_set_y, 1)).eval() / float(train_set_y.shape[0].eval())
    p2 = T.sum(T.eq(train_set_y, 2)).eval() / float(train_set_y.shape[0].eval())
    p3 = T.sum(T.eq(train_set_y, 3)).eval() / float(train_set_y.shape[0].eval())
    p4 = T.sum(T.eq(train_set_y, 4)).eval() / float(train_set_y.shape[0].eval())

    # print 'Probability 1: ',p1
    # print 'Probability 2: ',p2
    # print 'Probability 3: ',p3
    # print 'Probability 4: ',p4

    rng = np.random.RandomState(random_seed)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=activations,
                     W = W,
                     b = b,
                     use_bias=use_bias)

    print '#############################'
    print classifier.params
    print '#############################'

    # Build the expresson for the cost function.
    cost = classifier.negative_log_likelihood(y) + 0.0001 * classifier.L2_sqr + 0.0001*classifier.L1_sqr   # added today
    dropout_cost = classifier.dropout_negative_log_likelihood(y) + 0.0001 * classifier.L2_sqr + 0.0001* classifier.L1_sqr   # added today

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    def valid_score():
        return [validate_model(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
    def test_score():
        return [test_model(i) for i in xrange(n_test_batches)]

    def get_prediction(train_set_x, batch_size):
        prediction = theano.function(inputs = [index], outputs = classifier.pred,
                  givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})
        return prediction




    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # Compute gradients of the model wrt parameters
    # gparams = []
    # for param in classifier.params:
    #     # Use the right cost function here to train with or without dropout.
    #     gparam = T.grad(dropout_cost if dropout else cost, param)
    #     gparams.append(gparam)

    # # ... and allocate mmeory for momentum'd versions of the gradient
    # gparams_mom = []
    # for param in classifier.params:
    #     gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
    #         dtype=theano.config.floatX))
    #     gparams_mom.append(gparam_mom)

    # # Compute momentum for the current epoch
    # mom = ifelse(epoch < mom_epoch_interval,
    #         mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
    #         mom_end)

    # # Update the step direction using momentum
    # updates = OrderedDict()
    # for gparam_mom, gparam in zip(gparams_mom, gparams):
    #     # Misha Denil's original version
    #     #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
      
    #     # change the update rule to match Hinton's dropout paper
    #     updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # # ... and take a step along that direction
    # for param, gparam_mom in zip(classifier.params, gparams_mom):
    #     # Misha Denil's original version
    #     #stepped_param = param - learning_rate * updates[gparam_mom]
        
    #     # since we have included learning_rate in gparam_mom, we don't need it
    #     # here
    #     stepped_param = param + updates[gparam_mom]

    #     # This is a silly hack to constrain the norms of the rows of the weight
    #     # matrices.  This just checks if there are two dimensions to the
    #     # parameter and constrains it if so... maybe this is a bit silly but it
    #     # should work for now.
    #     if param.get_value(borrow=True).ndim == 2:
    #         #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
    #         #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
    #         #updates[param] = stepped_param * scale
            
    #         # constrain the norms of the COLUMNs of the weight, according to
    #         # https://github.com/BVLC/caffe/issues/109
    #         col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
    #         desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
    #         scale = desired_norms / (1e-7 + col_norms)
    #         updates[param] = stepped_param * scale
    #     else:
    #         updates[param] = stepped_param

    updates = sgd(dropout_cost if dropout else cost, classifier.params, learning_rate = learning_rate)

    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[epoch, index, theano.Param(learning_rate, default=0.1)], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]},
                on_unused_input = 'ignore')
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #         updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    ########################confusion matrix Block 1##########################    
    prediction = get_prediction(train_set_x,batch_size)
    y_truth = train_set_y.eval()
    y_truth = y_truth[0:(len(y_truth)-(len(y_truth)%batch_size))]
    cnf_freq = 1
    cnf_freq_v=5
    #################################
    prediction_v = get_prediction(valid_set_x,batch_size)
    y_truth_v = valid_set_y.eval()
    y_truth_v = y_truth_v[0:(len(y_truth_v)-(len(y_truth_v)%batch_size))]
    #######Added to see the confusion matrix of the validation data#############################



    patience = 40 * n_train_batches  # look as this many examples regardless
    patience_increase = 10.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch


    



    best_params = None
    best_validation_errors = np.inf
    best_validation_loss = np.inf
    best_iter = 0
    test_scores = 0.
    epoch_counter = 0
    start_time = time.clock()

    adapt_counter = 0
    log_valid_cost = []

    shapeimg = [(42,42),(50,50), (25,40), (50,10)]

    # results_file = open(results_file_name, 'wb')

    adaptive_lr = initial_learning_rate


    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1

        ################################confusion matrix block 2#################
        if epoch_counter%cnf_freq==0:
            pred_c = numpy.array([])
            for minibatch_index in xrange(n_train_batches):
                pred_c = numpy.concatenate([pred_c,numpy.array(prediction(minibatch_index))])
        
            cnf_matrix = confusion_matrix(y_truth, pred_c)
            print 'Training confusion matrix'
            print 
            print cnf_matrix
            print 
            ##########################################################################

        if epoch_counter%cnf_freq_v==0:
            pred_v = numpy.array([])
            for minibatch_index_v in xrange(n_valid_batches):
                pred_v = numpy.concatenate([pred_v,numpy.array(prediction(minibatch_index_v))])
        
            cnf_matrix_v = confusion_matrix(y_truth_v, pred_v)
            print 'validation confusion_matrix'
            print 
            print cnf_matrix_v  
            print           

        c = []
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(epoch_counter, minibatch_index, adaptive_lr)
            c.append(minibatch_avg_cost)

            ###################################################################################

            iter = (epoch_counter - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = valid_score()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch_counter, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                log_valid_cost.append(this_validation_loss)
##############################################Added on 13oct to see confusion matrix of validation data!##########################################################################

###########################################################################################################################################
                
                print 'Training cost: ', np.mean(c)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    
                    print 'Saving the best validation network'
                    genVariables = 'gen'
                    if dropout:
                        save_file = open(prefix + 'dropout_fine_tuning.pkl','wb')
                    else:
                        save_file = open(prefix + 'fine_tuning.pkl','wb')
                    cPickle.dump([1000,1000,1000], save_file)
                    cPickle.dump(genVariables, save_file)
                    for j in xrange(len(classifier.params)):
                        cPickle.dump(classifier.params[j].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)
                    save_file.close()
                    
                    
                
                    # test it on the test set
                    test_losses = test_score()
                    test_scores = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch_counter, minibatch_index + 1, n_train_batches,
                           test_scores * 100.))
                    

                else :
                    adapt_counter = adapt_counter+1
                # if adapt_counter>20:
                #     adapt_counter=0
                #     adaptive_lr=0.8*adaptive_lr
                #     print 'Reducing learning rate! ', adaptive_lr

        adaptive_lr = initial_learning_rate / ( 1 + 0.01 * epoch_counter)
        if epoch_counter %5 ==0:
            print 'current learning rate:-',adaptive_lr 

        # adaptive_lr=initial_learning_rate  # changed since we are using adadelta!!!

#        if epoch%1==0:
#            
#            prediction1 = prediction()
#            print prediction1[0]
                    
            #if patience <= iter:
            #    done_looping = True
            #    break

        if epoch_counter%10 == 0 and epoch_counter!=0 or epoch_counter == 399 or epoch_counter == 199:
            for i in xrange(len(classifier.params)/2 - 1):
                image = Image.fromarray(tile_raster_images(
                    X=classifier.params[2*i].get_value(borrow=True).T,
                    img_shape=shapeimg[i], tile_shape=(40,layer_sizes[i+1]/20),
                    tile_spacing=(1, 1)))
                image.save(prefix+str(i) + '_' + str(epoch_counter)+'.png')

        save_file = open(prefix + 'latest_fine_tuning2.pkl','wb')
        cPickle.dump([1000,1000,1000], save_file)
        cPickle.dump(genVariables, save_file)
        for j in xrange(len(classifier.params)):
            cPickle.dump(classifier.params[j].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)
        save_file.close()

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_scores * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # valid_file = open(prefix+'log_valid_error.txt', 'w')
    # valid_file.write('Best validation error: '+str(best_validation_loss*100))
    # valid_file.write('\nBest test error: '+str(test_score*100))
    # valid_file.close()
    # finetune_log_file = open(prefix + 'log_finetune_cost.txt', "a")
    # for l in log_valid_cost:
    #     finetune_log_file.write("%f\n"%l)
    # finetune_log_file.close()


    ############################################################################################
        # Compute loss on validation set
    #     	validation_losses = valid_score()
    #     	this_validation_errors = np.mean(validation_losses)

    #     # Report and save progress.
    #     print "epoch {}, Validation error {}, learning_rate={}{}".format(
    #             epoch_counter, this_validation_errors,
    #             learning_rate.get_value(borrow=True),
    #             " **" if this_validation_errors < best_validation_errors else "")
    #     print 'Training cost: ', np.mean(c)
                
    #     if this_validation_errors < best_validation_errors:
    #         print 'Saving best model..'
    #         genVariables = 1
    #         save_file = open('dropout_fine_tuning.pkl','wb')
    #         cPickle.dump([1000,1000,1000], save_file)
    #         cPickle.dump(genVariables, save_file)
    #         for k in xrange(len(classifier.params)):
    #             cPickle.dump(classifier.params[k].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)
    #         save_file.close()

    #     best_validation_errors = min(best_validation_errors,
    #             this_validation_errors)
    #     results_file.write("{0}\n".format(this_validation_errors))
    #     results_file.flush()

    #     new_learning_rate = decay_learning_rate()

    # end_time = time.clock()
    # print(('Optimization complete. Best validation score of %f %% '
    #        'obtained at iteration %i, with test performance %f %%') %
    #       (best_validation_errors * 100., best_iter, test_score * 100.))
    # print >> sys.stderr, ('The code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((end_time - start_time) / 60.))


def runMLP2(initial_learning_rate = 0.1,
    n_epochs = 100,
    batch_size = 100,
    layer_sizes = [ 22*22, 1000, 1000, 1000, 5 ],
    dropout_rates = [ 0.0, 0.0, 0.5, 0.5 ],
    pretrain_pkl = '/media/brain/1A34723D34721BC7/BRATS/codes/results/test231_11x11_costpenalty/11x11_costpenaltypre_training.pkl',
    train_patch = '/media/brain/1A34723D34721BC7/BRATS/varghese/Recon_2013_data/BRATS_training_patches/u_trainpatch_2D_11x11_costpenalty_.npy',
    train_label = '/media/brain/1A34723D34721BC7/BRATS/varghese/Recon_2013_data/BRATS_training_patches/u_trainlabel_2D_11x11_costpenalty_.npy',
    valid_patch = '/media/brain/1A34723D34721BC7/BRATS/varghese/Recon_2013_data/BRATS_validation_patches/u_validpatch_2D_11x11_costpenalty_.npy',
    valid_label = '/media/brain/1A34723D34721BC7/BRATS/varghese/Recon_2013_data/BRATS_validation_patches/u_validlabel_2D_11x11_costpenalty_.npy',
    prefix = '',
    activations = [ Sigmoid, Sigmoid, Sigmoid ],
    dropout = True,
    learning_rate_decay = 1,
    squared_filter_length_limit = 15.0):

    random_seed = 1234
    mom_start = 0.5
    mom_end = 0.99
    mom_epoch_interval = 500
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}

    dataset = train_patch, train_label, valid_patch, valid_label
    #dataset = 'data/mnist.pkl.gz'
    f = open(pretrain_pkl,'rb')
    g = cPickle.load(f)
    # g = cPickle.load(f)

    W = []
    b = []
    for i in xrange(len(layer_sizes) - 1):
        W.append(cPickle.load(f))
        b.append(cPickle.load(f))

    test_mlp(initial_learning_rate=initial_learning_rate,
             learning_rate_decay=learning_rate_decay,
             squared_filter_length_limit=squared_filter_length_limit,
             n_epochs=n_epochs,
             batch_size=batch_size,
             layer_sizes=layer_sizes,
             mom_params=mom_params,
             activations=activations,
             dropout=dropout,
             dropout_rates=dropout_rates,
             dataset=dataset,
             use_bias=True,
             W = W,
             b = b,
             random_seed=random_seed,
             prefix = prefix)

	# set the random seed to enable reproduciable results
	# It is used for initializing the weight matrices
	# and generating the dropout masks for each mini-batch

    # initial_learning_rate = 0.1
    # learning_rate_decay = 1
    # squared_filter_length_limit = 15.0
    # n_epochs = 100
    # batch_size = 100
    # layer_sizes = [ 22*22, 1000, 1000, 1000, 5 ]
    
    # # dropout rate for each layer
    # dropout_rates = [ 0.2, 0.5, 0.5, 0.5 ]
    # # activation functions for each layer
    # # For this demo, we don't need to set the activation functions for the 
    # # on top layer, since it is always 10-way Softmax
    # activations = [ Sigmoid, Sigmoid, Sigmoid ]
    
    #### the params for momentum

    
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end

if __name__ == '__main__':

	# import sys

	# if len(sys.argv) < 2:
	# 	print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
	# 	exit(1)

	# elif sys.argv[1] == "dropout":
	# 	dropout = True
	# 	results_file_name = "results_dropout.txt"

	# elif sys.argv[1] == "backprop":
	# 	dropout = False
	# 	results_file_name = "results_backprop.txt"

	# else:
	# 	print "I don't know how to '{0}'".format(sys.argv[1])
	# 	exit(1)

	runMLP2()



    
    

