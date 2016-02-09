from load_data import *
from SdA import *
import getopt
import cPickle
from utils import tile_raster_images
import numpy as np
try:
    import PIL.Image as Image
except ImportError:
    import Image
import os
from sklearn.metrics import confusion_matrix


def test_SdA(finetune_lr=0.1, pretraining_epochs=1,
             pretrain_lr=0.001, training_epochs=1, 
             b_patch_filename = 'b_Training_patches_norm.npy', b_groundtruth_filename = 'b_Training_labels_norm.npy',
             b_valid_filename = 'b_Validation_patches_norm.npy', b_validtruth_filename = 'b_Validation_labels_norm.npy',
             u_patch_filename = 'u_Training_patches_norm.npy', u_groundtruth_filename = 'u_Training_labels_norm.npy',
             u_valid_filename = 'u_Validation_patches_norm.npy', u_validtruth_filename = 'u_Validation_labels_norm.npy',
             batch_size=100, n_ins = 605, n_outs = 5, hidden_layers_sizes = [1000,1000,1000],prefix = '11_11_3_G4_', corruption_levels=[0.2,0.2,0.2], resumeTraining = False, StopAtPretraining = False):
                 
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations to run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
   
    print '###########################'
    print 'Pretraining epochs: ', pretraining_epochs
    print 'Finetuning epochs: ', training_epochs
    print '###########################'
    
    W = []
    b = []
    
    #########################################################
    #########################################################
    
    #@@@@@@@@ Needs to be worked on @@@@@@@@@@@@@@@@@
    # Snippet to resume training if the program crashes halfway through #
    opts, arg = getopt.getopt(sys.argv[1:],"rp:")
    for opt, arg in opts:
        if opt == '-r':
            resumeTraining = True                               # make this true to resume training from saved model    
        elif opt == '-p':
            prefix = arg
            
    flag = 0
    
    if(resumeTraining):
        
        flag = 1
        
        path = '/media/brain/1A34723D34721BC7/BRATS/codes/results/test_255_9x9x3/9x9x3pre_training.pkl'
                
        savedModel_preTraining = file(path,'rb')
        genVariables_preTraining = cPickle.load(savedModel_preTraining)
        layer_number, epochs_done_preTraining, mean_cost , pretrain_lr = genVariables_preTraining
        epoch_flag = 1
        print 'Inside resumeTraining!!!!!!!!!!!!!!!!!!'
        no_of_layers = len(hidden_layers_sizes) + 1
        
        for i in xrange(no_of_layers):
            W.append(cPickle.load(savedModel_preTraining))
            b.append(cPickle.load(savedModel_preTraining))    
   
              
    ##############################################################
    ##############################################################

    if flag == 0:
                
        datasets = load_data(b_patch_filename,b_groundtruth_filename,b_valid_filename,b_validtruth_filename)
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        
    
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
    
        # numpy random generator
        # start-snippet-3
        numpy_rng = numpy.random.RandomState(89677)
        print '... building the model'
        
    #    print 'W: ', W
    #    print 'b: ', b
        
        ################################################################
        ################CONSTRUCTION OF SdA CLASS#######################
        sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=n_ins,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=n_outs)
            
        print 'SdA constructed'
        ################################################################
        ################################################################
        
        ################################################################
        # end-snippet-3 start-snippet-4
        #########################
        # PRETRAINING THE MODEL #
        #########################
    
        flag = open(prefix+'flag.pkl','wb')
        cPickle.dump(1,flag, protocol = cPickle.HIGHEST_PROTOCOL)
        flag.close()
            
        print '... getting the pretraining functions'
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,batch_size=batch_size)
        print 'Length of pretraining function: ', len(pretraining_fns)

        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        log_pretrain_cost = []

        

        shapeimg = [(42,42),(50,50), (25,40), (50,10)]

        # shapeimg = [(33,44),(32,25), (20,20), (20,10)]
        #corruption_levels = [.001, .001, .001]
        for i in xrange(sda.n_layers):
            
            # if i < layer_number:
            #     i = layer_number
                #print i
                # go through pretraining epochs
            best_cost = numpy.inf
            adapt_counter = 0
            learning_rate = pretrain_lr
            learning_rate_0=pretrain_lr

            if i==0:
                num_of_epochs = pretraining_epochs
            else:
                num_of_epochs = pretraining_epochs
            for epoch in xrange(num_of_epochs):


                ##########################################            
                # if epoch_flag is 1 and epoch < epochs_done_preTraining:
                #     epoch = epochs_done_preTraining
                #     epoch_flag = 0
                    ##########################################
                    # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    #sprint batch_index
                    c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=learning_rate))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
                current_cost = numpy.mean(c)
                log_pretrain_cost.append(numpy.mean(c))
                if current_cost < best_cost:
                    best_cost = current_cost
                if current_cost > best_cost :
                    adapt_counter = adapt_counter+1
                # if adapt_counter>25:
                itr = epoch + 1
                learning_rate = learning_rate_0/ ( 1 + itr * 0.005)       # HAVE TO Change this number!!!!!!!! to anneal faster.......
                print 'Reducing learning rate', learning_rate
                # learning_rate=learning_rate_0
                adapt_counter = 0
                if itr%5 ==0:
                    print 'current learning_rate:',learning_rate

            
                previous_cost = current_cost
                

                if epoch%50 == 0 and epoch!=0 or epoch == 399 or epoch == 199:
                    image = Image.fromarray(tile_raster_images(
                        X=sda.params[2*i].get_value(borrow=True).T,
                        img_shape=shapeimg[i], tile_shape=(40,hidden_layers_sizes[i]/20),
                        tile_spacing=(1, 1)))
                    image.save(prefix+str(i) + '_' + str(epoch)+'.png')
        

            save_valid = open(prefix+'pre_training.pkl', 'wb')
            genVariables = ['gen']
            cPickle.dump(genVariables,save_valid,protocol = cPickle.HIGHEST_PROTOCOL)
            for j in xrange(len(sda.params)):
                cPickle.dump(sda.params[j].get_value(borrow=True), save_valid, protocol = cPickle.HIGHEST_PROTOCOL)
            save_valid.close()


        pretrain_log_file = open(prefix + 'log_pretrain_cost.txt', "a")
        for l in log_pretrain_cost:
            pretrain_log_file.write("%f\n"%l)
        pretrain_log_file.close()
        
        # for k in [0,2,4,6]:
        #     print k
        #     image = Image.fromarray(tile_raster_images(
        #        X=sda.params[k].get_value(borrow=True).T,
        #        img_shape=shapeimg[k/2], tile_shape=(40,hidden_layers_sizes[k/2]/20),
        #        tile_spacing=(1, 1)))
        #     image.save(prefix+str(k/2)+'.png')


        #print sda.params[0]
        end_time = time.clock()

        print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
                     
    print '###################'
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model   


    if flag == 1:
    
        datasets = load_data(u_patch_filename,u_groundtruth_filename,u_valid_filename,u_validtruth_filename)
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        
        n_train_batches /= batch_size
        
        numpy_rng = numpy.random.RandomState(89677)
        print '... building the model'
        
    #    print 'W: ', W
    #    print 'b: ', b
        
        ################################################################
        ################CONSTRUCTION OF SdA CLASS#######################
        sda = SdA(
            numpy_rng=numpy_rng,
            n_ins=n_ins,
            hidden_layers_sizes=hidden_layers_sizes,
            n_outs=n_outs, W = W, b = b)
        
        print 'SdA constructed'
        
    if StopAtPretraining == False:  
        
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = sda.build_finetune_functions(datasets=datasets,batch_size=batch_size)
        print batch_size

        print '... finetunning the model'
        ########################confusion matrix Block 1##########################    
        prediction = sda.get_prediction(train_set_x,batch_size)
        y_truth = np.load(u_groundtruth_filename)
        y_truth = y_truth[0:(len(y_truth)-(len(y_truth)%batch_size))]
        cnf_freq = 1
        ##################################################################  
        # early-stopping parameters
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

        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()

        finetune_lr_initial = finetune_lr

        done_looping = False
        epoch = 0
        flag = open(prefix+'flag.pkl','wb')
        cPickle.dump(2,flag, protocol = cPickle.HIGHEST_PROTOCOL)
        flag.close()
        
        log_valid_cost=[]
        adapt_counter = 0
        while (epoch < training_epochs) and (not done_looping):
            
    #        if epochFlag_fineTuning is 1 and epoch < epochs_done_fineTuning:
    #            epoch = epochs_done_fineTuning
    #            epochFlag_fineTuning = 0
                
            epoch = epoch + 1
            ################################confusion matrix block 2#################
            if epoch%cnf_freq==0:
                pred_c = np.array([])
                for minibatch_index in xrange(n_train_batches):
                    pred_c = np.concatenate([pred_c,np.array(prediction(minibatch_index))])
            
                cnf_matrix = confusion_matrix(y_truth, pred_c)
                print cnf_matrix
            ##########################################################################
            c = []
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_fn(index=minibatch_index,lr=finetune_lr)
                c.append(minibatch_avg_cost)
    #            if iterFlag is 1 and iter < iters_done:
    #                iter = iters_done
    #                iterFlag = 0
                        
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))
                    log_valid_cost.append(this_validation_loss)

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
                        genVariables = [epoch,best_validation_loss,finetune_lr,patience,iter]
                        save_file = open(prefix+'fine_tuning.pkl','wb')
                        cPickle.dump(hidden_layers_sizes, save_file)
                        cPickle.dump(genVariables, save_file)
                        for j in xrange(len(sda.params)):
                            cPickle.dump(sda.params[j].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)
                        save_file.close()
                        
                        
                    
                        # test it on the test set
                        test_losses = test_model()
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))
                               
                        print 'Training cost: ', np.mean(c)
                    else:
                        adapt_counter = adapt_counter+1
                    if adapt_counter>20:
                        adapt_counter=0
                        finetune_lr = 0.8*finetune_lr
                        print 'Reduced learning rate : ', finetune_lr

                    else:
                        finetune_lr = finetune_lr_initial / (1 + epoch * 5e-05)
                        
                #if patience <= iter:
                #    done_looping = True
                #    break

        end_time = time.clock()
        print(
            (
                'Optimization complete with best validation score of %f %%, '
                'on iteration %i, '
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print >> sys.stderr, ('The training code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

        valid_file = open(prefix+'log_valid_error.txt', 'w')
        valid_file.write('Best validation error: '+str(best_validation_loss*100))
        valid_file.write('\nBest test error: '+str(test_score*100))
        valid_file.close()
        finetune_log_file = open(prefix + 'log_finetune_cost.txt', "a")
        for l in log_valid_cost:
            finetune_log_file.write("%f\n"%l)
        finetune_log_file.close()

if __name__ == '__main__':
    test_SdA()
