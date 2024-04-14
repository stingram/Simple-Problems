Let us assume that you have a DNN model with N parameters. Implement library methods to aggregate gradients from multiple workers.

When backward pass starts, framework will call backward_init() 
When every gradient becomes available on backward pass, framework will call on_gradient_available() for each parameter in the model 
You have API send_gradient(buffer, num_gradients, receiver) to send gradients to another worker and have a 
callback onRecvGradient(buffer, num_gradients, sender) to know when you receive sent gradients from other workers.
Every worker should have aggregated gradients for all parameters at end of backward pass and call makeAggregatedGradientsAvailable() 
(decide the params) to let framework know that final gradients are available, so that framework can start the next forward pass.

 

# Helper methods which you can use in your implementation

makeAggregatedGradientsAvailable(averaged_gradients: list)

send_gradient(grad_buffer, num_gradients, receiver)

int my_worker_id() #get your worker ID
get_worker_list() # get list of workers





# Class you need to implement here
class GradientAggregator:

#init required member variables here

    def __ init __ ():
        deep_learning_framework.register_callback(self.backward_init, self.on_gradient_available)
        network_manager.register_recv_callback(self.on_recv)
        #logic here
        self.aggregator_rank = 0
        self._params_received = {}
        self.model_size = MODEL_SIZE
        self.world_size = WORLD_SIZE
        self.from_aggregator = set()
        

    def backward_init(self, ordered_parameter_list: list, worker_list):
        #logic here
        # forward done, backward about it done
        # parameters for which we have gradients: ordered_parameter_list
        # TODO - 2d initialization (ZERO OUT GRADIENT)
        self.grad_buffer = [len(ordered_parameter_list)]*len(worker_list())
        self.recevied_comm = set()

    def on_gradient_available(self, param, grad):
        
        #logic here, for every parameter that has a gradient that has been computed
        if my_worked_id() != self.aggregator_rank:
            send_gradient(param,grad,1,self.aggregator_rank)
            
        # TODO - aggregator rank
        


    def on_recv(param, gradient, sender):
        #logic here
        # combine gradients from all the other GPUs
        for _ in range(num_gradients):
            self.grad_buffer[param] += gradient
        
        if sender not in self.params_received:
            self.params_received[sender] = 0
        self.params_received[sender] += 1
        
        # logic for aggregator
        if self.params_received[sender] == self.model_size:
            if my_worker_id() == self.aggregator_rank
                # only after all the params from this sender was provided
                self.received_comm.add(sender)
                
                # now check all others 
                if len(self.received_comm) = self.world_size -1:
                    # now we broadcast
                    for receiver in get_worker_list():
                        # send whole block to each sequentially
                        if receiver != self.aggregator_rank
                            send_gradient(param,grad,1,receiver) # TODO, SEND WHOLE BLOCK, NOT just 1 at a time
        
        

given 4 gpus, then 3 of them could send all params  to GPU0         

