class Network(object):
    
    def __init__(self):
        self.logging = True

        # Bookkeeping
        self.iteration = 0

    def _reset_bookkeeping(self):
        self.iteration = 0

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.activate(p[0]))

    def get_single_error(self, output, targets):
        error = 0.0       
        for k in range(len(targets)):
            error = error + (targets[k]-output[k])**2
        return error
        
    def get_error(self, patterns):
        error = 0.0        
        for pattern in patterns:
            inputs  = pattern[0]
            targets = pattern[1]
            output = self.activate(inputs)
            error = error + self.get_single_error(output, targets)  
        error = error/len(patterns)         
        return error

    def random_train(self, patterns, N, M):
        error = 0.0
        for i in range(len(patterns)):
            p = patterns[random.randint(0, len(patterns)-1)]
            inputs = p[0]
            targets = p[1]
            self.update(inputs)
            error = error + self.backPropagate(targets, N, M)
        return error
        
    def iterative_train(self, patterns, N, M):
        error = 0.0
        #random.shuffle(pattern_indexes)
        #for index in pattern_indexes:
            #p = patterns[index]
        for p in patterns:
            inputs = p[0]
            targets = p[1]
            self.update(inputs)
            error = error + self.backPropagate(targets, N, M)
        return error
        
    def shuffle_train(self, patterns, N, M):
        error = 0.0
        pattern_indexes = range(len(patterns))
        pattern_indexes = pattern_indexes
        random.shuffle(pattern_indexes)
        for index in pattern_indexes:
            p = patterns[index]
            inputs = p[0]
            targets = p[1]
            self.update(inputs)
            error = error + self.backPropagate(targets, N, M)
        return error
    
    def train(self, patterns, iterations=1000, error_break=0.02, *args, **kwargs):
        self._reset_bookkeeping()
        self.reset()

        for self.iteration in range(iterations):            
            for pattern in patterns:
                self.learn(pattern[0], pattern[1], *args, **kwargs)
                
            if error_break != 0.0 or self.logging:
                error = self.get_error(patterns)
                if self.logging:
                    print "Iteration {}, Error: {}".format(self.iteration, error)
                if error < error_break:
                    return

    ##################
    # Abstract methods
    ##################
    def activate(self, inputs):
        raise NotImplementedError()

    def learn(self, inputs, targets):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()